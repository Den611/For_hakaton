#!/usr/bin/env python3
"""Квартальні дані ВООЗ (mpox, AFRO) + прогноз Facebook Prophet -> data.json для карти.

    pip install prophet pandas numpy
    python forecast.py AFRO_only.json            # -> data.json
    python forecast.py AFRO_only.json --end 2030 --csv result.csv

Врахування неповноти даних:
  * квартали БЕЗ звітів (і з повнотою < --min-coverage) виключаються з навчання як пропуски,
    а не вважаються нулями; «звіт із нулем» лишається справжнім нулем;
  * останні --drop-last кварталів (1) вважаються ПОПЕРЕДНІМИ: показуються, але не беруть
    участі в навчанні (звіти за них ще доповнюються);
  * для кожного кварталу рахується повнота звітів (частка тижнів, за які є рядок ВООЗ):
    «звіт із нулем» і «звіту немає» розрізняються у coverage;
  * бектест (rolling origin, 1 квартал уперед) порівнює реальні значення з інтервалом
    Prophet і, якщо покриття нижче заявленого, розширює інтервали на емпіричний коефіцієнт.
    Це враховує нестабільність звітування/тестування лише в тій мірі, в якій вона видна в
    історії; даних про обсяги тестування чи геноми в наборі немає, тож модель їх не бачить.
  * індикатор зсуву частки смертей до випадків (cfr_shift) — лише СИГНАЛ можливої зміни
    ураховуваності/тестування, а не доказ і не поправка до прогнозу.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ---------- дані ----------
def load_weekly(path, include_probable):
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw["value"] if isinstance(raw, dict) else raw
    df = pd.DataFrame(rows)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = (df.sort_values(["ISO3", "DATE"])
            .drop_duplicates(["ISO3", "DATE"], keep="last")
            .reset_index(drop=True))
    diff = df.groupby("ISO3")["TOTAL_CONF_CASES"].diff().fillna(df["TOTAL_CONF_CASES"])
    new = df["NEW_CONF_CASES"].fillna(diff)           # пропуски -> різниця накопичених
    if include_probable:
        new = new + df["NEW_PROB_CASES"].fillna(0)
    df["NEW"] = new.clip(lower=0).fillna(0)
    df["DEATHS"] = (df["NEW_CONF_DEATHS"].fillna(0).clip(lower=0)
                    if "NEW_CONF_DEATHS" in df else 0.0)
    df["Q"] = df["DATE"].dt.to_period("Q")
    return df


def quarterly_series(df):
    """{ISO3: (cases, coverage, перший_квартал_звіту, deaths)}.
    cases/deaths: 0 там, де звітів немає (відрізняти їх треба за coverage);
    coverage: частка очікуваних тижнів, за які рядок є (0 = взагалі не звітувала)."""
    first_q, last_q, last_date = df["Q"].min(), df["Q"].max(), df["DATE"].max()
    idx = pd.period_range(first_q, last_q, freq="Q")
    weeks = pd.Series(
        [((min(q.end_time.normalize(), last_date) - q.start_time).days + 1) / 7 for q in idx],
        index=idx)
    out = {}
    for iso, g in df.groupby("ISO3"):
        cases = g.groupby("Q")["NEW"].sum().reindex(idx, fill_value=0.0)
        cov = (g.groupby("Q").size().reindex(idx, fill_value=0) / weeks).clip(upper=1.0)
        deaths = g.groupby("Q")["DEATHS"].sum().reindex(idx, fill_value=0.0)
        out[iso] = (cases, cov, g["Q"].min(), deaths)
    return out, first_q, last_q


def cfr_shift(cases, deaths, cov, min_n=20, min_prior=50, min_ratio=2.0):
    """Частка смертей до випадків по кварталах + сигнал різкого зсуву проти попередніх кварталів.
    Евристика (нормальне наближення 95% + зміна щонайменше вдвічі), а не статистичний доказ."""
    cfr, flag = {}, {}
    pc = pd_ = 0.0
    for q in cases.index:
        n, d = float(cases[q]), float(deaths[q])
        cfr[q], flag[q] = None, None
        if cov[q] <= 0:
            continue
        if n >= min_n:
            cfr[q] = round(100 * d / n, 1)
            if pc >= min_prior:
                p0 = (pd_ + 0.5) / (pc + 1)
                exp_d = n * p0
                outside = abs(d - exp_d) > 1.96 * np.sqrt(n * p0 * (1 - p0)) + 1
                ratio = (d + 0.5) / (exp_d + 0.5)
                flag[q] = int(outside and (ratio >= min_ratio or ratio <= 1 / min_ratio))
        pc += n
        pd_ += d
    return cfr, flag


# ---------- модель ----------
def n_train(s, first_rep, drop_last):
    """Скільки кварталів із реальними звітами піде в навчання (без попередніх останніх)."""
    settled = s.iloc[:len(s) - drop_last] if drop_last else s
    return int(settled[first_rep:].notna().sum())


def to_model(x, use_log):
    x = np.asarray(x, dtype=float)
    return np.log1p(x) if use_log else x


def from_model(x, use_log):
    x = np.expm1(x) if use_log else x
    return np.clip(x, 0, None)


def fit_predict(train, future, use_log, width, cap_factor, cps=0.05):
    """Prophet на ряді train (Series з PeriodIndex) -> (yhat, lo, hi) у просторі моделі."""
    train = train.dropna()          # квартали без/з неповними звітами — пропуски, не нулі
    if train.nunique() == 1:
        v = np.full(len(future), to_model(train.iloc[-1], use_log))
        return v, v.copy(), v.copy()

    from prophet import Prophet

    df = pd.DataFrame({"ds": train.index.start_time, "y": to_model(train.values, use_log)})
    logistic = cap_factor is not None
    if logistic:  # стеля, щоб на довгому горизонті тренд не «вибухав»
        cap = float(to_model(train.max() * cap_factor, use_log))
        df["cap"], df["floor"] = cap, 0.0
    m = Prophet(growth="logistic" if logistic else "linear", changepoint_prior_scale=cps,
                yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
                interval_width=width, uncertainty_samples=500)
    m.fit(df)
    fut = pd.DataFrame({"ds": future.start_time})
    if logistic:
        fut["cap"], fut["floor"] = cap, 0.0
    p = m.predict(fut)
    return p["yhat"].values, p["yhat_lower"].values, p["yhat_upper"].values


def backtest_ratios(items, drop_last, min_points, use_log, width, cap_factor, origins, cps=0.05):
    """|похибка| / півширина інтервалу для 1-квартального прогнозу з кількох минулих точок."""
    ratios = []
    for iso, (s, _, first_rep) in items:
        settled = s.iloc[:len(s) - drop_last] if drop_last else s
        for m in range(max(1, len(settled) - origins), len(settled)):
            train = settled.iloc[:m]
            if train[first_rep:].notna().sum() < min_points or np.isnan(settled.iloc[m]):
                continue
            yh, lo, hi = fit_predict(train, settled.index[m:m + 1], use_log, width, cap_factor, cps)
            hw = (hi[0] - lo[0]) / 2
            if hw < 1e-6:
                continue
            ratios.append(abs(to_model(settled.iloc[m], use_log) - yh[0]) / hw)
    return np.array(ratios)


# ---------- основна програма ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="data.json")
    ap.add_argument("--start", type=int, default=None, help="перший рік осі (за замовчуванням — початок даних)")
    ap.add_argument("--end", type=int, default=2030, help="останній рік прогнозу (2030)")
    ap.add_argument("--min-points", type=int, default=6, help="мін. кварталів від першого звіту (6)")
    ap.add_argument("--interval", type=float, default=0.8, help="ширина інтервалу (0.8)")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="квартали з меншою часткою звітних тижнів виключаються з навчання (0.5)")
    ap.add_argument("--assume-zero", choices=["none", "trailing", "all"], default="none",
                    help="квартали без звітів: none — пропуски й сірі (за замовч.); "
                         "trailing — після останнього звіту країни рахувати як 0 (спалах завершився); "
                         "all — усі квартали без звітів як 0. Це ПРИПУЩЕННЯ: у підказці воно позначене")
    ap.add_argument("--trailing-zero", action="store_true", help="те саме, що --assume-zero trailing")
    ap.add_argument("--drop-last", type=int, default=1,
                    help="скільки останніх кварталів вважати попередніми й не навчати на них (1)")
    ap.add_argument("--no-backtest", action="store_true", help="не калібрувати інтервали бектестом")
    ap.add_argument("--origins", type=int, default=3, help="скільки точок бектесту на країну (3)")
    ap.add_argument("--max-widen", type=float, default=3.0, help="максимальний коефіцієнт розширення (3)")
    ap.add_argument("--cap", type=float, default=1.5, help="стеля = частка × історичний максимум (1.5)")
    ap.add_argument("--no-cap", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--include-probable", action="store_true")
    ap.add_argument("--params", help="best_params.json з tune.py")
    ap.add_argument("--csv")
    a = ap.parse_args()
    use_log, cap_factor, cps = not a.no_log, (None if a.no_cap else a.cap), 0.05
    if a.params:                      # результат tune.py застосовується лише якщо він себе виправдав
        bp = json.load(open(a.params))
        if bp.get("applied"):
            cps, cap_factor = bp["cps"], bp["cap_factor"]
            print(f"Параметри з {a.params}: changepoint_prior_scale={cps}, "
                  f"cap={cap_factor}", file=sys.stderr)
        else:
            print(f"{a.params}: тюнінг не дав вигоди на перевірочних кварталах — типові параметри",
                  file=sys.stderr)

    df = load_weekly(a.input, a.include_probable)
    series, data_q, last_q = quarterly_series(df)
    start_q = data_q if a.start is None else min(pd.Period(f"{a.start}Q1"), data_q)
    end_q = pd.Period(f"{a.end}Q4")
    horizon = (end_q - last_q).n
    if horizon < 1:
        sys.exit(f"--end {a.end} не пізніше за останні дані ({last_q})")
    quarters = pd.period_range(start_q, end_q, freq="Q")
    pos = {q: i for i, q in enumerate(quarters)}
    n = len(quarters)
    forecast_from, data_from = pos[last_q] + 1, pos[data_q]
    provisional_from = forecast_from - a.drop_last if a.drop_last else None

    raw = dict(series)
    if a.trailing_zero:
        a.assume_zero = "trailing"
    items, assumed = [], {}
    for iso, (c, cv, fr, _) in sorted(raw.items()):
        m = c.where(cv >= a.min_coverage)
        no_rep = cv == 0
        if a.assume_zero == "trailing":
            mask = no_rep & (cv.index > cv[cv > 0].index.max())
        elif a.assume_zero == "all":
            mask = no_rep
        else:
            mask = no_rep & False
        m[mask] = 0.0                       # у навчанні — нуль; без режиму лишається пропуском
        assumed[iso] = mask
        items.append((iso, (m, cv, fr)))
    eligible = [(iso, v) for iso, v in items if n_train(v[0], v[2], a.drop_last) >= a.min_points]

    # --- калібрування інтервалів ---
    widen, bt = 1.0, {"n": 0}
    if not a.no_backtest:
        print(f"Бектест: {len(eligible)} країн × до {a.origins} точок…", file=sys.stderr)
        r = backtest_ratios(eligible, a.drop_last, a.min_points, use_log, a.interval, cap_factor, a.origins, cps)
        bt["n"] = int(r.size)
        if r.size >= 10:
            bt["coverage"] = round(float((r <= 1).mean()), 2)
            widen = float(min(max(1.0, np.quantile(r, a.interval)), a.max_widen))
        print(f"  точок: {r.size}; покриття інтервалу: {bt.get('coverage', 'н/д')} "
              f"(ціль {a.interval}); коефіцієнт розширення: {widen:.2f}", file=sys.stderr)
    bt["widen_factor"] = round(widen, 2)

    # --- прогноз ---
    data, lower, upper, errors, coverage, quality = {}, {}, {}, {}, {}, {}
    cfr_out, shift_out = {}, {}
    skipped, rows, forecasted = [], [], []
    for k, (iso, (s, cov, first_rep)) in enumerate(items, 1):
        cases_raw, _, _, deaths = raw[iso]
        cf, sh = cfr_shift(cases_raw, deaths, cov)
        v, lo, hi, er, cv, cf_l, sh_l = ([None] * n for _ in range(7))
        for q in s.index:
            i = pos[q]
            cv[i], cf_l[i], sh_l[i] = round(float(cov[q]), 2), cf[q], sh[q]
            if cov[q] > 0 or assumed[iso][q]:    # є звіт (навіть 0) або прийнятий нуль
                x = int(round(cases_raw[q])) if cov[q] > 0 else 0
                v[i] = lo[i] = hi[i] = x
                er[i] = 0
                rows.append((iso, str(q), x, None, None, False))

        active = n_train(s, first_rep, a.drop_last)
        if active >= a.min_points:
            forecasted.append(iso)
            print(f"[{k}/{len(items)}] {iso}: {active} кв. у навчанні -> прогноз", file=sys.stderr)
            settled = s.iloc[:len(s) - a.drop_last] if a.drop_last else s
            future = pd.period_range(last_q + 1, periods=horizon, freq="Q")
            yh, l, h = fit_predict(settled, future, use_log, a.interval, cap_factor, cps)
            l, h = yh - widen * (yh - l), yh + widen * (h - yh)
            yh, l, h = (from_model(z, use_log) for z in (yh, l, h))
            if not a.no_cap:     # стеля і для прогнозу, і для меж інтервалу (log-шкала їх надуває)
                top = np.nanmax(s.values) * a.cap
                yh, l, h = np.minimum(yh, top), np.minimum(l, top), np.minimum(h, top)
            for j in range(horizon):
                i = forecast_from + j
                v[i], lo[i], hi[i] = int(round(yh[j])), int(round(l[j])), int(round(h[j]))
                er[i] = int(round((h[j] - l[j]) / 2))
                rows.append((iso, str(quarters[i]), v[i], lo[i], hi[i], True))
        else:
            skipped.append(f"{iso} ({active} кв.)")

        c_act = cov[first_rep:]
        quality[iso] = {"quarters_in_training": active,
                        "excluded_no_report": int((c_act == 0).sum()),
                        "excluded_low_coverage": int(((c_act > 0) & (c_act < a.min_coverage)).sum()),
                        "cfr_shift_quarters": int(sum(1 for x in sh.values() if x == 1))}
        data[iso], lower[iso], upper[iso], errors[iso], coverage[iso] = v, lo, hi, er, cv
        cfr_out[iso], shift_out[iso] = cf_l, sh_l

    vals = [x for arr in data.values() for x in arr if x is not None]
    payload = {
        "disease": "Мавпяча віспа, Африка",
        "unit": "підтверджених випадків за квартал",
        "years": [f"{q.year} Q{q.quarter}" for q in quarters],
        "data_from": data_from, "provisional_from": provisional_from, "forecast_from": forecast_from,
        "max": max(vals) if vals else 1,
        "data": data, "errors": errors, "lower": lower, "upper": upper, "coverage": coverage,
        "cfr": cfr_out, "cfr_shift": shift_out, "quality": quality,
        "meta": {"model": "Facebook Prophet", "log1p": use_log, "cap_factor": cap_factor,
                 "interval": a.interval, "changepoint_prior_scale": cps, "drop_last": a.drop_last, "min_coverage": a.min_coverage, "assume_zero": a.assume_zero, "last_observed": str(last_q),
                 "backtest": bt, "no_forecast": skipped, "forecasted": forecasted, "min_points": a.min_points,
                 "includes_probable": a.include_probable},
    }
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, allow_nan=False)
    if a.csv:
        pd.DataFrame(rows, columns=["iso3", "quarter", "cases", "lower", "upper", "is_forecast"]
                     ).to_csv(a.csv, index=False)
    print(f"\nГотово: {a.output}. Остання чверть у даних: {last_q} (попередні: {a.drop_last}), "
          f"прогноз до {a.end} Q4.", file=sys.stderr)
    print(f"Прогноз є для {len(forecasted)} країн (поріг {a.min_points} кв. у навчанні): "
          + ", ".join(forecasted), file=sys.stderr)
    if skipped:
        print("Без прогнозу (замало історії): " + ", ".join(skipped), file=sys.stderr)


if __name__ == "__main__":
    main()