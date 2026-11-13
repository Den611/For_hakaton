import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def load_weekly(path, include_probable):
    raw = json.load(open(path, encoding="utf-8"))
    rows = raw["value"] if isinstance(raw, dict) else raw
    df = pd.DataFrame(rows)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = (df.sort_values(["ISO3", "DATE"])
            .drop_duplicates(["ISO3", "DATE"], keep="last")
            .reset_index(drop=True))

    diff = df.groupby("ISO3")["TOTAL_CONF_CASES"].diff()
    diff = diff.fillna(df["TOTAL_CONF_CASES"])
    new = df["NEW_CONF_CASES"].fillna(diff)
    if include_probable:
        new = new + df["NEW_PROB_CASES"].fillna(0)
    df["NEW"] = new.clip(lower=0).fillna(0)
    df["Q"] = df["DATE"].dt.to_period("Q")
    return df


def quarterly_series(df):
    first_q, last_q = df["Q"].min(), df["Q"].max()
    idx = pd.period_range(first_q, last_q, freq="Q")
    out = {}
    for iso, g in df.groupby("ISO3"):
        s = g.groupby("Q")["NEW"].sum()
        out[iso] = (s.reindex(idx, fill_value=0.0), s.index.min())
    return out, first_q, last_q


def prophet_forecast(s, horizon, use_log, width, cap_factor):
    if s.nunique() == 1:                  
        v = np.full(horizon, float(s.iloc[-1]))
        return v, v.copy(), v.copy()

    from prophet import Prophet

    y = np.log1p(s.values) if use_log else s.values.astype(float)
    train = pd.DataFrame({"ds": s.index.start_time, "y": y})
    logistic = cap_factor is not None
    if logistic:
        top = s.max() * cap_factor
        cap = float(np.log1p(top) if use_log else top)
        train["cap"], train["floor"] = cap, 0.0
    m = Prophet(
        growth="logistic" if logistic else "linear",
        yearly_seasonality=False,   
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=width,
        uncertainty_samples=500,
    )
    m.fit(train)
    future = pd.period_range(s.index[-1] + 1, periods=horizon, freq="Q")
    fut = pd.DataFrame({"ds": future.start_time})
    if logistic:
        fut["cap"], fut["floor"] = cap, 0.0
    p = m.predict(fut)
    yhat, lo, hi = p["yhat"].values, p["yhat_lower"].values, p["yhat_upper"].values
    if use_log:
        yhat, lo, hi = np.expm1(yhat), np.expm1(lo), np.expm1(hi)
    return (np.clip(yhat, 0, None), np.clip(lo, 0, None), np.clip(hi, 0, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="AFRO_only.json")
    ap.add_argument("-o", "--output", default="data.json")
    ap.add_argument("--start", type=int, default=None,
                    help="перший рік осі часу (за замовчуванням — перший квартал даних)")
    ap.add_argument("--end", type=int, default=2030, help="останній рік осі часу / прогнозу (2030)")
    ap.add_argument("--min-points", type=int, default=6,
                    help="мінімум кварталів від першого звіту країни для прогнозу (6)")
    ap.add_argument("--interval", type=float, default=0.8, help="ширина інтервалу (0.8)")
    ap.add_argument("--cap", type=float, default=1.5,
                    help="стеля прогнозу = ця частка × історичний максимум за квартал (1.5)")
    ap.add_argument("--no-cap", action="store_true", help="лінійний ріст без стелі")
    ap.add_argument("--no-log", action="store_true",
                    help="не застосовувати log1p (за замовчуванням застосовується)")
    ap.add_argument("--include-probable", action="store_true",
                    help="додати ймовірні випадки до підтверджених")
    ap.add_argument("--csv", help="додатково зберегти довгу таблицю у CSV")
    a = ap.parse_args()

    df = load_weekly(a.input, a.include_probable)
    series, data_q, last_q = quarterly_series(df)
    start_q = data_q if a.start is None else min(pd.Period(f"{a.start}Q1"), data_q)
    end_q = pd.Period(f"{a.end}Q4")
    horizon = (end_q - last_q).n
    if horizon < 1:
        sys.exit(f"--end {a.end} не пізніше за останні дані ({last_q})")
    quarters = pd.period_range(start_q, end_q, freq="Q")
    pos = {q: i for i, q in enumerate(quarters)}
    n, forecast_from, data_from = len(quarters), pos[last_q] + 1, pos[data_q]
    use_log = not a.no_log

    data, lower, upper, errors = {}, {}, {}, {}
    skipped, rows = [], []
    for k, (iso, (s, first_rep)) in enumerate(sorted(series.items()), 1):
        v = [None] * n
        lo = [None] * n
        hi = [None] * n
        er = [None] * n
        for q, x in s.items():
            i = pos[q]
            v[i] = lo[i] = hi[i] = int(round(x))
            er[i] = 0
            rows.append((iso, str(q), int(round(x)), None, None, False))

        active = len(s[first_rep:])
        if active >= a.min_points:
            print(f"[{k}/{len(series)}] {iso}: {active} кв. з першого звіту -> прогноз", file=sys.stderr)
            yh, l, h = prophet_forecast(s, horizon, use_log, a.interval, None if a.no_cap else a.cap)
            for j in range(horizon):
                i = forecast_from + j
                v[i], lo[i], hi[i] = int(round(yh[j])), int(round(l[j])), int(round(h[j]))
                er[i] = int(round((h[j] - l[j]) / 2))
                rows.append((iso, str(quarters[i]), v[i], lo[i], hi[i], True))
        else:
            skipped.append(f"{iso} ({active} кв.)")
        data[iso], lower[iso], upper[iso], errors[iso] = v, lo, hi, er

    vals = [x for arr in data.values() for x in arr if x is not None]
    payload = {
        "disease": "Mpox (мавпяча віспа), регіон AFRO",
        "unit": "підтверджених випадків за квартал",
        "years": [f"{q.year} Q{q.quarter}" for q in quarters],   
        "forecast_from": forecast_from, "data_from": data_from,
        "max": max(vals) if vals else 1,
        "data": data, "errors": errors, "lower": lower, "upper": upper,
        "meta": {"model": "Facebook Prophet", "log1p": use_log, "cap_factor": None if a.no_cap else a.cap, "interval": a.interval,
                 "last_observed": str(last_q), "no_forecast": skipped},
    }
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, allow_nan=False)
    if a.csv:
        pd.DataFrame(rows, columns=["iso3", "quarter", "cases", "lower", "upper", "is_forecast"]
                     ).to_csv(a.csv, index=False)

    print(f"\nГотово: {a.output}. Остання фактична чверть: {last_q}, "
          f"прогноз: {horizon} кв. (до {a.end} Q4)", file=sys.stderr)
    if skipped:
        print("Без прогнозу (замало історії): " + ", ".join(skipped), file=sys.stderr)


if __name__ == "__main__":
    main()