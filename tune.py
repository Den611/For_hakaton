import argparse
import itertools
import json
import sys

import numpy as np
import pandas as pd

import forecast as F

CPS = [0.01, 0.05, 0.1, 0.5]
GROWTH = [("logistic", 1.5), ("linear", None)]     
DEFAULT = (0.05, "logistic", 1.5)          


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default="best_params.json")
    ap.add_argument("--min-points", type=int, default=6)
    ap.add_argument("--min-coverage", type=float, default=0.5)
    ap.add_argument("--drop-last", type=int, default=1)
    ap.add_argument("--tune-origins", type=int, default=2)
    ap.add_argument("--test-origins", type=int, default=2)
    ap.add_argument("--interval", type=float, default=0.8)
    a = ap.parse_args()

    df = F.load_weekly(a.input, False)
    series, _, _ = F.quarterly_series(df)
    K = a.tune_origins + a.test_origins

    pairs = []
    for iso, (c, cv, fr, _) in sorted(series.items()):
        s = c.where(cv >= a.min_coverage)
        settled = s.iloc[:len(s) - a.drop_last] if a.drop_last else s
        for m in range(len(settled) - K, len(settled)):
            if m < 2 or np.isnan(settled.iloc[m]):
                continue
            train = settled.iloc[:m]
            if train[fr:].notna().sum() < a.min_points:
                continue
            grp = "tune" if m < len(settled) - a.test_origins else "test"
            pairs.append((iso, train, float(settled.iloc[m]), grp))
    n_t = sum(p[3] == "tune" for p in pairs)
    print(f"Точок: підбір {n_t}, перевірка {len(pairs) - n_t}; "
          f"конфігурацій {len(CPS) * len(GROWTH)}", file=sys.stderr)
    if n_t < 10 or len(pairs) - n_t < 10:
        sys.exit("Замало точок для чесного підбору — тюнінг не виконано.")

    def evaluate(cfg):
        cps, growth, cap = cfg
        res = {"tune": [], "test": []}
        for iso, train, y, grp in pairs:
            fut = pd.period_range(train.index[-1] + 1, periods=1, freq="Q")
            yh, _, _ = F.fit_predict(train, fut, True, a.interval, cap if growth == "logistic" else None, cps)
            res[grp].append((abs(np.log1p(y) - yh[0]), y - float(F.from_model(yh, True)[0])))
        return {g: {"log_mae": float(np.mean([e[0] for e in v])),
                    "rmse": float(np.sqrt(np.mean([e[1] ** 2 for e in v])))} for g, v in res.items()}

    results = {}
    for k, (cps, (growth, cap)) in enumerate(itertools.product(CPS, GROWTH), 1):
        cfg = (cps, growth, cap)
        print(f"[{k}/{len(CPS) * len(GROWTH)}] cps={cps} growth={growth}", file=sys.stderr)
        results[cfg] = evaluate(cfg)

    base = results.get(DEFAULT) or evaluate(DEFAULT)
    best = min(results, key=lambda c: results[c]["tune"]["log_mae"])   # вибір ТІЛЬКИ за "tune"
    bt, bb = results[best]["test"], base["test"]
    gain_mae = 1 - bt["log_mae"] / bb["log_mae"]
    gain_rmse = 1 - bt["rmse"] / bb["rmse"]
    applied = best != DEFAULT and gain_mae > 0

    print("\nКонфігурація                       tune log-MAE   test log-MAE   test RMSE", file=sys.stderr)
    for cfg, r in sorted(results.items(), key=lambda kv: kv[1]["tune"]["log_mae"]):
        mark = " <- обрано" if cfg == best else (" <- типова" if cfg == DEFAULT else "")
        print(f"cps={cfg[0]:<5} {cfg[1]:<9} {r['tune']['log_mae']:>12.3f} {r['test']['log_mae']:>14.3f}"
              f" {r['test']['rmse']:>11.1f}{mark}", file=sys.stderr)
    print(f"\nНа перевірочних точках: log-MAE {gain_mae:+.1%}, RMSE {gain_rmse:+.1%} "
          f"відносно типових параметрів. Застосовувати: {'так' if applied else 'ні'}.", file=sys.stderr)

    json.dump({"cps": best[0], "growth": best[1], "cap_factor": best[2], "applied": applied,
               "test_points": len(pairs) - n_t,
               "test_gain": {"log_mae": round(gain_mae, 3), "rmse": round(gain_rmse, 3)}},
              open(a.output, "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()