"""Evaluate LightGBM quantile forecasts: pinball loss and interval coverage.
Run: python -m m5.evaluation.quantile_eval --store CA_1
"""
import argparse

import numpy as np
import pandas as pd

from m5.config import FOLDS
from m5.data.load import load_sales
from m5.evaluation.metrics import coverage, pinball_loss
from m5.models.lgbm_quantile import QuantileLightGBMForecaster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["fold3"])
    parser.add_argument("--store", default=None)
    parser.add_argument("--train-days", type=int, default=365)
    args = parser.parse_args()

    meta, Y, _ = load_sales()
    if args.store:
        mask = (meta["store_id"].astype(str) == args.store).to_numpy()
        meta, Y = meta.loc[mask].reset_index(drop=True), Y[mask]

    model = QuantileLightGBMForecaster(train_days=args.train_days)
    model.bind(meta)

    rows = []
    for fold_name in args.folds:
        fold = FOLDS[fold_name]
        Y_train = Y[:, : fold.train_end]
        Y_true = Y[:, fold.valid_slice]

        preds = model.predict_quantiles(Y_train, 28)
        crossing = ((preds[0.1] > preds[0.5]) | (preds[0.5] > preds[0.9])).sum()
        total = preds[0.1].size
        print(f"  quantile crossing: {crossing}/{total} points ({crossing/total:.4%})")
        lo, mid, hi = preds[0.1], preds[0.5], preds[0.9]

        row = {"fold": fold_name}
        for q, pred in preds.items():
            row[f"pinball_{q}"] = pinball_loss(Y_true, pred, q)
        row["coverage_80"] = coverage(Y_true, lo, hi)  # nominal 80% interval (0.1 to 0.9)
        row["mean_interval_width"] = float(np.mean(hi - lo))
        rows.append(row)
        print(f"{fold_name}: pinball(0.1)={row['pinball_0.1']:.3f} "
              f"pinball(0.5)={row['pinball_0.5']:.3f} pinball(0.9)={row['pinball_0.9']:.3f} "
              f"coverage_80={row['coverage_80']:.3f} (target: 0.80)")

    out = pd.DataFrame(rows)
    out.to_csv("results/quantile_eval.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()