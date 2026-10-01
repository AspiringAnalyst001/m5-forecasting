"""Per-department and per-intermittency error analysis for LightGBM vs baseline.
Run: python -m m5.evaluation.segment_analysis
"""
import numpy as np
import pandas as pd

from m5.config import FOLDS
from m5.data.load import load_sales
from m5.models.baselines import SeasonalWindowAverage
from m5.models.lgbm import LightGBMForecaster


def group_wape(y_true: np.ndarray, y_pred: np.ndarray, group: pd.Series) -> pd.DataFrame:
    """Aggregate WAPE per group: sum(|error|)/sum(|actual|) across all series+days
    in the group. This is what WRMSSE-style weighting approximates — a handful of
    near-zero series can't dominate the way they do in a mean of per-series ratios."""
    abs_err = np.abs(y_true - y_pred).sum(axis=1)   # per series, summed over 28 days
    abs_act = np.abs(y_true).sum(axis=1)
    df = pd.DataFrame({"group": group.to_numpy(), "abs_err": abs_err, "abs_act": abs_act})
    out = df.groupby("group").sum()
    out["wape"] = out["abs_err"] / out["abs_act"]
    return out[["wape"]]


def main() -> None:
    meta, Y, _ = load_sales()
    fold = FOLDS["fold3"]
    Y_train, Y_true = Y[:, : fold.train_end], Y[:, fold.valid_slice]

    baseline = SeasonalWindowAverage(7, 8)
    base_pred = baseline.predict(Y_train, 28)

    model = LightGBMForecaster(name="lgbm_fast", train_days=365, num_rounds=200)
    model.bind(meta)
    lgbm_pred = model.predict(Y_train, 28)

    zero_share = (Y_train[:, -180:] == 0).mean(axis=1)
    cat = meta["cat_id"].astype(str)
    dept = meta["dept_id"].astype(str)
    intermittency = pd.cut(zero_share, [0, 0.3, 0.6, 0.8, 1.0],
                            labels=["low (<30%)", "moderate (30-60%)",
                                    "high (60-80%)", "very high (>80%)"])

    for name, group in [("category", cat), ("department", dept), ("intermittency", intermittency)]:
        base_g = group_wape(Y_true, base_pred, group).rename(columns={"wape": "wape_baseline"})
        lgbm_g = group_wape(Y_true, lgbm_pred, group).rename(columns={"wape": "wape_lgbm"})
        merged = base_g.join(lgbm_g)
        merged["improvement"] = (merged["wape_baseline"] - merged["wape_lgbm"]) / merged["wape_baseline"]
        print(f"\nAggregate WAPE by {name} (sum error / sum actual):\n")
        print(merged.sort_values("improvement", ascending=False).round(3))

    overall_base = group_wape(Y_true, base_pred, pd.Series(["all"] * len(Y_true)))
    overall_lgbm = group_wape(Y_true, lgbm_pred, pd.Series(["all"] * len(Y_true)))
    print(f"\nOverall aggregate WAPE: baseline={overall_base['wape'].iloc[0]:.3f}  "
          f"lgbm={overall_lgbm['wape'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()