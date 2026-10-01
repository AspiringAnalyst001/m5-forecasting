"""Break down WRMSSE by aggregation level to see where the model wins or loses.
Run: python -m m5.evaluation.level_analysis
"""
import pandas as pd

LEVEL_NAMES = [
    "total", "state", "store", "category", "department", "state+category",
    "state+department", "store+category", "store+department", "item",
    "item+state", "item+store",
]


def main() -> None:
    df = pd.read_csv("results/backtest_all.csv")
    dupes = df[df.duplicated(subset=["model", "fold"], keep=False)]
    if len(dupes):
        print("WARNING: duplicate (model, fold) rows found, keeping the last:")
        print(dupes[["model", "fold", "wrmsse"]].to_string(index=False))
        df = df.drop_duplicates(subset=["model", "fold"], keep="last")

    fold3 = df[df["fold"] == "fold3"].set_index("model")
    level_cols = [f"level_{i}" for i in range(1, 13)]

    models = ["seasonal_mean_8w", "mean_28", "lgbm_fast", "lgbm_v1"]
    models = [m for m in models if m in fold3.index]

    table = fold3.loc[models, level_cols].T
    table.index = LEVEL_NAMES
    pd.set_option("display.width", 200)
    print("\nWRMSSE by aggregation level (fold3, full data):\n")
    print(table.round(3))

    if "seasonal_mean_8w" in models and "lgbm_fast" in models:
        baseline = fold3.loc["seasonal_mean_8w", level_cols].astype(float).to_numpy()
        model_vals = fold3.loc["lgbm_fast", level_cols].astype(float).to_numpy()
        improvement = (baseline - model_vals) / baseline
        imp = pd.Series(improvement, index=LEVEL_NAMES, name="pct_improvement")
        print("\nLightGBM's improvement over seasonal_mean_8w, by level:\n")
        print((imp * 100).round(1).map(lambda x: f"{x}%"))


if __name__ == "__main__":
    main()