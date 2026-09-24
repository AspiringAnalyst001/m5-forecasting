"""Rolling-origin backtesting.

Examples:
    python -m m5.evaluation.backtest
    python -m m5.evaluation.backtest --models sf_tsb --store CA_1
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from m5.config import FOLDS, HORIZON, ROOT, Fold
from m5.data.load import load_calendar, load_prices, load_sales
from m5.evaluation.metrics import WRMSSE, last28_dollar_sales, mase, wape
from m5.models.baselines import Forecaster
from m5.models.registry import get_models

RESULTS_DIR = ROOT / "results"


def backtest_fold(forecaster: Forecaster, Y: np.ndarray, fold: Fold, wrmsse: WRMSSE) -> dict:
    """Forecast one fold. The forecaster only ever receives the training days."""
    Y_train = np.ascontiguousarray(Y[:, : fold.train_end])   # a copy: no path to the future
    Y_train.flags.writeable = False
    Y_true = Y[:, fold.valid_slice]

    start = time.perf_counter()
    Y_pred = np.asarray(forecaster.predict(Y_train, HORIZON), dtype=np.float32)
    seconds = time.perf_counter() - start

    if Y_pred.shape != Y_true.shape:
        raise ValueError(f"{forecaster.name}: got {Y_pred.shape}, expected {Y_true.shape}")
    Y_pred = np.clip(np.nan_to_num(Y_pred), 0, None)

    row = {
        "model": forecaster.name,
        "fold": fold.name,
        "wrmsse": wrmsse.score(Y_true, Y_pred),
        "wape": wape(Y_true, Y_pred),
        "mase": mase(Y_true, Y_pred, Y_train),
        "seconds": seconds,
    }
    for i, value in enumerate(wrmsse.score_by_level(Y_true, Y_pred), start=1):
        row[f"level_{i}"] = value
    return row


def run_backtest(model_names, fold_names, store=None, allow_holdout=False) -> pd.DataFrame:
    if "holdout" in fold_names and not allow_holdout:
        raise ValueError("The holdout fold is for the final evaluation only (--allow-holdout).")
    models = get_models()
    unknown = set(model_names) - set(models)
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Available: {sorted(models)}")

    meta, Y, _ = load_sales()
    cal, prices = load_calendar(), load_prices()
    if store:   # dev mode: a smaller hierarchy, so scores are NOT comparable to full runs
        mask = (meta["store_id"].astype(str) == store).to_numpy()
        meta, Y = meta.loc[mask].reset_index(drop=True), Y[mask]

    for name in model_names:
        if hasattr(models[name], "bind"):
            models[name].bind(meta)

    rows = []
    for fold_name in fold_names:
        fold = FOLDS[fold_name]
        dollars = last28_dollar_sales(meta, Y, cal, prices, fold.train_end)
        wrmsse = WRMSSE(meta, Y[:, : fold.train_end], dollars)
        for name in model_names:
            row = backtest_fold(models[name], Y, fold, wrmsse)
            print(f"{fold_name} {name:20s} WRMSSE={row['wrmsse']:.3f} ({row['seconds']:.0f}s)")
            rows.append(row)
        del wrmsse
    return pd.DataFrame(rows)


def save_results(new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Merge into the results file, replacing any earlier rows for the same model+fold."""
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        replaced = set(zip(new["model"], new["fold"]))
        old = old[[(m, f) not in replaced for m, f in zip(old["model"], old["fold"])]]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["naive", "snaive_7", "mean_28", "seasonal_mean_8w"])
    parser.add_argument("--folds", nargs="+", default=["fold1", "fold2", "fold3"])
    parser.add_argument("--store", default=None, help="dev mode, e.g. CA_1")
    parser.add_argument("--allow-holdout", action="store_true")
    args = parser.parse_args()

    results = run_backtest(args.models, args.folds, args.store, args.allow_holdout)
    path = RESULTS_DIR / f"backtest_{args.store or 'all'}.csv"
    merged = save_results(results, path)
    pd.set_option("display.width", 200)

    folds_per_model = merged.groupby("model")["fold"].apply(set)
    common = sorted(set.intersection(*folds_per_model))
    if common:
        board = merged[merged["fold"].isin(common)]
        print(f"\nLeaderboard (mean over folds every model has: {common}):")
        print(board.groupby("model")[["wrmsse", "wape", "mase", "seconds"]]
              .mean().sort_values("wrmsse").round(3))
    print("\nWRMSSE by fold:")
    print(merged.pivot(index="model", columns="fold", values="wrmsse").round(3))
    print(f"\nSaved to {path}")

if __name__ == "__main__":
    main()