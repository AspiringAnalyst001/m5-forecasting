"""Score two trivial forecasts on fold3 to verify the metric behaves sensibly."""
import numpy as np

from m5.config import FOLDS
from m5.data.load import load_sales
from m5.evaluation.metrics import build_wrmsse, wape

if __name__ == "__main__":
    fold = FOLDS["fold3"]
    _, Y, _ = load_sales()
    Y_true = Y[:, fold.valid_slice]

    naive = np.repeat(Y[:, fold.train_end - 1 : fold.train_end], 28, axis=1)
    snaive = np.tile(Y[:, fold.train_end - 7 : fold.train_end], 4)

    wrmsse = build_wrmsse(fold.train_end)
    for name, pred in [("naive", naive), ("seasonal naive (7d)", snaive)]:
        print(f"{name:22s} WRMSSE={wrmsse.score(Y_true, pred):.3f}  WAPE={wape(Y_true, pred):.3f}")
    print("snaive WRMSSE by level:", np.round(wrmsse.score_by_level(Y_true, snaive), 3))