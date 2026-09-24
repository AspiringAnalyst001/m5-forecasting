"""Forecast accuracy metrics: WRMSSE (official M5 metric) and WAPE."""
import numpy as np
import pandas as pd
from scipy import sparse

from m5.data.load import load_calendar, load_prices, load_sales

# The 12 aggregation levels of M5, top to bottom
LEVELS: list[list[str]] = [
    [],
    ["state_id"],
    ["store_id"],
    ["cat_id"],
    ["dept_id"],
    ["state_id", "cat_id"],
    ["state_id", "dept_id"],
    ["store_id", "cat_id"],
    ["store_id", "dept_id"],
    ["item_id"],
    ["item_id", "state_id"],
    ["item_id", "store_id"],
]


def build_aggregation_matrix(meta: pd.DataFrame) -> tuple[sparse.csr_matrix, list[int]]:
    """Sparse matrix S of shape (n_all_series, n_bottom_series) and the size of each level."""
    n = len(meta)
    blocks, sizes = [], []
    for cols in LEVELS:
        if cols:
            codes = meta.groupby(cols, observed=True).ngroup().to_numpy()
        else:
            codes = np.zeros(n, dtype=np.int64)
        m = int(codes.max()) + 1
        block = sparse.csr_matrix(
            (np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(m, n)
        )
        blocks.append(block)
        sizes.append(m)
    return sparse.vstack(blocks).tocsr(), sizes


def compute_scales(Ya: np.ndarray) -> np.ndarray:
    """Mean squared one-step naive error per series, starting at the first non-zero sale."""
    nonzero = Ya > 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), Ya.shape[1] - 1)
    diff2 = np.diff(Ya, axis=1) ** 2
    cols = np.arange(diff2.shape[1])[None, :]
    mask = cols >= first[:, None]
    total = (diff2 * mask).sum(axis=1, dtype=np.float64)
    return total / np.maximum(mask.sum(axis=1), 1)


class WRMSSE:
    def __init__(self, meta: pd.DataFrame, Y_train: np.ndarray, dollars_last28: np.ndarray):
        self.S, self.sizes = build_aggregation_matrix(meta)
        Ya = self.S @ Y_train.astype(np.float32)
        self.scale = compute_scales(Ya)
        self.valid = self.scale > 0

        agg_dollars = self.S @ dollars_last28.astype(np.float64)
        weights = np.zeros(len(agg_dollars))
        start = 0
        for size in self.sizes:
            block = agg_dollars[start : start + size]
            weights[start : start + size] = block / block.sum() / len(self.sizes)
            start += size
        self.weights = weights

    def _rmsse(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
        # Aggregation is linear, so aggregate the errors directly
        err = self.S @ (Y_true.astype(np.float32) - Y_pred.astype(np.float32))
        mse = (err**2).mean(axis=1)
        safe_scale = np.where(self.valid, self.scale, 1.0)
        return np.where(self.valid, np.sqrt(mse / safe_scale), 0.0)

    def score(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        return float((self.weights * self._rmsse(Y_true, Y_pred)).sum())

    def score_by_level(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> list[float]:
        contrib = self.weights * self._rmsse(Y_true, Y_pred)
        out, start = [], 0
        for size in self.sizes:
            out.append(float(contrib[start : start + size].sum() * len(self.sizes)))
            start += size
        return out


def last28_dollar_sales(
    meta: pd.DataFrame, Y: np.ndarray, cal: pd.DataFrame, prices: pd.DataFrame, train_end: int
) -> np.ndarray:
    """Dollar sales per bottom series over the 28 days up to train_end (used as weights)."""
    days = [f"d_{d}" for d in range(train_end - 27, train_end + 1)]
    units = Y[:, train_end - 28 : train_end].astype(np.float32)
    weeks_per_day = cal.set_index("d").loc[days, "wm_yr_wk"].to_numpy()

    p = prices[prices["wm_yr_wk"].isin(np.unique(weeks_per_day))].copy()
    p["item_id"] = p["item_id"].astype(str)
    p["store_id"] = p["store_id"].astype(str)
    wide = p.pivot(index=["item_id", "store_id"], columns="wm_yr_wk", values="sell_price")

    idx = pd.MultiIndex.from_arrays(
        [meta["item_id"].astype(str), meta["store_id"].astype(str)]
    )
    price_mat = wide.reindex(idx)[list(weeks_per_day)].to_numpy(dtype=np.float32)
    return np.nan_to_num(units * price_mat).sum(axis=1)


def build_wrmsse(train_end: int) -> WRMSSE:
    """Build the metric for a given training cutoff (no future information used)."""
    meta, Y, _ = load_sales()
    dollars = last28_dollar_sales(meta, Y, load_calendar(), load_prices(), train_end)
    return WRMSSE(meta, Y[:, :train_end], dollars)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: total absolute error / total actual volume."""
    return float(np.abs(y_true - y_pred).sum() / np.abs(y_true).sum())


def mase(Y_true: np.ndarray, Y_pred: np.ndarray, Y_train: np.ndarray, season: int = 1) -> float:
    """Mean absolute scaled error, averaged over bottom-level series with a valid scale."""
    Yt = Y_train.astype(np.float32)
    nonzero = Yt > 0
    first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), Yt.shape[1] - 1)
    diffs = np.abs(Yt[:, season:] - Yt[:, :-season])
    mask = np.arange(diffs.shape[1])[None, :] >= first[:, None]
    scale = (diffs * mask).sum(axis=1, dtype=np.float64) / np.maximum(mask.sum(axis=1), 1)

    ok = scale > 0
    if not ok.any():
        return float("nan")
    mae = np.abs(Y_true.astype(np.float32) - Y_pred.astype(np.float32)).mean(axis=1)
    return float((mae[ok] / scale[ok]).mean())