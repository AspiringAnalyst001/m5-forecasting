"""Leak-free features. Every target-derived feature is shifted by >= 28 days (the horizon),
so one model can forecast all 28 days at once and no feature can see the future."""
import numpy as np
import pandas as pd

MIN_LAG = 28


def shift(A: np.ndarray, k: int) -> np.ndarray:
    """out[:, t] = A[:, t - k]; NaN where t < k."""
    out = np.full(A.shape, np.nan, dtype=np.float32)
    if k < A.shape[1]:
        out[:, k:] = A[:, :-k]
    return out


def _window_sums(A: np.ndarray, window: int, shift_by: int):
    """NaN-aware sum and count over A[:, t-shift_by-window+1 : t-shift_by+1] for every t."""
    assert shift_by >= MIN_LAG, "target features must be shifted by at least the horizon"
    n, T = A.shape
    valid = ~np.isnan(A)
    cs = np.zeros((n, T + 1), dtype=np.float64)
    cn = np.zeros((n, T + 1), dtype=np.int32)
    np.cumsum(np.where(valid, A, 0.0), axis=1, dtype=np.float64, out=cs[:, 1:])
    np.cumsum(valid, axis=1, dtype=np.int32, out=cn[:, 1:])

    b = np.arange(T) - shift_by            # inclusive end column
    a = np.maximum(b - window + 1, 0)      # inclusive start column (partial window at the start)
    ok = b >= 0
    s = np.zeros((n, T), dtype=np.float64)
    c = np.zeros((n, T), dtype=np.int32)
    s[:, ok] = cs[:, b[ok] + 1] - cs[:, a[ok]]
    c[:, ok] = cn[:, b[ok] + 1] - cn[:, a[ok]]
    return s, c


def rolling_mean_shifted(A, window, shift_by=MIN_LAG):
    s, c = _window_sums(A, window, shift_by)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan).astype(np.float32)


def rolling_std_shifted(A, window, shift_by=MIN_LAG):
    s, c = _window_sums(A, window, shift_by)
    s2, _ = _window_sums(A * A, window, shift_by)
    denom = np.maximum(c, 1)
    var = np.maximum(s2 / denom - (s / denom) ** 2, 0.0)
    return np.where(c > 1, np.sqrt(var), np.nan).astype(np.float32)


def weekday_mean_shifted(A, weeks=8, shift_by=MIN_LAG):
    """Mean of the same weekday over the last `weeks` weeks, starting `shift_by` days back."""
    total = np.zeros(A.shape, dtype=np.float64)
    count = np.zeros(A.shape, dtype=np.int32)
    for k in range(weeks):
        lagged = shift(A, shift_by + 7 * k)
        ok = ~np.isnan(lagged)
        total += np.where(ok, lagged, 0.0)
        count += ok
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def target_features(A: np.ndarray) -> dict[str, np.ndarray]:
    """A: (n_series, T) sales, NaN where the item is not on sale or the day is unknown."""
    with np.errstate(invalid="ignore"):
        Z = np.where(np.isnan(A), np.nan, (A > 0)).astype(np.float32)   # sold-that-day indicator
    feats = {f"lag_{k}": shift(A, k) for k in (28, 35, 42, 49, 364)}
    for w in (7, 28, 91):
        feats[f"rmean_{w}"] = rolling_mean_shifted(A, w)
    feats["rstd_28"] = rolling_std_shifted(A, 28)
    feats["nz_28"] = rolling_mean_shifted(Z, 28)
    feats["nz_91"] = rolling_mean_shifted(Z, 91)
    feats["dow_mean_8w"] = weekday_mean_shifted(A, 8)
    return feats


def calendar_arrays(cal: pd.DataFrame) -> dict[str, np.ndarray]:
    date = pd.to_datetime(cal["date"])
    out = {
        "dow": date.dt.dayofweek.to_numpy(dtype=np.int8),
        "dom": date.dt.day.to_numpy(dtype=np.int8),
        "month": date.dt.month.to_numpy(dtype=np.int8),
        # +1 so that "no event" is 0 rather than -1
        "event_1": (cal["event_name_1"].astype("category").cat.codes.to_numpy() + 1).astype(np.int16),
        "event_type_1": (cal["event_type_1"].astype("category").cat.codes.to_numpy() + 1).astype(np.int8),
    }
    for state in ("CA", "TX", "WI"):
        out[f"snap_{state}"] = cal[f"snap_{state}"].to_numpy(dtype=np.int8)
    return out


def price_matrix(prices_store: pd.DataFrame, item_ids: np.ndarray, week_of_day: np.ndarray) -> np.ndarray:
    """(n_items, n_days) daily price for one store; NaN where the item is not listed."""
    weeks = np.unique(week_of_day)
    P = np.full((len(item_ids), len(weeks)), np.nan, dtype=np.float32)
    pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    rows = prices_store["item_id"].astype(str).map(pos).to_numpy()
    price_weeks = prices_store["wm_yr_wk"].to_numpy()
    keep = ~np.isnan(rows) & np.isin(price_weeks, weeks)
    cols = np.searchsorted(weeks, price_weeks[keep])
    P[rows[keep].astype(np.int64), cols] = prices_store["sell_price"].to_numpy()[keep]
    return P[:, np.searchsorted(weeks, week_of_day)]


def price_features(P: np.ndarray) -> dict[str, np.ndarray]:
    on_sale = ~np.isnan(P)
    n, T = P.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        price_norm = P / np.fmax.accumulate(P, axis=1)     # price vs the highest price seen so far
        price_chg = P / shift(P, 7) - 1.0                   # change vs a week earlier
    first = np.where(on_sale.any(axis=1), on_sale.argmax(axis=1), T)
    age = (np.arange(T)[None, :] - first[:, None]).astype(np.float32)
    age[~on_sale] = np.nan
    return {
        "sell_price": P,
        "price_norm": price_norm.astype(np.float32),
        "price_chg_7d": price_chg.astype(np.float32),
        "age_days": age,
    }