"""Baseline forecasters. Each takes Y_train (n_series, T) and returns (n_series, horizon)."""
from collections.abc import Callable
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    name: str

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray: ...


class Naive:
    name = "naive"

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        return np.repeat(Y_train[:, -1:], horizon, axis=1).astype(np.float32)


class SeasonalNaive:
    def __init__(self, season: int = 7):
        self.season = season
        self.name = f"snaive_{season}"

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        reps = -(-horizon // self.season)
        return np.tile(Y_train[:, -self.season :], reps)[:, :horizon].astype(np.float32)


class WindowAverage:
    def __init__(self, window: int = 28):
        self.window = window
        self.name = f"mean_{window}"

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        m = Y_train[:, -self.window :].mean(axis=1, dtype=np.float32)
        return np.repeat(m[:, None], horizon, axis=1)


class SeasonalWindowAverage:
    """Average of each weekday over the last N weeks (a strong, cheap baseline)."""

    def __init__(self, season: int = 7, weeks: int = 8):
        self.season, self.weeks = season, weeks
        self.name = f"seasonal_mean_{weeks}w"

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        n = Y_train.shape[0]
        block = Y_train[:, -self.weeks * self.season :].reshape(n, self.weeks, self.season)
        profile = block.mean(axis=1, dtype=np.float32)
        reps = -(-horizon // self.season)
        return np.tile(profile, reps)[:, :horizon]


class StatsForecastBaseline:
    """Wrap a statsforecast model. Uses only the last `history` days and skips pre-launch zeros."""

    def __init__(self, name: str, model_factory: Callable, history: int = 365, n_jobs: int = 2):
        self.name, self.model_factory = name, model_factory
        self.history, self.n_jobs = history, n_jobs

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        from statsforecast import StatsForecast

        window = Y_train[:, -self.history :].astype(np.float32)
        n, T = window.shape
        nonzero = window > 0
        first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), T - 1)
        start = np.minimum(first, T - 28)          # keep at least 28 points per series
        keep = np.arange(T)[None, :] >= start[:, None]
        rows, cols = np.nonzero(keep)

        df = pd.DataFrame(
            {
                "unique_id": rows.astype(np.int32),
                "ds": pd.Timestamp("2000-01-01") + pd.to_timedelta(cols, unit="D"),
                "y": window[rows, cols],
            }
        )
        sf = StatsForecast(models=[self.model_factory()], freq="D", n_jobs=self.n_jobs)
        out = sf.forecast(df=df, h=horizon)
        if "unique_id" not in out.columns:
            out = out.reset_index()
        out = out.sort_values(["unique_id", "ds"])
        col = [c for c in out.columns if c not in ("unique_id", "ds")][0]
        pred = out[col].to_numpy().reshape(n, horizon)
        return np.nan_to_num(np.clip(pred, 0, None)).astype(np.float32)


def sf_baseline(name: str, cls_name: str, **kwargs) -> StatsForecastBaseline:
    def factory():
        import statsforecast.models as sm

        return getattr(sm, cls_name)(**kwargs)

    return StatsForecastBaseline(name, factory)


def get_baselines() -> dict[str, Forecaster]:
    models = [
        Naive(),
        SeasonalNaive(7),
        WindowAverage(28),
        SeasonalWindowAverage(7, 8),
        sf_baseline("sf_tsb", "TSB", alpha_d=0.2, alpha_p=0.2),
        sf_baseline("sf_croston_opt", "CrostonOptimized"),
        sf_baseline("sf_auto_ets", "AutoETS", season_length=7),
    ]
    return {m.name: m for m in models}