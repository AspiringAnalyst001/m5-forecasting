"""Domain rules wrapped around any forecaster."""
import numpy as np
import pandas as pd

from m5.data.load import load_calendar


class ClosureRule:
    """Zero the forecast on calendar dates when sales collapsed in the past (e.g. Dec 25)."""

    def __init__(self, base, threshold: float = 0.1, dates: pd.Series | None = None):
        self.base, self.threshold, self.dates = base, threshold, dates
        self.name = f"{base.name}_closed"

    def bind(self, meta: pd.DataFrame) -> None:
        if hasattr(self.base, "bind"):
            self.base.bind(meta)
        if self.dates is None:
            self.dates = pd.to_datetime(load_calendar()["date"]).reset_index(drop=True)

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        pred = np.array(self.base.predict(Y_train, horizon), dtype=np.float32)
        T = Y_train.shape[1]
        totals = Y_train.sum(axis=0, dtype=np.int64)
        closed = totals < self.threshold * np.median(totals)

        past = self.dates[:T][closed]
        closed_dates = set(zip(past.dt.month, past.dt.day))
        future = self.dates[T : T + horizon]
        zero = np.array([(m, d) in closed_dates for m, d in zip(future.dt.month, future.dt.day)])
        pred[:, zero] = 0.0
        return pred