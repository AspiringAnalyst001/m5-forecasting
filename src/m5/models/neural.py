"""NeuralForecast wrapper (N-HiTS etc.) that plugs into the same backtester.

Runs on a GPU machine (Colab/Kaggle): pip install -e ".[dl]"
Note: PatchTST does not support exogenous inputs, so use use_exog=False for it.
"""
import numpy as np
import pandas as pd

from m5.data.load import load_calendar, load_prices
from m5.features.build import price_features, price_matrix

FUTR_COLS = ["price_norm", "listed", "snap", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "event"]
STATES = ["CA", "TX", "WI"]


class NeuralForecastWrapper:
    def __init__(self, name, model_cls="NHITS", loss="mse", history=730, input_size=112,
                 max_steps=3000, use_exog=True, model_kwargs=None):
        self.name, self.model_cls, self.loss = name, model_cls, loss
        self.history, self.input_size, self.max_steps = history, input_size, max_steps
        self.use_exog = use_exog
        self.model_kwargs = model_kwargs or {}
        self.bound = False

    # ---- known-in-advance inputs, prepared once ----
    def bind(self, meta: pd.DataFrame) -> None:
        cal = load_calendar()
        date = pd.to_datetime(cal["date"]).reset_index(drop=True)
        self.dates = date.to_numpy()
        dow, doy = date.dt.dayofweek.to_numpy(), date.dt.dayofyear.to_numpy()
        self.cal = {
            "dow_sin": np.sin(2 * np.pi * dow / 7).astype(np.float32),
            "dow_cos": np.cos(2 * np.pi * dow / 7).astype(np.float32),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25).astype(np.float32),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25).astype(np.float32),
            "event": cal["event_name_1"].notna().to_numpy().astype(np.float32),
        }
        n, n_days = len(meta), len(cal)
        self.n_series = n

        state_code = pd.Categorical(meta["state_id"].astype(str), categories=STATES).codes
        snap_by_state = np.stack([cal[f"snap_{s}"].to_numpy(dtype=np.int8) for s in STATES])
        self.snap = snap_by_state[state_code]                      # (n, n_days)

        prices = load_prices()
        stores = meta["store_id"].astype(str).to_numpy()
        items = meta["item_id"].astype(str).to_numpy()
        week_of_day = cal["wm_yr_wk"].to_numpy()
        self.pnorm = np.full((n, n_days), np.nan, dtype=np.float32)
        self.listed = np.zeros((n, n_days), dtype=bool)
        for store in np.unique(stores):
            rows = np.flatnonzero(stores == store)
            P = price_matrix(prices[prices["store_id"] == store], items[rows], week_of_day)
            self.listed[rows] = ~np.isnan(P)
            self.pnorm[rows] = price_features(P)["price_norm"]
        del prices

        static = pd.get_dummies(meta[["dept_id", "store_id"]].astype(str)).astype(np.float32)
        self.static_cols = list(static.columns)
        static.insert(0, "unique_id", np.arange(n, dtype=np.int64))
        self.static_df = static.reset_index(drop=True)
        self.bound = True

    # ---- frame builders (unit-tested) ----
    def _frame(self, rows, days, y=None) -> pd.DataFrame:
        data = {"unique_id": rows.astype(np.int64), "ds": self.dates[days]}
        if y is not None:
            data["y"] = y.astype(np.float32)
        if self.use_exog:
            data["price_norm"] = np.nan_to_num(self.pnorm[rows, days], nan=1.0)
            data["listed"] = self.listed[rows, days].astype(np.float32)
            data["snap"] = self.snap[rows, days].astype(np.float32)
            for k in ("dow_sin", "dow_cos", "doy_sin", "doy_cos", "event"):
                data[k] = self.cal[k][days]
        return pd.DataFrame(data)

    def training_frame(self, Y_train: np.ndarray) -> pd.DataFrame:
        """Last `history` days per series, starting at the first sale (no pre-launch zeros)."""
        T = Y_train.shape[1]
        window = Y_train[:, -self.history :]
        W = window.shape[1]
        nonzero = window > 0
        first = np.where(nonzero.any(axis=1), nonzero.argmax(axis=1), W - 1)
        start = np.minimum(first, W - 28)                          # keep at least 28 days
        keep = np.arange(W)[None, :] >= start[:, None]
        rows, cols = np.nonzero(keep)                              # sorted by series, then day
        return self._frame(rows, cols + (T - W), window[rows, cols])

    def future_frame(self, T: int, horizon: int) -> pd.DataFrame:
        rows = np.repeat(np.arange(self.n_series), horizon)
        days = np.tile(T + np.arange(horizon), self.n_series)
        return self._frame(rows, days)

    def _loss(self):
        from neuralforecast.losses.pytorch import MSE, DistributionLoss

        if self.loss == "mse":
            return MSE()
        if self.loss == "poisson":
            return DistributionLoss(distribution="Poisson")
        raise ValueError(f"unknown loss {self.loss!r}")

    # ---- forecaster interface ----
    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        assert self.bound, "call bind(meta) before predict"
        assert Y_train.shape[0] == self.n_series, "Y_train rows must match the bound metadata"
        import neuralforecast.models as nfm
        from neuralforecast import NeuralForecast

        n, T = Y_train.shape
        df = self.training_frame(Y_train)
        fut = self.future_frame(T, horizon) if self.use_exog else None

        kwargs = dict(
            h=horizon, input_size=self.input_size, loss=self._loss(), max_steps=self.max_steps,
            scaler_type="standard", start_padding_enabled=True, random_seed=1,
            enable_checkpointing=False, logger=False,
        )
        if self.use_exog:
            kwargs.update(futr_exog_list=FUTR_COLS, stat_exog_list=self.static_cols)
        kwargs.update(self.model_kwargs)
        model = getattr(nfm, self.model_cls)(**kwargs)

        nf = NeuralForecast(models=[model], freq="D")
        nf.fit(df=df, static_df=self.static_df if self.use_exog else None)
        del df
        out = nf.predict(futr_df=fut)

        if "unique_id" not in out.columns:
            out = out.reset_index()
        out = out.sort_values(["unique_id", "ds"])
        alias = getattr(model, "alias", None) or self.model_cls
        if alias in out.columns:
            col = alias
        else:
            col = [c for c in out.columns if c not in ("unique_id", "ds")
                   and not any(s in c for s in ("-lo-", "-hi-", "median"))][0]
        print(f"  {self.name}: forecast column '{col}' from {list(out.columns)}")

        pred = out[col].to_numpy().reshape(n, horizon)
        pred = np.clip(np.nan_to_num(pred), 0, None).astype(np.float32)
        pred[~self.listed[:, T : T + horizon]] = 0.0               # unlisted items forecast 0
        return pred