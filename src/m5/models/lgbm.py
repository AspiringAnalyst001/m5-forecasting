"""LightGBM forecaster: one model per store, target features shifted >= 28 days, Tweedie loss."""
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd

from m5.config import RESULTS_DIR
from m5.data.load import load_calendar, load_prices
from m5.features.build import calendar_arrays, price_features, price_matrix, target_features

CATEGORICAL = ["dept", "cat", "event_1", "event_type_1"]

DEFAULT_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "verbosity": -1,
    "seed": 42,
}


class LightGBMForecaster:
    def __init__(self, name="lgbm_v1", train_days=730, num_rounds=400, params=None, num_threads=0):
        self.name = name
        self.train_days, self.num_rounds = train_days, num_rounds
        self.params = {**DEFAULT_PARAMS, **(params or {}), "num_threads": num_threads}
        self.importances: dict[str, pd.Series] = {}
        self.stores: dict[str, dict] | None = None

    def bind(self, meta: pd.DataFrame) -> None:
        """Receive series metadata (row-aligned with Y) and prepare the known-in-advance inputs."""
        cal = load_calendar()
        self.cal = calendar_arrays(cal)
        week_of_day = cal["wm_yr_wk"].to_numpy()
        prices = load_prices()

        stores = meta["store_id"].astype(str).to_numpy()
        items = meta["item_id"].astype(str).to_numpy()
        states = meta["state_id"].astype(str).to_numpy()
        dept = meta["dept_id"].astype(str).astype("category").cat.codes.to_numpy()
        cat = meta["cat_id"].astype(str).astype("category").cat.codes.to_numpy()

        self.n_series = len(meta)
        self.stores = {}
        for store in np.unique(stores):
            rows = np.flatnonzero(stores == store)
            p = prices[prices["store_id"] == str(store)]
            self.stores[str(store)] = {
                "rows": rows,
                "P": price_matrix(p, items[rows], week_of_day),
                "dept": dept[rows],
                "cat": cat[rows],
                "state": states[rows[0]],
            }
        del prices

    def _matrix(self, s: dict, feats: dict, mask: np.ndarray):
        """Build a float32 design matrix directly (no pandas copies)."""
        i, t = np.nonzero(mask)
        cal_cols = ("dow", "dom", "month", "event_1", "event_type_1")
        names = list(feats) + list(cal_cols) + ["snap", "dept", "cat"]
        X = np.empty((len(i), len(names)), dtype=np.float32)
        for j, name in enumerate(feats):
            X[:, j] = feats[name][i, t]
        j = len(feats)
        for name in cal_cols:
            X[:, j] = self.cal[name][t]
            j += 1
        X[:, j] = self.cal[f"snap_{s['state']}"][t]
        X[:, j + 1] = s["dept"][i]
        X[:, j + 2] = s["cat"][i]
        return X, names, i, t

    def _forecast_store(self, store: str, s: dict, Y_store: np.ndarray, T: int, horizon: int):
        T_pad = T + horizon
        P = s["P"][:, :T_pad]
        on_sale = ~np.isnan(P)

        # Sales are known only up to T, and only while the item is on sale
        A = np.full(P.shape, np.nan, dtype=np.float32)
        A[:, :T] = np.where(on_sale[:, :T], Y_store.astype(np.float32), np.nan)

        feats = target_features(A)
        feats.update(price_features(P))

        t_idx = np.arange(T_pad)[None, :]
        train_mask = ~np.isnan(A) & (t_idx >= T - self.train_days)
        pred_mask = on_sale & (t_idx >= T)

        X, names, i, t = self._matrix(s, feats, train_mask)
        y = A[i, t]
        train_set = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=CATEGORICAL)
        model = lgb.train(self.params, train_set, num_boost_round=self.num_rounds)
        self.importances[store] = pd.Series(model.feature_importance("gain"), index=names)
        del train_set, X, y, i, t, train_mask      # free before building the prediction rows

        Xp, _, ip, tp = self._matrix(s, feats, pred_mask)
        out = np.zeros((len(Y_store), horizon), dtype=np.float32)   # unlisted items forecast 0
        out[ip, tp - T] = model.predict(Xp)
        return out

    def predict(self, Y_train: np.ndarray, horizon: int) -> np.ndarray:
        if self.stores is None:
            raise RuntimeError("call bind(meta) before predict")
        assert Y_train.shape[0] == self.n_series, "Y_train rows must match the bound metadata"
        T = Y_train.shape[1]
        out = np.zeros((Y_train.shape[0], horizon), dtype=np.float32)
        for store, s in self.stores.items():
            out[s["rows"]] = self._forecast_store(store, s, Y_train[s["rows"]], T, horizon)
            print(f"  {self.name}: {store} done")
            gc.collect()   # free the large LGBM model before the next store

        RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(self.importances).to_csv(RESULTS_DIR / f"importance_{self.name}_T{T}.csv")
        return out