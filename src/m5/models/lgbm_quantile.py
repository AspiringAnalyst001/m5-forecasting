"""Quantile LightGBM forecaster: trains one model per quantile, reusing the same
leak-free features as the point forecaster. Used for prediction intervals."""
import numpy as np
import pandas as pd

from m5.models.lgbm import DEFAULT_PARAMS, CATEGORICAL, LightGBMForecaster

QUANTILE_PARAMS = {k: v for k, v in DEFAULT_PARAMS.items() if k != "tweedie_variance_power"}
QUANTILE_PARAMS["objective"] = "quantile"


class QuantileLightGBMForecaster(LightGBMForecaster):
    """Like LightGBMForecaster, but predict() returns a dict of {quantile: array}
    instead of a single point forecast. Not used directly by the backtester —
    call predict_quantiles() instead of predict()."""

    def __init__(self, name="lgbm_quantiles", train_days=365, num_rounds=200,
                 quantiles=(0.1, 0.5, 0.9), num_threads=0):
        super().__init__(name=name, train_days=train_days, num_rounds=num_rounds,
                          params=QUANTILE_PARAMS, num_threads=num_threads)
        self.quantiles = quantiles

    def predict_quantiles(self, Y_train: np.ndarray, horizon: int) -> dict[float, np.ndarray]:
        if self.stores is None:
            raise RuntimeError("call bind(meta) before predict_quantiles")
        assert Y_train.shape[0] == self.n_series
        T = Y_train.shape[1]
        out = {q: np.zeros((Y_train.shape[0], horizon), dtype=np.float32) for q in self.quantiles}

        for store, s in self.stores.items():
            T_pad = T + horizon
            P = s["P"][:, :T_pad]
            on_sale = ~np.isnan(P)
            from m5.features.build import target_features, price_features

            A = np.full(P.shape, np.nan, dtype=np.float32)
            A[:, :T] = np.where(on_sale[:, :T], Y_train[s["rows"]].astype(np.float32), np.nan)
            feats = target_features(A)
            feats.update(price_features(P))

            t_idx = np.arange(T_pad)[None, :]
            train_mask = ~np.isnan(A) & (t_idx >= T - self.train_days)
            pred_mask = on_sale & (t_idx >= T)

            X, names, i, t = self._matrix(s, feats, train_mask)
            y = A[i, t]
            Xp, _, ip, tp = self._matrix(s, feats, pred_mask)

            for q in self.quantiles:
                params = {**self.params, "alpha": q}
                import lightgbm as lgb
                train_set = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=CATEGORICAL)
                model = lgb.train(params, train_set, num_boost_round=self.num_rounds)
                pred = np.zeros((len(Y_train[s["rows"]]), horizon), dtype=np.float32)
                pred[ip, tp - T] = model.predict(Xp)
                out[q][s["rows"]] = np.clip(np.nan_to_num(pred), 0, None)
            print(f"  {self.name}: {store} done")
        return out