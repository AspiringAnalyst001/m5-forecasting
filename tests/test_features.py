import numpy as np
import pandas as pd

from m5.features.build import price_matrix, rolling_mean_shifted, target_features


def test_rolling_mean_matches_pandas():
    rng = np.random.default_rng(0)
    A = rng.poisson(3, size=(3, 200)).astype(np.float32)
    A[:, :30] = np.nan                                   # pre-launch days
    got = rolling_mean_shifted(A, 7)
    for i in range(3):
        want = pd.Series(A[i]).shift(28).rolling(7, min_periods=1).mean().to_numpy()
        np.testing.assert_allclose(got[i], want, rtol=1e-5, equal_nan=True)


def test_features_ignore_the_27_days_before_t():
    rng = np.random.default_rng(1)
    A = rng.poisson(3, size=(5, 400)).astype(np.float32)
    t = 300
    B = A.copy()
    B[:, t - 27 :] = rng.poisson(50, size=(5, 400 - (t - 27))).astype(np.float32)
    fa, fb = target_features(A), target_features(B)
    for name in fa:
        np.testing.assert_allclose(fa[name][:, t], fb[name][:, t], equal_nan=True, err_msg=name)


def test_features_do_react_to_day_t_minus_28():
    rng = np.random.default_rng(2)
    A = rng.poisson(3, size=(5, 400)).astype(np.float32)
    t = 300
    B = A.copy()
    B[:, t - 28] += 10
    assert not np.allclose(target_features(A)["lag_28"][:, t], target_features(B)["lag_28"][:, t])


def test_price_matrix_maps_weeks_to_days():
    prices = pd.DataFrame(
        {"item_id": ["A", "A", "B"], "wm_yr_wk": [1, 2, 2], "sell_price": [1.0, 1.5, 2.0]}
    )
    P = price_matrix(prices, np.array(["A", "B"]), np.array([1, 1, 2, 2]))
    expected = np.array([[1.0, 1.0, 1.5, 1.5], [np.nan, np.nan, 2.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(P, expected, equal_nan=True)