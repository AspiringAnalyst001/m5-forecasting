import numpy as np
import pandas as pd

from m5.evaluation.metrics import LEVELS, WRMSSE, build_aggregation_matrix, wape


def make_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "store_id": ["CA_1", "CA_2", "CA_1", "CA_2"],
            "state_id": ["CA"] * 4,
            "cat_id": ["FOODS"] * 4,
            "dept_id": ["FOODS_1"] * 4,
        }
    )


def make_metric(seed: int = 0):
    rng = np.random.default_rng(seed)
    Y_train = rng.poisson(3, size=(4, 100))
    dollars = np.ones(4)
    return WRMSSE(make_meta(), Y_train, dollars), rng


def test_aggregation_matrix_shape():
    S, sizes = build_aggregation_matrix(make_meta())
    assert S.shape == (20, 4)
    assert len(sizes) == len(LEVELS)
    assert sum(sizes) == 20


def test_total_row_sums_all_series():
    S, _ = build_aggregation_matrix(make_meta())
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert (S @ y)[0] == 10.0


def test_weights_sum_to_one():
    metric, _ = make_metric()
    assert np.isclose(metric.weights.sum(), 1.0)


def test_perfect_forecast_scores_zero():
    metric, rng = make_metric()
    Y_true = rng.poisson(3, size=(4, 28))
    assert metric.score(Y_true, Y_true) == 0.0


def test_worse_forecast_scores_higher():
    metric, rng = make_metric()
    Y_true = rng.poisson(3, size=(4, 28))
    good = Y_true + 1
    bad = Y_true + 5
    assert metric.score(Y_true, bad) > metric.score(Y_true, good) > 0


def test_wape():
    y = np.array([10.0, 10.0])
    p = np.array([8.0, 12.0])
    assert np.isclose(wape(y, p), 0.2)