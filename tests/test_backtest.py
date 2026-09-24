import numpy as np
import pandas as pd

from m5.config import FOLDS, N_DAYS
from m5.evaluation.backtest import backtest_fold
from m5.evaluation.metrics import WRMSSE
from m5.models.baselines import Naive, SeasonalNaive, SeasonalWindowAverage


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


def periodic_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    pattern = rng.integers(1, 20, size=(4, 7))
    Y = np.tile(pattern, (1, N_DAYS // 7 + 1))[:, :N_DAYS].astype(np.int16)
    fold = FOLDS["fold2"]
    wrmsse = WRMSSE(make_meta(), Y[:, : fold.train_end], np.ones(4))
    return Y, fold, wrmsse


class Spy:
    name = "spy"

    def __init__(self):
        self.seen = None

    def predict(self, Y_train, horizon):
        self.seen = Y_train.shape[1]
        assert not Y_train.flags.writeable
        return np.zeros((Y_train.shape[0], horizon))


def test_forecaster_only_sees_training_days():
    Y, fold, wrmsse = periodic_data()
    spy = Spy()
    backtest_fold(spy, Y, fold, wrmsse)
    assert spy.seen == fold.train_end


def test_seasonal_naive_is_perfect_on_periodic_data():
    Y, fold, wrmsse = periodic_data()
    row = backtest_fold(SeasonalNaive(7), Y, fold, wrmsse)
    assert np.isclose(row["wrmsse"], 0.0)
    assert np.isclose(row["wape"], 0.0)


def test_seasonal_window_average_is_perfect_on_periodic_data():
    Y, fold, wrmsse = periodic_data()
    row = backtest_fold(SeasonalWindowAverage(7, 8), Y, fold, wrmsse)
    assert np.isclose(row["wrmsse"], 0.0)


def test_naive_is_worse_than_seasonal_naive_on_periodic_data():
    Y, fold, wrmsse = periodic_data()
    naive = backtest_fold(Naive(), Y, fold, wrmsse)["wrmsse"]
    assert naive > 0.1