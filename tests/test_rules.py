import numpy as np
import pandas as pd

from m5.models.rules import ClosureRule


class Ones:
    name = "ones"

    def predict(self, Y_train, horizon):
        return np.ones((Y_train.shape[0], horizon), dtype=np.float32)


def test_closure_rule_zeroes_repeat_closures_only():
    dates = pd.Series(pd.date_range("2011-01-29", periods=1200))
    T = int((pd.Timestamp("2013-12-25") - dates[0]).days) - 3     # Dec 25 is forecast day 4
    rng = np.random.default_rng(0)
    Y = (rng.poisson(5, size=(4, T)) + 1).astype(np.int16)
    past_xmas = np.flatnonzero((dates[:T].dt.month == 12) & (dates[:T].dt.day == 25))
    Y[:, past_xmas] = 0                                            # closed in 2011 and 2012

    rule = ClosureRule(Ones(), dates=dates)
    rule.bind(pd.DataFrame())
    pred = rule.predict(Y, 7)

    assert (pred[:, 3] == 0).all()                                  # 2013-12-25 zeroed
    assert (np.delete(pred, 3, axis=1) == 1).all()                  # everything else untouched