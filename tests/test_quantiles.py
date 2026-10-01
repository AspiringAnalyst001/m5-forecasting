import numpy as np

from m5.evaluation.metrics import coverage, pinball_loss


def test_pinball_loss_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball_loss(y, y, 0.5) == 0.0


def test_pinball_loss_penalizes_asymmetrically():
    y_true = np.array([10.0])
    under = np.array([5.0])   # prediction below truth
    over = np.array([15.0])   # prediction above truth
    # At q=0.9, underprediction should be penalized more than overprediction
    assert pinball_loss(y_true, under, 0.9) > pinball_loss(y_true, over, 0.9)
    # At q=0.1, the reverse
    assert pinball_loss(y_true, over, 0.1) > pinball_loss(y_true, under, 0.1)


def test_coverage_full_when_interval_contains_everything():
    y = np.array([1.0, 5.0, 3.0])
    lo = np.zeros(3)
    hi = np.full(3, 10.0)
    assert coverage(y, lo, hi) == 1.0


def test_coverage_zero_when_interval_excludes_everything():
    y = np.array([1.0, 5.0, 3.0])
    lo = np.full(3, 100.0)
    hi = np.full(3, 200.0)
    assert coverage(y, lo, hi) == 0.0