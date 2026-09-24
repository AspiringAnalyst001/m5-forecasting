import pandas as pd
import pytest

from m5.config import PROCESSED_DIR
from m5.data.load import load_calendar, load_sales

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "sales_wide.parquet").exists(), reason="processed data not available"
)


def test_frames_respect_the_forecast_origin():
    from m5.models.neural import FUTR_COLS, NeuralForecastWrapper

    meta, Y, _ = load_sales()
    mask = (meta["store_id"].astype(str) == "CA_1").to_numpy()
    meta, Y = meta.loc[mask].reset_index(drop=True), Y[mask]
    T, h = 1885, 28

    w = NeuralForecastWrapper(name="t", use_exog=True)
    w.bind(meta)
    dates = pd.to_datetime(load_calendar()["date"]).to_numpy()

    train, fut = w.training_frame(Y[:, :T]), w.future_frame(T, h)

    assert train["ds"].max() == dates[T - 1]          # last training day
    assert fut["ds"].min() == dates[T]                # first forecast day
    assert fut["ds"].max() == dates[T + h - 1]
    assert "y" not in fut.columns
    assert len(fut) == len(meta) * h
    assert not train[FUTR_COLS].isna().any().any()
    assert not fut[FUTR_COLS].isna().any().any()

    per_series = train.groupby("unique_id")["ds"].agg(["min", "max", "count"])
    span = (per_series["max"] - per_series["min"]).dt.days + 1
    assert (span == per_series["count"]).all()        # no gaps inside any series