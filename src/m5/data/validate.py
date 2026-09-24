"""Data contracts for the processed M5 tables. Run: python -m m5.data.validate"""
import pandas as pd
import pandera.pandas as pa

from m5.config import N_DAYS, N_SERIES, STATES, STORES
from m5.data.load import load_calendar, load_prices, load_sales

SALES_META_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(unique=True),
        "store_id": pa.Column(checks=pa.Check.isin(STORES)),
        "state_id": pa.Column(checks=pa.Check.isin(STATES)),
        "cat_id": pa.Column(checks=pa.Check.isin(["FOODS", "HOBBIES", "HOUSEHOLD"])),
    }
)

CALENDAR_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, unique=True, coerce=True),
        "d": pa.Column(unique=True),
        "wm_yr_wk": pa.Column(int, coerce=True),
        "snap_CA": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "snap_TX": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
        "snap_WI": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    }
)

PRICES_SCHEMA = pa.DataFrameSchema(
    {
        "store_id": pa.Column(checks=pa.Check.isin(STORES)),
        "wm_yr_wk": pa.Column(int, coerce=True),
        "sell_price": pa.Column(float, pa.Check.gt(0), coerce=True),
    },
    unique=["store_id", "item_id", "wm_yr_wk"],
)


def validate_all() -> None:
    meta, Y, day_cols = load_sales()
    SALES_META_SCHEMA.validate(meta, lazy=True)
    assert Y.shape == (N_SERIES, N_DAYS), f"Unexpected sales shape {Y.shape}"
    assert (Y >= 0).all(), "Negative sales found"
    assert day_cols == [f"d_{i}" for i in range(1, N_DAYS + 1)], "Day columns not contiguous"

    cal = load_calendar()
    CALENDAR_SCHEMA.validate(cal, lazy=True)
    assert len(cal) >= N_DAYS, "Calendar shorter than sales history"
    gaps = cal["date"].diff().dropna()
    assert (gaps == pd.Timedelta(days=1)).all(), "Calendar dates are not contiguous"

    PRICES_SCHEMA.validate(load_prices(), lazy=True)
    print("All data checks passed.")


if __name__ == "__main__":
    validate_all()