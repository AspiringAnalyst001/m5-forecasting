"""Load the processed Parquet files created by prepare.py."""
import numpy as np
import pandas as pd

from m5.config import PROCESSED_DIR


def load_sales() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Return (meta, Y, day_cols). Y has shape (n_series, n_days), dtype int16."""
    sales = pd.read_parquet(PROCESSED_DIR / "sales_wide.parquet")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    Y = sales[day_cols].to_numpy()
    meta = sales.drop(columns=day_cols)
    return meta, Y, day_cols


def load_calendar() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "calendar.parquet")


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "prices.parquet")