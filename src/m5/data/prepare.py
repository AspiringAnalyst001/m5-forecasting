"""Convert raw M5 CSVs into compact Parquet files that fit in 8 GB RAM."""
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def prepare_sales() -> None:
    header = pd.read_csv(RAW / "sales_train_evaluation.csv", nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in day_cols}
    dtypes.update({c: "category" for c in ID_COLS if c != "id"})
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv", dtype=dtypes)
    sales.to_parquet(OUT / "sales_wide.parquet")
    print("sales_wide:", sales.shape, f"{sales.memory_usage(deep=True).sum() / 1e6:.0f} MB")


def prepare_calendar() -> None:
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal.to_parquet(OUT / "calendar.parquet")
    print("calendar:", cal.shape)


def prepare_prices() -> None:
    dtypes = {"store_id": "category", "item_id": "category",
              "wm_yr_wk": "int16", "sell_price": "float32"}
    prices = pd.read_csv(RAW / "sell_prices.csv", dtype=dtypes)
    prices.to_parquet(OUT / "prices.parquet")
    print("prices:", prices.shape, f"{prices.memory_usage(deep=True).sum() / 1e6:.0f} MB")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_calendar()
    prepare_sales()
    prepare_prices()