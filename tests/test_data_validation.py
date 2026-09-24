import pytest

from m5.config import PROCESSED_DIR
from m5.data.validate import validate_all


@pytest.mark.skipif(
    not (PROCESSED_DIR / "sales_wide.parquet").exists(), reason="processed data not available"
)
def test_processed_data_is_valid():
    validate_all()  