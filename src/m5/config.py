"""Project-wide constants: paths, dataset facts, and validation folds."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Override with an env var later (e.g. a Google Drive path in Colab)
DATA_DIR = Path(os.environ.get("M5_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(os.environ.get("M5_RESULTS_DIR", ROOT / "results"))

N_SERIES = 30_490
N_DAYS = 1_941          # d_1 .. d_1941 in sales_train_evaluation
HORIZON = 28
STORES = [f"{s}_{i}" for s, n in (("CA", 4), ("TX", 3), ("WI", 3)) for i in range(1, n + 1)]
STATES = ["CA", "TX", "WI"]


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: int      # last training day number (1-based)
    valid_start: int
    valid_end: int

    @property
    def valid_slice(self) -> slice:
        """Column slice of the (n_series, n_days) array for the validation window."""
        return slice(self.valid_start - 1, self.valid_end)


FOLDS = {
    "fold1": Fold("fold1", 1829, 1830, 1857),
    "fold2": Fold("fold2", 1857, 1858, 1885),
    "fold3": Fold("fold3", 1885, 1886, 1913),
    "xmas": Fold("xmas", 1777, 1778, 1805),
    "holdout": Fold("holdout", 1913, 1914, 1941),  # touch ONLY at the very end
}