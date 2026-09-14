from pathlib import Path

import pandas as pd

DEFAULT_FILENAME = "AB_NYC_2019.csv"
DATASET_URL = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/AB_NYC_2019.csv"


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load the CSV from a local path or the public fallback URL."""
    csv_path = Path(path) if path else Path("data/raw") / DEFAULT_FILENAME
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if path:
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(DATASET_URL)
