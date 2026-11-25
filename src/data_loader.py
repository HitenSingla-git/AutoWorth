"""Data loading utilities for the Used Car Price Prediction project."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "dataset" / "used_cars.csv"
DATASET_SOURCE_URL = (
    "https://raw.githubusercontent.com/StatguyUser/"
    "feature_engineering_and_selection_for_explanable_models/main/Car%20details%20v3.csv"
)
KAGGLE_ORIGINAL_PAGE = "https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho"


def load_raw_dataset(filepath: Path | str = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the raw used-car CSV dataset from disk.

    Args:
        filepath: Path to the CSV file. Defaults to ``dataset/used_cars.csv``.

    Returns:
        A pandas DataFrame containing the raw vehicle listings.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If the dataset is empty or missing required columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Please ensure 'dataset/used_cars.csv' exists."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset at '{path}' is empty.")

    required_columns = {"name", "year", "selling_price", "km_driven", "fuel", "transmission"}
    missing_required = required_columns - set(df.columns)
    if missing_required:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_required)}")

    return df


def inspect_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a structured summary of dataset dimensions, missing values, and duplicates.

    Args:
        df: Raw or cleaned DataFrame.

    Returns:
        Dictionary containing row count, column count, dtypes, missing value counts,
        duplicate row count, and selling price summary statistics.
    """
    missing_counts = df.isnull().sum().to_dict()
    dtypes_map = {col: str(dtype) for col, dtype in df.dtypes.items()}

    summary: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": dtypes_map,
        "missing_values": {k: int(v) for k, v in missing_counts.items()},
        "total_missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "source_url": DATASET_SOURCE_URL,
        "kaggle_reference": KAGGLE_ORIGINAL_PAGE,
        "geographical_context": "India (CarDekho listings)",
        "price_unit": "INR (Indian Rupees)",
    }

    if "selling_price" in df.columns:
        prices = pd.to_numeric(df["selling_price"], errors="coerce").dropna()
        summary["selling_price_stats"] = {
            "min": float(prices.min()),
            "median": float(prices.median()),
            "mean": float(prices.mean()),
            "max": float(prices.max()),
            "std": float(prices.std()),
        }

    return summary
