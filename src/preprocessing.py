"""Preprocessing and feature engineering pipeline for Used Car Price Prediction."""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reference year matches the latest manufacturing year in the CarDekho dataset (2020).
REFERENCE_YEAR = 2020
MAX_KM_DRIVEN = 500_000
RANDOM_STATE = 42
TEST_SIZE = 0.20

NUMERIC_FEATURES: List[str] = [
    "vehicle_age",
    "km_driven",
    "mileage",
    "engine",
    "max_power",
    "seats",
]

CATEGORICAL_FEATURES: List[str] = [
    "brand",
    "car_model",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
]

FEATURE_COLUMNS: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "selling_price"


def extract_numeric_value(series: pd.Series) -> pd.Series:
    """Extract the leading floating-point number from strings with unit suffixes."""
    extracted = series.astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0]
    return pd.to_numeric(extracted, errors="coerce")


def clean_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw dataset columns into model-ready feature columns.

    Handles unit stripping for ``mileage``, ``engine``, and ``max_power``,
    replaces physically invalid zero values in ``mileage`` and ``max_power`` with NaN
    for downstream imputation, derives ``brand``, ``car_model``, and ``vehicle_age``,
    and removes the free-text ``torque`` column.
    """
    cleaned = df.copy()

    # Extract brand and 2-token model name from full listing title
    name_tokens = cleaned["name"].astype(str).str.strip().str.split()
    cleaned["brand"] = name_tokens.str[0].fillna("Unknown")
    cleaned["car_model"] = name_tokens.str[:2].str.join(" ").fillna("Unknown")

    # Convert manufacturing year to vehicle age relative to REFERENCE_YEAR
    year_num = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned["vehicle_age"] = (REFERENCE_YEAR - year_num).clip(lower=0)

    # Parse numeric values from unit-annotated text columns
    for col in ("mileage", "engine", "max_power"):
        if col in cleaned.columns:
            cleaned[col] = extract_numeric_value(cleaned[col])

    # Zero mileage or zero bhp is invalid/missing in this dataset; mark as NaN for imputation
    if "mileage" in cleaned.columns:
        cleaned.loc[cleaned["mileage"] <= 0, "mileage"] = np.nan
    if "max_power" in cleaned.columns:
        cleaned.loc[cleaned["max_power"] <= 0, "max_power"] = np.nan

    cleaned["km_driven"] = pd.to_numeric(cleaned["km_driven"], errors="coerce")
    cleaned["seats"] = pd.to_numeric(cleaned["seats"], errors="coerce")

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")

    return cleaned


def prepare_cleaned_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Deduplicate, filter extreme odometer outliers, and clean raw features.

    Args:
        df: Raw DataFrame loaded from ``dataset/used_cars.csv``.

    Returns:
        Tuple of (cleaned_df, cleaning_report).
    """
    raw_rows = int(len(df))

    # Step 1: Remove exact duplicate listings
    dedup_df = df.drop_duplicates().copy()
    duplicates_removed = raw_rows - int(len(dedup_df))

    # Step 2: Remove extreme odometer outliers (> 500,000 km; only 3 anomalous rows)
    dedup_df["km_driven"] = pd.to_numeric(dedup_df["km_driven"], errors="coerce")
    valid_km_mask = dedup_df["km_driven"].notna() & (dedup_df["km_driven"] > 0) & (
        dedup_df["km_driven"] <= MAX_KM_DRIVEN
    )
    outliers_removed = int((~valid_km_mask).sum())
    filtered_df = dedup_df.loc[valid_km_mask].copy()

    # Step 3: Clean and engineer features
    cleaned_df = clean_raw_features(filtered_df)
    cleaned_df = cleaned_df.dropna(subset=[TARGET_COLUMN, "vehicle_age"]).reset_index(drop=True)

    # Identify IQR-based price outlier count for reporting (kept in data as luxury cars are valid)
    q1 = cleaned_df[TARGET_COLUMN].quantile(0.25)
    q3 = cleaned_df[TARGET_COLUMN].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    luxury_outliers_count = int((cleaned_df[TARGET_COLUMN] > upper_fence).sum())

    cleaning_report: Dict[str, Any] = {
        "raw_rows": raw_rows,
        "duplicates_removed": duplicates_removed,
        "odometer_outliers_removed": outliers_removed,
        "odometer_threshold_km": MAX_KM_DRIVEN,
        "final_rows": int(len(cleaned_df)),
        "reference_year": REFERENCE_YEAR,
        "luxury_price_iqr_upper_fence": float(upper_fence),
        "luxury_price_iqr_outlier_count": luxury_outliers_count,
        "missing_before_imputation": {
            col: int(cleaned_df[col].isnull().sum()) for col in FEATURE_COLUMNS
        },
    }
    return cleaned_df, cleaning_report


def build_preprocessor() -> ColumnTransformer:
    """Create a Scikit-learn ColumnTransformer with median/mode imputation, scaling, and OHE."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_train_test_split(
    cleaned_df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split cleaned DataFrame into 80:20 train and test sets without data leakage."""
    X = cleaned_df[FEATURE_COLUMNS].copy()
    y = cleaned_df[TARGET_COLUMN].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
