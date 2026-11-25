"""Inference and input validation module for Used Car Price Prediction."""

import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explainability import explain_single_prediction
from src.preprocessing import FEATURE_COLUMNS, REFERENCE_YEAR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"


def load_model_and_metadata(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    metadata_path: Path | str = DEFAULT_METADATA_PATH,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load the saved best Scikit-learn pipeline and training metadata JSON."""
    m_path = Path(model_path)
    meta_path = Path(metadata_path)
    if not m_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{m_path}'. Run 'python -m src.train' first."
        )
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{meta_path}'. Run 'python -m src.train' first."
        )

    pipeline = joblib.load(m_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return pipeline, metadata


def _to_positive_float(value: Any, field_name: str, allow_none: bool = False) -> float:
    """Validate numeric inputs, optionally converting None/NaN for pipeline imputation."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        if allow_none:
            return float("nan")
        raise ValueError(f"'{field_name}' is required and cannot be empty.")

    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{field_name}' must be a valid number (got {value!r}).") from exc

    if not math.isfinite(num):
        raise ValueError(f"'{field_name}' must be a finite number.")
    if num < 0:
        raise ValueError(f"'{field_name}' cannot be negative (got {num}).")
    return num


def prepare_prediction_input(user_input: Dict[str, Any]) -> pd.DataFrame:
    """Validate and convert raw user form input into a single-row feature DataFrame."""
    if not isinstance(user_input, dict) or not user_input:
        raise ValueError("Input payload must be a non-empty dictionary.")

    year_val = _to_positive_float(user_input.get("year"), "year", allow_none=False)
    if year_val < 1980 or year_val > 2030:
        raise ValueError(f"'year' must be between 1980 and 2030 (got {int(year_val)}).")

    vehicle_age = max(0.0, float(REFERENCE_YEAR) - year_val)
    km_driven = _to_positive_float(user_input.get("km_driven"), "km_driven", allow_none=False)
    if km_driven > 1_000_000:
        raise ValueError("'km_driven' exceeds realistic upper limit (1,000,000 km).")

    mileage = _to_positive_float(user_input.get("mileage"), "mileage", allow_none=True)
    if not math.isnan(mileage) and mileage == 0:
        mileage = float("nan")

    engine = _to_positive_float(user_input.get("engine"), "engine", allow_none=True)
    if not math.isnan(engine) and engine == 0:
        engine = float("nan")

    max_power = _to_positive_float(user_input.get("max_power"), "max_power", allow_none=True)
    if not math.isnan(max_power) and max_power == 0:
        max_power = float("nan")

    seats = _to_positive_float(user_input.get("seats"), "seats", allow_none=True)

    brand = str(user_input.get("brand", "")).strip() or "Unknown"
    car_model = str(user_input.get("car_model", "")).strip()
    if not car_model:
        car_model = brand

    fuel = str(user_input.get("fuel", "Petrol")).strip() or "Petrol"
    seller_type = str(user_input.get("seller_type", "Individual")).strip() or "Individual"
    transmission = str(user_input.get("transmission", "Manual")).strip() or "Manual"
    owner = str(user_input.get("owner", "First Owner")).strip() or "First Owner"

    row = {
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "brand": brand,
        "car_model": car_model,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_car_price(
    user_input: Dict[str, Any],
    pipeline: Pipeline | None = None,
    metadata: Dict[str, Any] | None = None,
    include_explanation: bool = True,
) -> Dict[str, Any]:
    """Predict used-car resale price in INR with optional local SHAP explanation."""
    if pipeline is None or metadata is None:
        pipeline, metadata = load_model_and_metadata()

    input_df = prepare_prediction_input(user_input)
    raw_pred = float(pipeline.predict(input_df)[0])
    if not math.isfinite(raw_pred):
        raise RuntimeError("Model returned a non-finite prediction.")

    predicted_price = max(0.0, raw_pred)
    explanation = (
        explain_single_prediction(pipeline, input_df)
        if include_explanation
        else {"available": False, "contributions": []}
    )

    return {
        "predicted_price_inr": predicted_price,
        "predicted_price_lakhs": predicted_price / 100_000.0,
        "model_name": metadata.get("best_model_name", "Best Regression Model"),
        "explanation": explanation,
        "processed_features": input_df.iloc[0].to_dict(),
    }
