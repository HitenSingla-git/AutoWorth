"""Automated test suite for the Used Car Price Prediction system."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import DEFAULT_DATASET_PATH, inspect_dataset, load_raw_dataset
from src.explainability import explain_single_prediction
from src.prediction import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    load_model_and_metadata,
    predict_car_price,
    prepare_prediction_input,
)
from src.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    get_train_test_split,
    prepare_cleaned_dataframe,
)


def test_dataset_loads_correctly():
    df = load_raw_dataset(DEFAULT_DATASET_PATH)
    summary = inspect_dataset(df)
    assert summary["rows"] == 8128
    assert summary["columns"] == 13
    assert "selling_price" in summary["column_names"]
    assert summary["duplicate_rows"] > 0


def test_preprocessing_and_leak_free_split():
    df = load_raw_dataset(DEFAULT_DATASET_PATH)
    cleaned_df, report = prepare_cleaned_dataframe(df)
    assert report["final_rows"] == 6923
    assert report["duplicates_removed"] == 1202
    assert report["odometer_outliers_removed"] == 3
    for col in FEATURE_COLUMNS:
        assert col in cleaned_df.columns

    X_train, X_test, y_train, y_test = get_train_test_split(cleaned_df)
    assert len(X_train) + len(X_test) == len(cleaned_df)
    assert pytest.approx(len(X_test) / len(cleaned_df), abs=0.01) == 0.20

    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_saved_model_and_metadata_exist_and_reload():
    assert DEFAULT_MODEL_PATH.exists(), "models/best_model.pkl missing"
    assert DEFAULT_METADATA_PATH.exists(), "models/model_metadata.json missing"

    pipeline, metadata = load_model_and_metadata()
    assert metadata["best_model_name"] == "Random Forest Regressor"
    assert len(metadata["model_results"]) == 3

    model_names = {r["model_name"] for r in metadata["model_results"]}
    assert model_names == {
        "Linear Regression",
        "Decision Tree Regressor",
        "Random Forest Regressor",
    }
    for res in metadata["model_results"]:
        assert 0.0 < res["r2"] < 1.0
        assert res["mae"] > 0
        assert res["rmse"] >= res["mae"]
    assert hasattr(pipeline, "predict")


def test_sample_predictions_finite_and_non_negative():
    pipeline, metadata = load_model_and_metadata()
    samples = [
        {
            "brand": "Maruti",
            "car_model": "Maruti Swift",
            "year": 2016,
            "km_driven": 45000,
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "mileage": 20.4,
            "engine": 1197,
            "max_power": 83.1,
            "seats": 5,
        },
        {
            "brand": "Hyundai",
            "car_model": "Hyundai Creta",
            "year": 2018,
            "km_driven": 32000,
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Automatic",
            "owner": "First Owner",
            "mileage": 17.0,
            "engine": 1582,
            "max_power": 126.2,
            "seats": 5,
        },
        {
            "brand": "BMW",
            "car_model": "BMW 3",
            "year": 2017,
            "km_driven": 38000,
            "fuel": "Diesel",
            "seller_type": "Dealer",
            "transmission": "Automatic",
            "owner": "First Owner",
            "mileage": 18.8,
            "engine": 1995,
            "max_power": 190.0,
            "seats": 5,
        },
        # Unseen brand/model should be handled cleanly by OneHotEncoder(handle_unknown='ignore')
        {
            "brand": "UnseenBrandXYZ",
            "car_model": "UnseenBrandXYZ ModelQ",
            "year": 2015,
            "km_driven": 60000,
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "Second Owner",
            "mileage": None,
            "engine": None,
            "max_power": None,
            "seats": 5,
        },
    ]

    preds = []
    for sample in samples:
        out = predict_car_price(sample, pipeline=pipeline, metadata=metadata)
        price = out["predicted_price_inr"]
        assert isinstance(price, float)
        assert math.isfinite(price)
        assert price >= 0.0
        preds.append(price)

    # Luxury BMW should predict higher than a standard Maruti Swift
    assert preds[2] > preds[0]


def test_invalid_inputs_handled_gracefully():
    with pytest.raises(ValueError):
        prepare_prediction_input({})
    with pytest.raises(ValueError):
        prepare_prediction_input({"year": "invalid", "km_driven": 50000})
    with pytest.raises(ValueError):
        prepare_prediction_input({"year": 2016, "km_driven": -500})
    with pytest.raises(ValueError):
        prepare_prediction_input({"year": 1950, "km_driven": 50000})


def test_explainability_and_charts_generated():
    pipeline, metadata = load_model_and_metadata()
    shap_info = metadata["shap_explainability"]
    assert shap_info["shap_available"] is True
    assert len(shap_info["top10_shap"]) == 10

    input_df = prepare_prediction_input(
        {
            "brand": "Maruti",
            "car_model": "Maruti Swift",
            "year": 2016,
            "km_driven": 45000,
            "fuel": "Petrol",
            "seller_type": "Individual",
            "transmission": "Manual",
            "owner": "First Owner",
            "mileage": 20.4,
            "engine": 1197,
            "max_power": 83.1,
            "seats": 5,
        }
    )
    local_expl = explain_single_prediction(pipeline, input_df)
    assert local_expl["available"] is True
    assert len(local_expl["contributions"]) > 0

    for chart_key, chart_path in metadata["charts"].items():
        assert Path(chart_path).exists(), f"Missing chart for {chart_key}: {chart_path}"
