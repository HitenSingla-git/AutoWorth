"""Training pipeline for Linear Regression, Decision Tree, and Random Forest models."""

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_loader import DEFAULT_DATASET_PATH, inspect_dataset, load_raw_dataset
from src.evaluation import (
    compute_regression_metrics,
    generate_depreciation_charts,
    generate_eda_charts,
    generate_model_comparison_charts,
)
from src.explainability import (
    compute_global_feature_importance,
    compute_shap_global_explanation,
)
from src.preprocessing import (
    RANDOM_STATE,
    TEST_SIZE,
    build_preprocessor,
    get_train_test_split,
    prepare_cleaned_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CHARTS_DIR = PROJECT_ROOT / "reports" / "charts"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def build_candidate_models() -> Dict[str, Dict[str, Any]]:
    """Define the three primary regression models and their hyperparameters."""
    return {
        "Linear Regression": {
            "estimator": LinearRegression(),
            "params": {"fit_intercept": True},
            "description": "Ordinary Least Squares baseline regression model.",
        },
        "Decision Tree Regressor": {
            "estimator": DecisionTreeRegressor(
                max_depth=12,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            "params": {
                "max_depth": 12,
                "min_samples_leaf": 5,
                "random_state": RANDOM_STATE,
            },
            "description": "Non-linear tree-based regressor with depth regularization.",
        },
        "Random Forest Regressor": {
            "estimator": RandomForestRegressor(
                n_estimators=150,
                max_depth=18,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "params": {
                "n_estimators": 150,
                "max_depth": 18,
                "min_samples_leaf": 2,
                "random_state": RANDOM_STATE,
            },
            "description": "Ensemble of 150 decision trees trained with bootstrap aggregation.",
        },
    }


def run_training_pipeline(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    models_dir: Path = MODELS_DIR,
    charts_dir: Path = CHARTS_DIR,
) -> Dict[str, Any]:
    """Execute full data loading, cleaning, training, evaluation, XAI, and saving."""
    models_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and inspect raw dataset
    raw_df = load_raw_dataset(dataset_path)
    raw_summary = inspect_dataset(raw_df)

    # 2. Clean and split without data leakage
    cleaned_df, cleaning_report = prepare_cleaned_dataframe(raw_df)
    X_train, X_test, y_train, y_test = get_train_test_split(cleaned_df)

    # 3. Generate EDA and Depreciation charts
    eda_charts = generate_eda_charts(cleaned_df, charts_dir)
    depreciation_info = generate_depreciation_charts(cleaned_df, charts_dir)

    # 4. Train and evaluate all three models on identical split
    candidates = build_candidate_models()
    results: List[Dict[str, Any]] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    test_predictions_map: Dict[str, np.ndarray] = {}

    for model_name, spec in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", spec["estimator"]),
            ]
        )
        pipeline.fit(X_train, y_train)
        preds = np.maximum(0.0, pipeline.predict(X_test))
        metrics = compute_regression_metrics(y_test, preds)

        results.append(
            {
                "model_name": model_name,
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "params": spec["params"],
                "description": spec["description"],
            }
        )
        trained_pipelines[model_name] = pipeline
        test_predictions_map[model_name] = preds

    # Select best model by highest R2 score on test set
    best_result = max(results, key=lambda r: r["r2"])
    best_model_name = best_result["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    best_predictions = test_predictions_map[best_model_name]

    # 5. Generate model comparison charts
    comparison_charts = generate_model_comparison_charts(
        results=results,
        y_test=y_test,
        best_predictions=best_predictions,
        best_model_name=best_model_name,
        charts_dir=charts_dir,
    )

    # 6. Feature importance and SHAP explainability on best model
    feature_importance_info = compute_global_feature_importance(
        best_pipeline, X_test, y_test, charts_dir
    )
    shap_info = compute_shap_global_explanation(
        best_pipeline, X_test, y_test, charts_dir, sample_size=200
    )

    # 7. Collect dropdown metadata & cleaned dataset stats for Streamlit UI
    brand_to_models: Dict[str, List[str]] = {}
    for brand, grp in cleaned_df.groupby("brand"):
        brand_to_models[str(brand)] = sorted(grp["car_model"].unique().tolist())

    cleaned_stats = {
        "rows": int(len(cleaned_df)),
        "mean_price": float(cleaned_df["selling_price"].mean()),
        "median_price": float(cleaned_df["selling_price"].median()),
        "min_price": float(cleaned_df["selling_price"].min()),
        "max_price": float(cleaned_df["selling_price"].max()),
        "year_min": int(cleaned_df["year"].min()),
        "year_max": int(cleaned_df["year"].max()),
        "num_brands": int(cleaned_df["brand"].nunique()),
        "num_models": int(cleaned_df["car_model"].nunique()),
        "fuel_types": sorted(cleaned_df["fuel"].dropna().unique().tolist()),
        "seller_types": sorted(cleaned_df["seller_type"].dropna().unique().tolist()),
        "transmission_types": sorted(cleaned_df["transmission"].dropna().unique().tolist()),
        "owner_types": [
            "First Owner",
            "Second Owner",
            "Third Owner",
            "Fourth & Above Owner",
            "Test Drive Car",
        ],
        "seat_options": sorted([int(s) for s in cleaned_df["seats"].dropna().unique().tolist()]),
        "medians": {
            "km_driven": float(cleaned_df["km_driven"].median()),
            "mileage": float(cleaned_df["mileage"].median()),
            "engine": float(cleaned_df["engine"].median()),
            "max_power": float(cleaned_df["max_power"].median()),
            "seats": int(cleaned_df["seats"].median()),
        },
        "brand_to_models": brand_to_models,
    }

    # 8. Save best model pipeline & comprehensive metadata
    best_model_path = models_dir / "best_model.pkl"
    joblib.dump(best_pipeline, best_model_path)

    metadata: Dict[str, Any] = {
        "best_model_name": best_model_name,
        "best_model_metrics": best_result,
        "model_results": results,
        "split_config": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "raw_dataset_summary": raw_summary,
        "cleaning_report": cleaning_report,
        "cleaned_dataset_stats": cleaned_stats,
        "feature_importance": feature_importance_info,
        "shap_explainability": shap_info,
        "depreciation_summary": {
            "age_summary": depreciation_info["age_summary"],
            "fuel_summary": depreciation_info["fuel_summary"],
            "transmission_summary": depreciation_info["transmission_summary"],
        },
        "charts": {
            **eda_charts,
            **depreciation_info["charts"],
            **comparison_charts,
            "feature_importance_top10": feature_importance_info["chart_path"],
            "shap_summary": shap_info["summary_chart_path"],
            "shap_bar": shap_info["bar_chart_path"],
        },
    }

    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(f"Training complete. Best Model: {summary['best_model_name']}")
    for r in summary["model_results"]:
        print(
            f"  - {r['model_name']}: R2={r['r2']:.4f}, "
            f"MAE=Rs.{r['mae']:,.0f}, RMSE=Rs.{r['rmse']:,.0f}"
        )
