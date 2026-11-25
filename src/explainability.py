"""Feature importance and SHAP explainability utilities."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.preprocessing import FEATURE_COLUMNS

NAVY = "#0F2942"
PRIMARY_BLUE = "#1D6FA5"
CORAL = "#E05A47"


def _prettify_feature_name(raw_name: str) -> str:
    """Convert ColumnTransformer feature names like 'num__vehicle_age' to readable labels."""
    clean = raw_name.replace("num__", "").replace("cat__", "")
    clean = clean.replace("brand_", "Brand: ").replace("car_model_", "Model: ")
    clean = clean.replace("fuel_", "Fuel: ").replace("seller_type_", "Seller: ")
    clean = clean.replace("transmission_", "Trans: ").replace("owner_", "Owner: ")
    clean = clean.replace("vehicle_age", "Vehicle Age (yrs)").replace("km_driven", "KM Driven")
    clean = clean.replace("max_power", "Max Power (bhp)").replace("engine", "Engine (CC)")
    clean = clean.replace("mileage", "Mileage (kmpl)").replace("seats", "Seats")
    return clean


def _map_to_base_feature(raw_name: str) -> str:
    """Map an encoded feature name back to its original input feature name."""
    clean = raw_name.replace("num__", "").replace("cat__", "")
    for base in FEATURE_COLUMNS:
        if clean == base or clean.startswith(f"{base}_"):
            return base
    return clean


def compute_global_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    charts_dir: Path,
) -> Dict[str, Any]:
    """Compute top-10 encoded and grouped feature importances and save chart."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(estimator, "feature_importances_"):
        raw_importances = estimator.feature_importances_
        method_label = "Random Forest Gini Importance"
    else:
        perm = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        raw_importances = perm.importances_mean
        feature_names = list(X_test.columns)
        method_label = "Permutation Importance"

    imp_df = pd.DataFrame(
        {
            "raw_feature": feature_names,
            "feature": [_prettify_feature_name(f) for f in feature_names],
            "base_feature": [_map_to_base_feature(f) for f in feature_names],
            "importance": raw_importances,
        }
    ).sort_values("importance", ascending=False)

    top10_encoded = imp_df.head(10).copy()
    grouped_imp = (
        imp_df.groupby("base_feature")["importance"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grouped_imp["feature"] = grouped_imp["base_feature"].apply(_prettify_feature_name)

    # Save top-10 feature importance chart
    charts_dir.mkdir(parents=True, exist_ok=True)
    chart_path = charts_dir / "feature_importance_top10.png"
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plot_df = top10_encoded.iloc[::-1]
    bars = ax.barh(plot_df["feature"], plot_df["importance"], color=PRIMARY_BLUE)
    ax.set_title(f"Top 10 Most Influential Features ({method_label})", fontweight="bold", color=NAVY)
    ax.set_xlabel("Relative Importance Score")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.004, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5, color=NAVY)
    ax.set_xlim(0, top10_encoded["importance"].max() * 1.16)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return {
        "method": method_label,
        "top10_encoded": top10_encoded[["feature", "importance"]].to_dict(orient="records"),
        "grouped_features": grouped_imp[["base_feature", "feature", "importance"]].to_dict(orient="records"),
        "chart_path": str(chart_path),
    }


def compute_shap_global_explanation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    charts_dir: Path,
    sample_size: int = 200,
) -> Dict[str, Any]:
    """Generate global SHAP explanations using TreeExplainer on a bounded sample.

    Falls back to Permutation Importance (explicitly labeled) if SHAP is incompatible
    with the selected model type.
    """
    charts_dir.mkdir(parents=True, exist_ok=True)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    try:
        import shap

        X_sample = X_test.sample(n=min(sample_size, len(X_test)), random_state=42)
        X_transformed = preprocessor.transform(X_sample)
        feature_names = [_prettify_feature_name(f) for f in preprocessor.get_feature_names_out()]
        X_trans_df = pd.DataFrame(X_transformed, columns=feature_names)

        if hasattr(estimator, "estimators_") or estimator.__class__.__name__ == "DecisionTreeRegressor":
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer.shap_values(X_trans_df, check_additivity=False)
        else:
            explainer = shap.LinearExplainer(estimator, X_trans_df)
            shap_values = explainer.shap_values(X_trans_df)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Global mean |SHAP| bar plot
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame(
            {"feature": feature_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)
        top10_shap = shap_df.head(10).copy()

        # 1. Clean Matplotlib SHAP summary beeswarm plot
        summary_path = charts_dir / "shap_summary.png"
        plt.figure(figsize=(9.5, 5.5))
        shap.summary_plot(shap_values, X_trans_df, max_display=10, show=False)
        plt.title("SHAP Summary Plot (Top 10 Features, n=200 Sample)", fontweight="bold", color=NAVY)
        plt.tight_layout()
        plt.savefig(summary_path, dpi=180, bbox_inches="tight")
        plt.close("all")

        # 2. Clean bar chart of mean |SHAP value| in INR
        bar_path = charts_dir / "shap_importance_bar.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_df = top10_shap.iloc[::-1]
        bars = ax.barh(plot_df["feature"], plot_df["mean_abs_shap"] / 1000.0, color=NAVY)
        ax.set_title("Global SHAP Feature Impact (Mean |SHAP Value|)", fontweight="bold", color=NAVY)
        ax.set_xlabel("Average Impact on Predicted Price (INR Thousands)")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 1.5, bar.get_y() + bar.get_height() / 2, f"Rs. {w:.1f}k", va="center", fontsize=8.5, color=NAVY)
        ax.set_xlim(0, (top10_shap["mean_abs_shap"].max() / 1000.0) * 1.18)
        fig.tight_layout()
        fig.savefig(bar_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return {
            "shap_available": True,
            "method": "SHAP TreeExplainer" if hasattr(estimator, "estimators_") else "SHAP LinearExplainer",
            "sample_size": int(len(X_sample)),
            "top10_shap": top10_shap.to_dict(orient="records"),
            "summary_chart_path": str(summary_path),
            "bar_chart_path": str(bar_path),
            "fallback_note": None,
        }

    except Exception as exc:
        # Documented permutation importance fallback
        perm = permutation_importance(
            pipeline, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
        perm_df = pd.DataFrame(
            {
                "feature": [_prettify_feature_name(c) for c in X_test.columns],
                "mean_abs_shap": perm.importances_mean,
            }
        ).sort_values("mean_abs_shap", ascending=False)

        fallback_path = charts_dir / "shap_fallback_permutation.png"
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_df = perm_df.head(10).iloc[::-1]
        ax.barh(plot_df["feature"], plot_df["mean_abs_shap"], color=CORAL)
        ax.set_title("Permutation Importance (SHAP Fallback)", fontweight="bold", color=NAVY)
        ax.set_xlabel("Mean R2 Decrease on Permutation")
        fig.tight_layout()
        fig.savefig(fallback_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        return {
            "shap_available": False,
            "method": "Permutation Importance Fallback",
            "sample_size": int(len(X_test)),
            "top10_shap": perm_df.head(10).to_dict(orient="records"),
            "summary_chart_path": str(fallback_path),
            "bar_chart_path": str(fallback_path),
            "fallback_note": f"SHAP unavailable ({exc}); used Permutation Importance fallback.",
        }


def explain_single_prediction(
    pipeline: Pipeline,
    input_df: pd.DataFrame,
    top_k: int = 6,
) -> Dict[str, Any]:
    """Compute local SHAP feature contributions for a single vehicle prediction."""
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["model"]

    try:
        import shap

        X_transformed = preprocessor.transform(input_df)
        raw_names = list(preprocessor.get_feature_names_out())
        base_names = [_map_to_base_feature(n) for n in raw_names]

        if hasattr(estimator, "estimators_") or estimator.__class__.__name__ == "DecisionTreeRegressor":
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(X_transformed, check_additivity=False)
            expected_val = explainer.expected_value
        else:
            return {"available": False, "contributions": [], "note": "Local SHAP only enabled for tree models."}

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = float(np.asarray(expected_val).ravel()[0])
        else:
            expected_val = float(expected_val)

        row_shap = np.asarray(shap_vals[0], dtype=float)

        # Aggregate one-hot contributions back to original input feature names for clarity
        grouped: Dict[str, float] = {}
        for base, val in zip(base_names, row_shap):
            grouped[base] = grouped.get(base, 0.0) + float(val)

        contrib_list: List[Dict[str, Any]] = []
        for base, impact in grouped.items():
            raw_val = input_df.iloc[0][base] if base in input_df.columns else ""
            contrib_list.append(
                {
                    "feature": _prettify_feature_name(base),
                    "input_value": str(raw_val),
                    "shap_impact_inr": float(impact),
                    "abs_impact": abs(float(impact)),
                }
            )

        contrib_list.sort(key=lambda d: d["abs_impact"], reverse=True)
        return {
            "available": True,
            "base_value_inr": expected_val,
            "contributions": contrib_list[:top_k],
            "note": "SHAP local explanation aggregated by vehicle attribute.",
        }

    except Exception as exc:
        return {
            "available": False,
            "contributions": [],
            "note": f"Local explanation unavailable ({exc}).",
        }
