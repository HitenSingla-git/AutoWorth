"""Model evaluation, depreciation analytics, and visualization utilities."""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# Consistent navy/blue academic-fintech palette
NAVY = "#0F2942"
PRIMARY_BLUE = "#1D6FA5"
ACCENT_TEAL = "#2E9CCA"
LIGHT_BLUE = "#A9D6E5"
SLATE = "#4A5568"
CORAL = "#E05A47"

sns.set_theme(style="whitegrid", font_scale=1.0)


def compute_regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute R2, MAE, and RMSE for regression predictions."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
    }


def _save_fig(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_eda_charts(cleaned_df: pd.DataFrame, charts_dir: Path) -> Dict[str, str]:
    """Generate Exploratory Data Analysis charts from the cleaned dataset."""
    charts: Dict[str, str] = {}

    # 1. Selling price distribution (standard + log/capped view)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    prices_lakh = cleaned_df["selling_price"] / 100_000.0
    sns.histplot(prices_lakh, bins=40, kde=True, color=PRIMARY_BLUE, ax=axes[0])
    axes[0].set_title("Selling Price Distribution (All Cars)", fontweight="bold", color=NAVY)
    axes[0].set_xlabel("Selling Price (INR Lakhs)")
    axes[0].set_ylabel("Number of Listings")

    capped_lakh = prices_lakh[prices_lakh <= 25]
    sns.histplot(capped_lakh, bins=35, kde=True, color=ACCENT_TEAL, ax=axes[1])
    axes[1].set_title("Selling Price Distribution (<= 25 Lakhs)", fontweight="bold", color=NAVY)
    axes[1].set_xlabel("Selling Price (INR Lakhs)")
    axes[1].set_ylabel("Number of Listings")
    charts["eda_price_distribution"] = _save_fig(fig, charts_dir / "eda_price_distribution.png")

    # 2. Correlation heatmap of numerical features
    corr_cols = ["selling_price", "vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]
    corr_matrix = cleaned_df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.set_title("Correlation Matrix of Numerical Features", fontweight="bold", color=NAVY)
    charts["eda_correlation_heatmap"] = _save_fig(fig, charts_dir / "eda_correlation_heatmap.png")

    # 3. Top 12 brands by listing count and average price
    top_brands = cleaned_df["brand"].value_counts().head(12).index
    brand_stats = (
        cleaned_df[cleaned_df["brand"].isin(top_brands)]
        .groupby("brand")["selling_price"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(brand_stats.index, brand_stats["mean"] / 100_000.0, color=PRIMARY_BLUE)
    ax.set_title("Average Selling Price Across Top 12 Brands", fontweight="bold", color=NAVY)
    ax.set_xlabel("Average Selling Price (INR Lakhs)")
    ax.set_ylabel("Brand")
    for bar, (_, row) in zip(bars, brand_stats.iterrows()):
        ax.text(
            bar.get_width() + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"Rs. {row['mean']/100_000:.2f}L (n={int(row['count'])})",
            va="center",
            fontsize=8.5,
            color=NAVY,
        )
    ax.set_xlim(0, (brand_stats["mean"].max() / 100_000.0) * 1.25)
    charts["eda_top_brands"] = _save_fig(fig, charts_dir / "eda_top_brands.png")

    return charts


def generate_depreciation_charts(cleaned_df: pd.DataFrame, charts_dir: Path) -> Dict[str, Any]:
    """Analyze vehicle depreciation trends and save charts."""
    charts: Dict[str, str] = {}

    # 1. Vehicle age vs selling price (average trend + scatter)
    age_summary = (
        cleaned_df[cleaned_df["vehicle_age"] <= 20]
        .groupby("vehicle_age")["selling_price"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sample_df = cleaned_df[cleaned_df["vehicle_age"] <= 20].sample(
        n=min(1500, len(cleaned_df)), random_state=42
    )
    ax.scatter(
        sample_df["vehicle_age"],
        sample_df["selling_price"] / 100_000.0,
        alpha=0.25,
        color=LIGHT_BLUE,
        s=18,
        label="Individual Listings (Sample)",
    )
    ax.plot(
        age_summary["vehicle_age"],
        age_summary["mean"] / 100_000.0,
        color=NAVY,
        marker="o",
        linewidth=2.3,
        label="Mean Selling Price",
    )
    ax.plot(
        age_summary["vehicle_age"],
        age_summary["median"] / 100_000.0,
        color=CORAL,
        marker="s",
        linestyle="--",
        linewidth=1.8,
        label="Median Selling Price",
    )
    ax.set_title("Vehicle Age vs. Resale Selling Price", fontweight="bold", color=NAVY)
    ax.set_xlabel("Vehicle Age (Years, relative to 2020)")
    ax.set_ylabel("Selling Price (INR Lakhs)")
    ax.set_ylim(0, min(40, (cleaned_df["selling_price"].quantile(0.98) / 100_000.0) * 1.2))
    ax.legend()
    charts["depreciation_age_vs_price"] = _save_fig(
        fig, charts_dir / "depreciation_age_vs_price.png"
    )

    # 2. Kilometres driven vs selling price (binned + scatter)
    km_bins = [0, 25000, 50000, 75000, 100000, 150000, 500000]
    km_labels = ["0-25k", "25k-50k", "50k-75k", "75k-100k", "100k-150k", "150k+"]
    temp_df = cleaned_df.copy()
    temp_df["km_bucket"] = pd.cut(temp_df["km_driven"], bins=km_bins, labels=km_labels)
    km_summary = temp_df.groupby("km_bucket", observed=False)["selling_price"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(km_summary["km_bucket"].astype(str), km_summary["selling_price"] / 100_000.0, color=PRIMARY_BLUE)
    ax.set_title("Kilometres Driven vs. Average Resale Price", fontweight="bold", color=NAVY)
    ax.set_xlabel("Kilometres Driven Bucket")
    ax.set_ylabel("Average Selling Price (INR Lakhs)")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.12, f"Rs. {h:.2f}L", ha="center", fontsize=9, color=NAVY)
    charts["depreciation_km_vs_price"] = _save_fig(
        fig, charts_dir / "depreciation_km_vs_price.png"
    )

    # 3. Average selling price by fuel type
    fuel_summary = (
        cleaned_df.groupby("fuel")["selling_price"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(fuel_summary["fuel"], fuel_summary["mean"] / 100_000.0, color=[NAVY, PRIMARY_BLUE, ACCENT_TEAL, LIGHT_BLUE][:len(fuel_summary)])
    ax.set_title("Average Selling Price by Fuel Type", fontweight="bold", color=NAVY)
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Average Selling Price (INR Lakhs)")
    for bar, (_, row) in zip(bars, fuel_summary.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.15,
            f"Rs. {h:.2f}L\n(n={int(row['count'])})",
            ha="center",
            fontsize=9,
            color=NAVY,
        )
    ax.set_ylim(0, (fuel_summary["mean"].max() / 100_000.0) * 1.22)
    charts["depreciation_fuel_type"] = _save_fig(fig, charts_dir / "depreciation_fuel_type.png")

    # 4. Average selling price by transmission
    trans_summary = (
        cleaned_df.groupby("transmission")["selling_price"]
        .agg(["mean", "median", "count"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.8))
    bars = ax.bar(trans_summary["transmission"], trans_summary["mean"] / 100_000.0, color=[NAVY, ACCENT_TEAL][:len(trans_summary)], width=0.5)
    ax.set_title("Average Selling Price by Transmission Type", fontweight="bold", color=NAVY)
    ax.set_xlabel("Transmission")
    ax.set_ylabel("Average Selling Price (INR Lakhs)")
    for bar, (_, row) in zip(bars, trans_summary.iterrows()):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.25,
            f"Rs. {h:.2f}L\n(n={int(row['count'])})",
            ha="center",
            fontsize=9.5,
            color=NAVY,
        )
    ax.set_ylim(0, (trans_summary["mean"].max() / 100_000.0) * 1.22)
    charts["depreciation_transmission"] = _save_fig(
        fig, charts_dir / "depreciation_transmission.png"
    )

    return {
        "charts": charts,
        "age_summary": age_summary.to_dict(orient="records"),
        "fuel_summary": fuel_summary.to_dict(orient="records"),
        "transmission_summary": trans_summary.to_dict(orient="records"),
    }


def generate_model_comparison_charts(
    results: List[Dict[str, Any]],
    y_test: pd.Series,
    best_predictions: np.ndarray,
    best_model_name: str,
    charts_dir: Path,
) -> Dict[str, str]:
    """Generate R2, MAE/RMSE, and Actual vs Predicted comparison charts."""
    charts: Dict[str, str] = {}
    res_df = pd.DataFrame(results)

    # 1. R2 comparison chart
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [NAVY if m == best_model_name else LIGHT_BLUE for m in res_df["model_name"]]
    bars = ax.bar(res_df["model_name"], res_df["r2"], color=colors, width=0.52)
    ax.set_title("Model Comparison: R2 Score (Higher is Better)", fontweight="bold", color=NAVY)
    ax.set_ylabel("R2 Score")
    ax.set_ylim(0, 1.05)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.4f}", ha="center", fontweight="bold", color=NAVY)
    charts["model_comparison_r2"] = _save_fig(fig, charts_dir / "model_comparison_r2.png")

    # 2. MAE and RMSE comparison chart
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bars_mae = axes[0].bar(res_df["model_name"], res_df["mae"] / 1000.0, color=PRIMARY_BLUE, width=0.52)
    axes[0].set_title("Mean Absolute Error (Lower is Better)", fontweight="bold", color=NAVY)
    axes[0].set_ylabel("MAE (INR Thousands)")
    axes[0].tick_params(axis="x", rotation=12)
    for bar in bars_mae:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 2.5, f"Rs. {h:.1f}k", ha="center", fontsize=9, color=NAVY)
    axes[0].set_ylim(0, (res_df["mae"].max() / 1000.0) * 1.18)

    bars_rmse = axes[1].bar(res_df["model_name"], res_df["rmse"] / 1000.0, color=ACCENT_TEAL, width=0.52)
    axes[1].set_title("Root Mean Squared Error (Lower is Better)", fontweight="bold", color=NAVY)
    axes[1].set_ylabel("RMSE (INR Thousands)")
    axes[1].tick_params(axis="x", rotation=12)
    for bar in bars_rmse:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 4.0, f"Rs. {h:.1f}k", ha="center", fontsize=9, color=NAVY)
    axes[1].set_ylim(0, (res_df["rmse"].max() / 1000.0) * 1.18)

    charts["model_comparison_errors"] = _save_fig(fig, charts_dir / "model_comparison_errors.png")

    # 3. Actual vs Predicted scatter for best model
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    y_true_lakh = np.asarray(y_test) / 100_000.0
    y_pred_lakh = np.asarray(best_predictions) / 100_000.0
    ax.scatter(y_true_lakh, y_pred_lakh, alpha=0.45, color=PRIMARY_BLUE, s=22, label="Test Predictions")
    max_val = max(y_true_lakh.max(), y_pred_lakh.max())
    ax.plot([0, max_val], [0, max_val], color=CORAL, linestyle="--", linewidth=2, label="Ideal Fit (y = x)")
    ax.set_title(f"Actual vs. Predicted Selling Price ({best_model_name})", fontweight="bold", color=NAVY)
    ax.set_xlabel("Actual Selling Price (INR Lakhs)")
    ax.set_ylabel("Predicted Selling Price (INR Lakhs)")
    ax.legend()
    charts["actual_vs_predicted"] = _save_fig(fig, charts_dir / "actual_vs_predicted.png")

    return charts
