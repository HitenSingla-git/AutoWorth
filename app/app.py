"""Streamlit Web Application: USED CAR PRICE PREDICTOR."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from components import (
    NAVY,
    PRIMARY_BLUE,
    ACCENT_TEAL,
    format_inr,
    inject_custom_css,
    render_hero,
    render_kpi_card,
    render_prediction_result,
)
from src.data_loader import DATASET_SOURCE_URL, load_raw_dataset
from src.prediction import load_model_and_metadata, predict_car_price
from src.preprocessing import prepare_cleaned_dataframe

st.set_page_config(
    page_title="USED CAR PRICE PREDICTOR",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


@st.cache_resource
def get_model_artifacts():
    return load_model_and_metadata()


@st.cache_data
def get_cleaned_dataset():
    raw_df = load_raw_dataset()
    cleaned_df, _ = prepare_cleaned_dataframe(raw_df)
    return cleaned_df


def page_home(metadata: dict, cleaned_df: pd.DataFrame) -> None:
    render_hero(
        "USED CAR PRICE PREDICTOR",
        "Machine Learning-Based Resale Valuation, Model Benchmarking & Depreciation Analytics (CarDekho India Dataset)",
    )

    raw_sum = metadata["raw_dataset_summary"]
    clean_stats = metadata["cleaned_dataset_stats"]
    best_metrics = metadata["best_model_metrics"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "Dataset Records",
            f"{clean_stats['rows']:,}",
            f"From {raw_sum['rows']:,} raw CarDekho listings",
        )
    with col2:
        render_kpi_card(
            "Average Resale Price",
            format_inr(clean_stats["mean_price"]),
            f"Median: {format_inr(clean_stats['median_price'])}",
        )
    with col3:
        render_kpi_card(
            "Vehicle Coverage",
            f"{clean_stats['num_brands']} Brands",
            f"{clean_stats['num_models']} model groups ({clean_stats['year_min']}–{clean_stats['year_max']})",
        )
    with col4:
        render_kpi_card(
            "Best Model (Test R²)",
            f"{best_metrics['r2']:.4f}",
            f"{metadata['best_model_name']}",
        )

    st.markdown("### Project Overview & Workflow")
    left, right = st.columns([1.25, 1])
    with left:
        st.write(
            "This system predicts fair market resale prices for used cars in India using historical "
            "vehicle specifications from CarDekho. It compares three foundational regression algorithms "
            "(Linear Regression, Decision Tree Regressor, and Random Forest Regressor), studies vehicle "
            "depreciation patterns, and provides Explainable AI (SHAP) insights."
        )
        steps = [
            ("1. Data Acquisition & Cleaning", "Loaded 8,128 real Indian used-car listings; removed 1,202 exact duplicates and 3 extreme odometer anomalies (>500,000 km)."),
            ("2. Feature Engineering & Pipeline", "Extracted numeric mileage, engine (CC), max power (bhp), brand, car model, and vehicle age (relative to 2020) inside a leak-free Scikit-learn ColumnTransformer."),
            ("3. Model Training & Comparison", "Trained Linear Regression, Decision Tree, and Random Forest on an identical 80:20 split and evaluated using R², MAE, and RMSE."),
            ("4. Explainability & Analytics", "Computed Random Forest feature importance, SHAP global/local explanations, and depreciation curves across age, mileage, fuel, and transmission."),
        ]
        for title, desc in steps:
            st.markdown(
                f'<div class="workflow-step"><strong>{title}</strong><br><span style="font-size:0.9rem;color:#4A5568;">{desc}</span></div>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Available Vehicle Categories")
        cat_df = pd.DataFrame(
            {
                "Category": ["Fuel Types", "Transmission", "Seller Types", "Ownership Tiers"],
                "Available Values": [
                    ", ".join(clean_stats["fuel_types"]),
                    ", ".join(clean_stats["transmission_types"]),
                    ", ".join(clean_stats["seller_types"]),
                    ", ".join(clean_stats["owner_types"]),
                ],
            }
        )
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
        st.info(
            f"**Dataset Source:** CarDekho Vehicle Dataset (India) — [GitHub Mirror]({DATASET_SOURCE_URL}). "
            f"All prices are in Indian Rupees (INR)."
        )

    st.markdown("#### Sample Cleaned Dataset Records")
    preview_cols = [
        "brand", "car_model", "year", "vehicle_age", "km_driven",
        "fuel", "transmission", "owner", "mileage", "engine", "max_power", "seats", "selling_price",
    ]
    st.dataframe(cleaned_df[preview_cols].head(10), use_container_width=True, hide_index=True)


def page_prediction(pipeline, metadata: dict) -> None:
    render_hero(
        "USED CAR PRICE PREDICTOR — LIVE VALUATION",
        f"Enter vehicle specifications below to estimate fair resale value using {metadata['best_model_name']}.",
    )

    clean_stats = metadata["cleaned_dataset_stats"]
    medians = clean_stats["medians"]
    brand_to_models = clean_stats["brand_to_models"]
    brands = sorted(brand_to_models.keys())

    default_brand_idx = brands.index("Maruti") if "Maruti" in brands else 0

    col_brand, col_model = st.columns(2, gap="medium")
    with col_brand:
        selected_brand = st.selectbox(
            "Car Brand",
            options=brands,
            index=default_brand_idx,
            help="Select the vehicle manufacturer (e.g., Maruti, Hyundai, Honda).",
        )
        st.caption("Vehicle manufacturer (e.g., Maruti, Hyundai, Honda).")
    model_options = brand_to_models.get(selected_brand, [selected_brand]) + ["Other / Unlisted Model"]
    default_model_idx = model_options.index("Maruti Swift") if "Maruti Swift" in model_options else 0
    with col_model:
        selected_model = st.selectbox(
            "Car Model",
            options=model_options,
            index=default_model_idx,
            help="Select the car model family. Choose 'Other / Unlisted Model' if your variant is absent.",
        )
        st.caption("Specific car model; pick 'Other / Unlisted Model' if unlisted.")

    fuel_opts = clean_stats["fuel_types"]
    trans_opts = clean_stats["transmission_types"]
    seller_opts = clean_stats["seller_types"]

    with st.form("prediction_form"):
        # Row 1
        r1_c1, r1_c2, r1_c3 = st.columns(3, gap="medium")
        with r1_c1:
            year = st.number_input(
                "Manufacturing Year",
                min_value=1983,
                max_value=2025,
                value=2016,
                step=1,
                help="Year the vehicle was manufactured. Model was trained on 1983–2020 listings (reference year 2020).",
            )
            st.caption("Year the car was manufactured (dataset range: 1983–2020).")
        with r1_c2:
            owner = st.selectbox(
                "Ownership History",
                options=clean_stats["owner_types"],
                help="Number of previous registered owners of the vehicle.",
            )
            st.caption("Number of previous owners of the vehicle.")
        with r1_c3:
            mileage = st.number_input(
                "Fuel Efficiency (km/l)",
                min_value=5.0,
                max_value=45.0,
                value=float(round(medians["mileage"], 1)),
                step=0.5,
                help="ARAI/claimed fuel economy in kilometres per litre (km/l), or km/kg for CNG/LPG.",
            )
            st.caption("Distance the car can travel per litre of fuel (km/l).")

        # Row 2
        r2_c1, r2_c2, r2_c3 = st.columns(3, gap="medium")
        with r2_c1:
            km_driven = st.number_input(
                "Total Distance Driven (km)",
                min_value=100,
                max_value=500000,
                value=50000,
                step=1000,
                help="Total odometer reading in kilometres (km).",
            )
            st.caption("Total kilometres covered by the vehicle.")
        with r2_c2:
            seller_type = st.selectbox(
                "Seller Type",
                options=seller_opts,
                index=seller_opts.index("Individual") if "Individual" in seller_opts else 0,
                help="Whether the seller is an individual owner, dealer, or Trustmark certified dealer.",
            )
            st.caption("Whether the vehicle is sold by an individual or dealer.")
        with r2_c3:
            engine = st.number_input(
                "Engine Capacity (CC)",
                min_value=600,
                max_value=5000,
                value=int(round(medians["engine"])),
                step=50,
                help="Engine displacement volume measured in cubic centimetres (CC).",
            )
            st.caption("The size of the engine measured in cubic centimetres (CC).")

        # Row 3
        r3_c1, r3_c2, r3_c3 = st.columns(3, gap="medium")
        with r3_c1:
            fuel = st.selectbox(
                "Fuel Type",
                options=fuel_opts,
                index=fuel_opts.index("Petrol") if "Petrol" in fuel_opts else 0,
                help="Primary fuel type used by the engine (Petrol, Diesel, CNG, or LPG).",
            )
            st.caption("Primary fuel used by the vehicle engine.")
        with r3_c2:
            seats = st.selectbox(
                "Seating Capacity (Seats)",
                options=clean_stats["seat_options"],
                index=clean_stats["seat_options"].index(5) if 5 in clean_stats["seat_options"] else 0,
                help="Total number of passenger seats in the car.",
            )
            st.caption("Number of passenger seats available in the vehicle.")
        with r3_c3:
            max_power = st.number_input(
                "Maximum Engine Power (bhp)",
                min_value=30.0,
                max_value=450.0,
                value=float(round(medians["max_power"], 1)),
                step=2.0,
                help="Peak engine power output measured in brake horsepower (bhp).",
            )
            st.caption("Maximum power output produced by the engine (bhp).")

        # Row 4
        r4_c1, _, _ = st.columns(3, gap="medium")
        with r4_c1:
            transmission = st.selectbox(
                "Transmission Type",
                options=trans_opts,
                index=trans_opts.index("Manual") if "Manual" in trans_opts else 0,
                help="Gearbox type: Manual stick-shift or Automatic transmission.",
            )
            st.caption("Gearbox type (Manual or Automatic).")

        st.space("small")
        submitted = st.form_submit_button(
            "Predict Resale Price",
            type="primary",
            icon=":material/calculate:",
            width="stretch",
        )

    if submitted:
        car_model_val = selected_brand if selected_model == "Other / Unlisted Model" else selected_model
        payload = {
            "brand": selected_brand,
            "car_model": car_model_val,
            "year": year,
            "km_driven": km_driven,
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
        }
        try:
            result = predict_car_price(payload, pipeline=pipeline, metadata=metadata, include_explanation=True)
            render_prediction_result(result)

            expl = result["explanation"]
            if expl.get("available") and expl.get("contributions"):
                import plotly.express as px

                st.markdown("#### Key Factors Driving This Prediction (Local SHAP Explanation)")
                contrib_df = pd.DataFrame(expl["contributions"])
                contrib_df["direction"] = contrib_df["shap_impact_inr"].apply(
                    lambda v: "Increases Price (+)" if v >= 0 else "Decreases Price (-)"
                )
                contrib_df["impact_lakhs"] = contrib_df["shap_impact_inr"] / 100_000.0
                fig = px.bar(
                    contrib_df.iloc[::-1],
                    x="shap_impact_inr",
                    y="feature",
                    orientation="h",
                    color="direction",
                    color_discrete_map={
                        "Increases Price (+)": PRIMARY_BLUE,
                        "Decreases Price (-)": "#E05A47",
                    },
                    labels={"shap_impact_inr": "Price Impact (INR)", "feature": "Vehicle Attribute"},
                    title=f"Base Average Prediction: {format_inr(expl['base_value_inr'])}",
                )
                fig.update_layout(height=340, margin=dict(l=10, r=10, t=45, b=10))
                st.plotly_chart(fig, width="stretch")
        except ValueError as err:
            st.error(f"Invalid input: {err}")
        except Exception as err:
            st.error(f"Could not compute prediction: {err}")


def page_evaluation(metadata: dict) -> None:
    render_hero(
        "MODEL EVALUATION & ALGORITHM BENCHMARKING",
        "Comparison of Linear Regression, Decision Tree Regressor, and Random Forest Regressor on the held-out 20% test set.",
    )

    results = metadata["model_results"]
    split_cfg = metadata["split_config"]
    best_name = metadata["best_model_name"]
    best_m = metadata["best_model_metrics"]

    st.success(
        f"**Best Performing Model:** {best_name} — "
        f"**R² = {best_m['r2']:.4f}**, **MAE = {format_inr(best_m['mae'])}**, "
        f"**RMSE = {format_inr(best_m['rmse'])}** "
        f"(Evaluated on {split_cfg['test_rows']:,} test vehicles; trained on {split_cfg['train_rows']:,} vehicles)."
    )

    table_df = pd.DataFrame(
        [
            {
                "Algorithm": r["model_name"],
                "R² Score": round(r["r2"], 4),
                "MAE (INR)": format_inr(r["mae"]),
                "RMSE (INR)": format_inr(r["rmse"]),
                "Status": "Best Model" if r["model_name"] == best_name else "Baseline / Candidate",
            }
            for r in results
        ]
    )
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    import plotly.express as px

    res_df = pd.DataFrame(results)
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_r2 = px.bar(
            res_df,
            x="model_name",
            y="r2",
            text=res_df["r2"].map(lambda v: f"{v:.4f}"),
            color="model_name",
            color_discrete_sequence=[ACCENT_TEAL, PRIMARY_BLUE, NAVY],
            title="R² Score Comparison (Higher = Better)",
        )
        fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1.05], xaxis_title="", yaxis_title="R² Score")
        st.plotly_chart(fig_r2, use_container_width=True)

    with c2:
        fig_mae = px.bar(
            res_df,
            x="model_name",
            y="mae",
            text=res_df["mae"].map(lambda v: format_inr(v)),
            color="model_name",
            color_discrete_sequence=[ACCENT_TEAL, PRIMARY_BLUE, NAVY],
            title="Mean Absolute Error (Lower = Better)",
        )
        fig_mae.update_layout(showlegend=False, xaxis_title="", yaxis_title="MAE (INR)")
        st.plotly_chart(fig_mae, use_container_width=True)

    with c3:
        fig_rmse = px.bar(
            res_df,
            x="model_name",
            y="rmse",
            text=res_df["rmse"].map(lambda v: format_inr(v)),
            color="model_name",
            color_discrete_sequence=[ACCENT_TEAL, PRIMARY_BLUE, NAVY],
            title="Root Mean Squared Error (Lower = Better)",
        )
        fig_rmse.update_layout(showlegend=False, xaxis_title="", yaxis_title="RMSE (INR)")
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("#### Actual vs. Predicted Prices & Metric Definitions")
    col_img, col_def = st.columns([1.15, 1])
    with col_img:
        actual_pred_chart = Path(metadata["charts"]["actual_vs_predicted"])
        if actual_pred_chart.exists():
            st.image(str(actual_pred_chart), use_container_width=True)
    with col_def:
        st.markdown(
            """
            - **R² Score (Coefficient of Determination):** Measures the proportion of variance in used-car selling prices explained by the model. Closer to `1.0` indicates stronger predictive fit.
            - **MAE (Mean Absolute Error):** The average absolute difference between predicted and actual prices in INR—directly interpretable as typical rupee error.
            - **RMSE (Root Mean Squared Error):** Penalizes larger prediction errors more heavily than MAE, highlighting performance on higher-value outliers.
            """
        )


def page_analytics(metadata: dict, cleaned_df: pd.DataFrame) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    render_hero(
        "DEPRECIATION ANALYTICS & EXPLAINABLE AI (SHAP)",
        "Explore how vehicle age, odometer reading, fuel type, transmission, and technical specs relate to resale prices in the dataset.",
    )

    tab1, tab2, tab3 = st.tabs(["Depreciation Analysis", "Feature Importance", "SHAP Explainability"])

    with tab1:
        st.info("Note: These visualizations show statistical relationships observed in historical CarDekho listings and do not establish strict causation.")
        c1, c2 = st.columns(2)
        with c1:
            age_df = pd.DataFrame(metadata["depreciation_summary"]["age_summary"])
            fig_age = go.Figure()
            fig_age.add_trace(go.Scatter(
                x=age_df["vehicle_age"], y=age_df["mean"], mode="lines+markers",
                name="Mean Price", line=dict(color=NAVY, width=3)
            ))
            fig_age.add_trace(go.Scatter(
                x=age_df["vehicle_age"], y=age_df["median"], mode="lines+markers",
                name="Median Price", line=dict(color="#E05A47", dash="dash")
            ))
            fig_age.update_layout(
                title="Vehicle Age vs. Resale Price",
                xaxis_title="Vehicle Age (Years relative to 2020)",
                yaxis_title="Selling Price (INR)",
                height=380,
            )
            st.plotly_chart(fig_age, use_container_width=True)

        with c2:
            km_chart = Path(metadata["charts"]["depreciation_km_vs_price"])
            if km_chart.exists():
                st.image(str(km_chart), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fuel_df = pd.DataFrame(metadata["depreciation_summary"]["fuel_summary"])
            fig_fuel = px.bar(
                fuel_df, x="fuel", y="mean", text=fuel_df["mean"].map(format_inr),
                color="fuel", color_discrete_sequence=[NAVY, PRIMARY_BLUE, ACCENT_TEAL, "#7FB3D5"],
                title="Average Selling Price by Fuel Type",
            )
            fig_fuel.update_layout(showlegend=False, xaxis_title="Fuel Type", yaxis_title="Average Price (INR)", height=370)
            st.plotly_chart(fig_fuel, use_container_width=True)

        with c4:
            trans_df = pd.DataFrame(metadata["depreciation_summary"]["transmission_summary"])
            fig_trans = px.bar(
                trans_df, x="transmission", y="mean", text=trans_df["mean"].map(format_inr),
                color="transmission", color_discrete_sequence=[NAVY, ACCENT_TEAL],
                title="Average Selling Price by Transmission",
            )
            fig_trans.update_layout(showlegend=False, xaxis_title="Transmission", yaxis_title="Average Price (INR)", height=370)
            st.plotly_chart(fig_trans, use_container_width=True)

    with tab2:
        fi = metadata["feature_importance"]
        st.markdown(f"**Method:** {fi['method']}")
        col_a, col_b = st.columns([1.2, 1])
        with col_a:
            fi_chart = Path(fi["chart_path"])
            if fi_chart.exists():
                st.image(str(fi_chart), use_container_width=True)
        with col_b:
            st.markdown("#### Aggregated Importance by Input Attribute")
            grp_df = pd.DataFrame(fi["grouped_features"])
            grp_df["importance_pct"] = (grp_df["importance"] * 100).round(2)
            st.dataframe(
                grp_df[["feature", "importance_pct"]].rename(
                    columns={"feature": "Input Feature", "importance_pct": "Importance (%)"}
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tab3:
        shap_meta = metadata["shap_explainability"]
        st.markdown(
            f"**Explainability Method:** {shap_meta['method']} "
            f"(Evaluated on {shap_meta['sample_size']} test samples)"
        )
        if shap_meta.get("fallback_note"):
            st.warning(shap_meta["fallback_note"])

        s1, s2 = st.columns(2)
        with s1:
            sum_path = Path(shap_meta["summary_chart_path"])
            if sum_path.exists():
                st.image(str(sum_path), caption="SHAP Summary Plot", use_container_width=True)
        with s2:
            bar_path = Path(shap_meta["bar_chart_path"])
            if bar_path.exists():
                st.image(str(bar_path), caption="Mean |SHAP Value| Impact in INR", use_container_width=True)


def main() -> None:
    try:
        pipeline, metadata = get_model_artifacts()
        cleaned_df = get_cleaned_dataset()
    except Exception as exc:
        st.error(f"Startup error loading model artifacts or dataset: {exc}")
        st.stop()

    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select Section",
            [
                "Home / Overview",
                "Price Prediction",
                "Model Evaluation",
                "Analytics & Insights",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(f"**Active Model:**\n{metadata['best_model_name']}")
        st.markdown(f"**Test R² Score:** `{metadata['best_model_metrics']['r2']:.4f}`")
        st.markdown(f"**Dataset Size:** `{metadata['cleaned_dataset_stats']['rows']:,}` cars")

    if page == "Home / Overview":
        page_home(metadata, cleaned_df)
    elif page == "Price Prediction":
        page_prediction(pipeline, metadata)
    elif page == "Model Evaluation":
        page_evaluation(metadata)
    else:
        page_analytics(metadata, cleaned_df)


if __name__ == "__main__":
    main()
