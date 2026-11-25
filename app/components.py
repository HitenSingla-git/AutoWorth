"""Reusable UI components and styling for the Streamlit Used Car Price Predictor."""

from typing import Any, Dict

import streamlit as st

NAVY = "#0F2942"
PRIMARY_BLUE = "#1D6FA5"
ACCENT_TEAL = "#2E9CCA"


def format_inr(amount: float) -> str:
    """Format an amount in Indian Rupee grouping (e.g., ₹4,50,000)."""
    value = round(float(amount))
    s = str(abs(value))
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        formatted = ",".join(reversed(groups)) + "," + last_three
    prefix = "-₹" if value < 0 else "₹"
    return f"{prefix}{formatted}"


def inject_custom_css() -> None:
    """Apply clean automotive/fintech navy-blue-white dashboard styling."""
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #F5F8FC;
                color: #1A202C;
            }}
            section[data-testid="stSidebar"] {{
                background-color: {NAVY};
            }}
            section[data-testid="stSidebar"] * {{
                color: #F7FAFC !important;
            }}
            section[data-testid="stSidebar"] code {{
                background-color: rgba(255, 255, 255, 0.16) !important;
                color: #FFFFFF !important;
                padding: 0.15rem 0.4rem;
                border-radius: 4px;
            }}
            div[data-testid="stFormSubmitButton"] button {{
                background: linear-gradient(120deg, {NAVY} 0%, {PRIMARY_BLUE} 100%) !important;
                border: none !important;
                color: #FFFFFF !important;
                font-weight: 700 !important;
                font-size: 1.05rem !important;
                padding: 0.7rem 1.25rem !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 12px rgba(15, 41, 66, 0.22) !important;
                transition: transform 0.12s ease, box-shadow 0.12s ease !important;
            }}
            div[data-testid="stFormSubmitButton"] button:hover {{
                box-shadow: 0 6px 16px rgba(29, 111, 165, 0.34) !important;
            }}
            div[data-testid="stForm"] {{
                background-color: #FFFFFF;
                border: 1px solid #CBD5E1 !important;
                border-radius: 12px !important;
                padding: 1.25rem 1.35rem !important;
                box-shadow: 0 2px 10px rgba(15, 41, 66, 0.05);
            }}
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] p,
            .stSelectbox label,
            .stNumberInput label {{
                color: {NAVY} !important;
                font-weight: 600 !important;
                font-size: 0.94rem !important;
                opacity: 1 !important;
            }}
            [data-testid="stCaptionContainer"],
            [data-testid="stCaptionContainer"] p {{
                color: #334155 !important;
                font-size: 0.82rem !important;
                line-height: 1.35 !important;
                opacity: 1 !important;
                margin-top: -0.2rem !important;
                margin-bottom: 0.45rem !important;
            }}
            .hero-banner {{
                background: linear-gradient(120deg, {NAVY} 0%, {PRIMARY_BLUE} 100%);
                color: #FFFFFF;
                padding: 1.6rem 2rem;
                border-radius: 12px;
                margin-bottom: 1.4rem;
                box-shadow: 0 6px 18px rgba(15, 41, 66, 0.14);
            }}
            .hero-banner h1 {{
                margin: 0 0 0.35rem 0;
                font-size: 1.85rem;
                font-weight: 700;
                color: #FFFFFF;
                letter-spacing: 0.3px;
            }}
            .hero-banner p {{
                margin: 0;
                font-size: 0.98rem;
                color: #E2EEF9;
            }}
            .kpi-card {{
                background: #FFFFFF;
                border: 1px solid #DCE6F2;
                border-top: 4px solid {PRIMARY_BLUE};
                border-radius: 10px;
                padding: 1rem 1.15rem;
                box-shadow: 0 2px 8px rgba(15, 41, 66, 0.05);
                margin-bottom: 0.8rem;
            }}
            .kpi-label {{
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #4A5568;
                margin-bottom: 0.25rem;
            }}
            .kpi-value {{
                font-size: 1.5rem;
                font-weight: 700;
                color: {NAVY};
            }}
            .kpi-sub {{
                font-size: 0.78rem;
                color: #718096;
                margin-top: 0.2rem;
            }}
            .prediction-box {{
                background: linear-gradient(135deg, #FFFFFF 0%, #EDF5FC 100%);
                border: 1.5px solid {PRIMARY_BLUE};
                border-radius: 12px;
                padding: 1.4rem 1.6rem;
                margin-top: 1rem;
                box-shadow: 0 4px 14px rgba(29, 111, 165, 0.12);
            }}
            .prediction-price {{
                font-size: 2.2rem;
                font-weight: 800;
                color: {NAVY};
                margin: 0.25rem 0;
            }}
            .workflow-step {{
                background: #FFFFFF;
                border: 1px solid #DCE6F2;
                border-left: 4px solid {ACCENT_TEAL};
                border-radius: 8px;
                padding: 0.85rem 1rem;
                margin-bottom: 0.6rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str) -> None:
    """Render top navy-blue gradient banner."""
    st.markdown(
        f"""
        <div class="hero-banner">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "") -> None:
    """Render a crisp KPI metric card."""
    sub_html = f'<div class="kpi-sub">{subtext}</div>' if subtext else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(result: Dict[str, Any]) -> None:
    """Display the predicted resale price card and disclaimer."""
    inr_text = format_inr(result["predicted_price_inr"])
    lakhs_text = f"{result['predicted_price_lakhs']:.2f} Lakhs"
    model_name = result["model_name"]

    st.markdown(
        f"""
        <div class="prediction-box">
            <div class="kpi-label">Estimated Used-Car Resale Value</div>
            <div class="prediction-price">{inr_text} <span style="font-size:1.1rem;color:{PRIMARY_BLUE};font-weight:600;">({lakhs_text})</span></div>
            <div class="kpi-sub">
                Model Used: <strong>{model_name}</strong> &bull;
                Note: This is a data-driven statistical estimate from historical CarDekho listings and not a guaranteed market quote.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
