"""Presentation helpers for the Streamlit report prototype."""

from __future__ import annotations

import streamlit as st


def apply_page_styles() -> None:
    """Apply lightweight CSS for dense, report-like scanning."""
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: #ffffff;
            color: #171a18;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 1.5rem;
        }

        h1, h2, h3, p, span, label, div {
            color: #171a18;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d3d8d4;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 1px 2px rgba(20, 28, 24, 0.04);
        }

        [data-testid="stMetric"] label,
        [data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: #303832 !important;
            font-weight: 650 !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #111411 !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetric"] [data-testid="stMetricValue"] div {
            color: #111411 !important;
        }

        .chain-path {
            border: 1px solid #d3d8d4;
            border-radius: 8px;
            padding: 0.85rem;
            background: #ffffff;
            line-height: 1.9;
        }

        .node-pill {
            display: inline-block;
            border: 1px solid #b8c2bb;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            margin: 0.08rem;
            background: white;
            white-space: nowrap;
        }

        .edge-sign {
            display: inline-block;
            color: #59625d;
            margin: 0 0.15rem;
            white-space: nowrap;
        }

        .chain-summary {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            border: 1px solid #d3d8d4;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.7rem 0.85rem;
            margin: 0.55rem 0 0.15rem;
        }

        .chain-summary-title {
            font-weight: 720;
            color: #171a18;
        }

        .tier-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.16rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 760;
            white-space: nowrap;
            border: 1px solid transparent;
        }

        .tier-well-supported {
            background: #e8f1ec;
            color: #1f5b3f;
            border-color: #b8d5c5;
        }

        .tier-plausible {
            background: #fff3d8;
            color: #704c00;
            border-color: #e7ca87;
        }

        .tier-worth-considering {
            background: #fbe7e3;
            color: #7a2e24;
            border-color: #e4b4aa;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tier_label(tier: str, lang: str = "en") -> str:
    labels = {
        "en": {
            "well-supported": "Well Supported",
            "plausible": "Plausible",
            "worth-considering": "Worth Considering",
        },
        "ja": {
            "well-supported": "十分な根拠あり",
            "plausible": "もっともらしい",
            "worth-considering": "検討に値する",
        },
    }
    fallback = "不明" if lang == "ja" else "Unknown"
    return labels.get(lang, labels["en"]).get(tier, tier or fallback)
