"""Streamlit entry point for the No-Man interactive report prototype."""

from __future__ import annotations

import streamlit as st

from components import (
    filter_chains,
    render_chain_list,
    render_filters,
    render_header,
    render_network,
    render_overview,
)
from dummy_data import DUMMY_REPORT
from i18n import LANGUAGES, t
from report_loader import load_uploaded_report
from styles import apply_page_styles


def main() -> None:
    st.set_page_config(
        page_title="No-Man Interactive Report",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_page_styles()

    lang_label = st.sidebar.radio("Language / 言語", list(LANGUAGES), horizontal=True)
    lang = LANGUAGES[lang_label]

    source = st.sidebar.radio(
        t("source", lang),
        [t("dummy", lang), t("upload", lang)],
    )

    report = DUMMY_REPORT
    if source == t("upload", lang):
        uploaded = st.sidebar.file_uploader(
            t("upload_prompt", lang),
            type=["json"],
            help=t("upload_help", lang),
        )
        if uploaded is not None:
            try:
                report = load_uploaded_report(uploaded)
                st.sidebar.success(t("loaded_upload", lang))
            except ValueError as exc:
                st.sidebar.error(str(exc))
        else:
            st.sidebar.info(t("loaded_dummy", lang))
    else:
        st.sidebar.info(t("loaded_dummy", lang))

    chains = report.get("chains", [])

    render_header(report, lang)
    render_overview(report, lang)

    st.divider()
    show_labels = st.checkbox(t("show_node_labels", lang), value=False)
    render_network(chains, lang, show_labels=show_labels)

    st.divider()
    st.subheader(t("chains", lang))
    confidence_filter, severity_filter = render_filters(chains, lang)
    filtered = filter_chains(chains, confidence_filter, severity_filter, lang)
    render_chain_list(filtered, lang)


if __name__ == "__main__":
    main()
