"""Shared UI helpers for the DDI explorer pages."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import data as D

PAGE_ICON = "💊"


def page_setup(title: str, icon: str = PAGE_ICON) -> None:
    st.set_page_config(page_title=f"{title} · DDI Explorer", page_icon=icon,
                       layout="wide")


def require_db() -> None:
    """Stop the page with build instructions if the SQLite DB is missing."""
    if not D.db_exists():
        st.error("The explorer database has not been built yet.")
        st.markdown(
            "Run this once from the project root:\n\n"
            "```bash\npython explorer/build_db.py\n```\n\n"
            "It builds `results/ddi_study/ddi.db` from the CSV/JSON result files."
        )
        st.stop()


@st.cache_data(show_spinner=False)
def _drug_label_lookup() -> tuple[list[str], dict]:
    df = D.drug_options()
    labels = df["label"].tolist()
    label_to_id = dict(zip(df["label"], df["dbid"]))
    return labels, label_to_id


def drug_picker(label: str, key: str, default_index: int = 0) -> str | None:
    """Searchable selectbox of drugs; returns the chosen DrugBank ID."""
    labels, label_to_id = _drug_label_lookup()
    if not labels:
        st.warning("No drugs available.")
        return None
    choice = st.selectbox(label, labels, index=min(default_index, len(labels) - 1),
                          key=key)
    return label_to_id.get(choice)


def reaction_picker(label: str, key: str) -> str | None:
    rxns = D.reaction_options()
    if not rxns:
        return None
    return st.selectbox(label, rxns, key=key)


def download_df(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if df is None or df.empty:
        return
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")


def fmt_ror(x: float) -> str:
    if x is None:
        return "—"
    if x >= 1e6:
        return f"{x / 1e6:.2f}M"
    if x >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.2f}"


def metric_row(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.metric(label, value)
