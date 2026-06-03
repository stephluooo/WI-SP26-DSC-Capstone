"""F4 Global signal leaderboard + F15 export."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import data as D
import ui

ui.page_setup("Leaderboard")
st.title("Signal leaderboard")
st.caption("The strongest disproportionality signals across the whole study, with "
           "filters for statistical robustness.")
ui.require_db()

with st.sidebar:
    st.header("Filters")
    order_by = st.radio(
        "Rank by",
        options=["ror", "ci_low", "a"],
        format_func={"ror": "Absolute ROR", "ci_low": "CI lower bound (robust)",
                     "a": "Evidence (# co-reports a)"}.get,
    )
    min_a = st.slider("Minimum co-reports (a)", 3, 200, 10)
    min_ci = st.number_input("Minimum CI lower bound", value=1.5, step=0.5)
    rxn_like = st.text_input("Reaction contains", "")
    limit = st.select_slider("Rows", options=[100, 500, 1000, 5000, 10000], value=1000)

st.markdown(
    "**Why two rankings?** Absolute ROR rewards rare events (tiny `a`,`c`). Ranking by "
    "the **CI lower bound** demotes those low-evidence outliers — analogous to the "
    "bootstrap re-ranking in the paper."
)

df = D.leaderboard(order_by=order_by, min_a=min_a, min_ci_low=min_ci,
                   reaction_like=rxn_like, limit=int(limit))

if df.empty:
    st.info("No signals match these filters.")
    st.stop()

view = df.copy()
view["pair"] = view["drug_a_name"].fillna(view["drug_a"]) + " + " + \
    view["drug_b_name"].fillna(view["drug_b"])
view["ror"] = view["ror"].round(2)
cols = ["pair", "reaction", "a", "b", "c", "d", "ror", "ci_low", "ci_high",
        "drug_a", "drug_b"]
st.dataframe(view[cols], width='stretch', hide_index=True, height=620)
ui.download_df(view[cols], "leaderboard.csv", "Download filtered leaderboard (CSV)")
st.caption(f"{len(df):,} rows shown (limit {limit:,}).")
