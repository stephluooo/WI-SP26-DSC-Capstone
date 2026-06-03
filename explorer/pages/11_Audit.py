"""F14 Canonicalization audit viewer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Audit")
st.title("Canonicalization audit")
st.caption("How raw FAERS drug names were matched to DrugBank IDs. Use this to sanity-"
           "check the name-mapping that underpins every signal.")
ui.require_db()

if not D.has_table("match_details"):
    st.info("No `match_details` table found.")
    st.stop()

stats = D.match_stats()
ui.metric_row([
    ("Raw FAERS names", f"{stats['total']:,}"),
    ("Matched", f"{stats['matched']:,}"),
    ("Match rate", f"{stats['rate']*100:.1f}%"),
    ("Distinct DrugBank IDs", f"{stats['distinct_ids']:,}"),
])

pie = px.pie(values=[stats["matched"], stats["total"] - stats["matched"]],
             names=["Matched", "Unmatched"], hole=0.55,
             color_discrete_sequence=["#2ca02c", "#d62728"])
pie.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(pie, width='stretch')

st.divider()
st.subheader("Search the mapping")
c1, c2 = st.columns([3, 1])
with c1:
    search = st.text_input("FAERS name or DrugBank ID contains", "")
with c2:
    flt = st.selectbox("Filter", ["All", "Matched only", "Unmatched only"])

matched_only = {"All": None, "Matched only": True, "Unmatched only": False}[flt]
df = D.match_details(search=search, matched_only=matched_only, limit=2000)
st.dataframe(df, width='stretch', hide_index=True, height=480)
st.caption(f"{len(df):,} rows shown (limit 2,000).")
ui.download_df(df, "match_audit.csv", "Download results (CSV)")
