"""F10 Novel-prediction browser (with DrugBank known-DDI flagging)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Predictions")
st.title("Novel-prediction browser")
st.caption("The model's highest-confidence predicted interactions for drug pairs that "
           "were never co-reported together in FAERS.")
ui.require_db()

df = D.novel_predictions()
if df.empty:
    st.info("No novel predictions table found.")
    st.stop()

df = df.copy()
df["name_a"] = df.get("drug_a_name", df["drug_a"]).fillna(df["drug_a"])
df["name_b"] = df.get("drug_b_name", df["drug_b"]).fillna(df["drug_b"])
df["pair"] = df["name_a"] + " + " + df["name_b"]

# Flag which predictions are already documented in DrugBank.
flagged = False
if D.known_ddi_available():
    keys = tuple("|".join(sorted([a, b])) for a, b in zip(df["drug_a"], df["drug_b"]))
    known = D.known_keys_for(keys)
    df["in_drugbank"] = [k in known for k in keys]
    flagged = True

c1, c2, c3 = st.columns(3)
c1.metric("Predictions", f"{len(df):,}")
c2.metric("Min probability", f"{df['predicted_probability'].min():.3f}")
if flagged:
    n_known = int(df["in_drugbank"].sum())
    c3.metric("Already in DrugBank", f"{n_known} ({n_known/len(df)*100:.1f}%)")

with st.sidebar:
    st.header("Filters")
    min_p = st.slider("Minimum probability", 0.90, 1.0, 0.95, step=0.005)
    if flagged:
        show = st.radio("Show", ["All", "Only novel (not in DrugBank)",
                                 "Only confirmed (in DrugBank)"])
    else:
        show = "All"
        st.caption("DrugBank known-DDI table not loaded — per-row confirmation unavailable.")

view = df[df["predicted_probability"] >= min_p]
if flagged and show.startswith("Only novel"):
    view = view[~view["in_drugbank"]]
elif flagged and show.startswith("Only confirmed"):
    view = view[view["in_drugbank"]]

cols = ["pair", "predicted_probability"]
if flagged:
    cols.append("in_drugbank")
cols += ["drug_a", "drug_b"]
st.dataframe(view[cols], width='stretch', hide_index=True, height=520)
ui.download_df(view[cols], "novel_predictions.csv", "Download (CSV)")

# (1 - p) log scale chart, matching the paper's figure.
st.subheader("Top 20 by confidence")
plot = view.head(20).copy()
plot["one_minus_p"] = (1 - plot["predicted_probability"]).clip(lower=1e-6)
plot = plot.iloc[::-1]
color = plot["in_drugbank"].map({True: "in DrugBank", False: "novel"}) if flagged else None
fig = px.bar(plot, x="one_minus_p", y="pair", orientation="h", log_x=True,
             color=color,
             labels={"one_minus_p": "1 − probability (log; smaller = more confident)",
                     "pair": "", "color": ""})
fig.update_xaxes(autorange="reversed")
fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')
