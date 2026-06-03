"""F3 Reaction explorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Reaction")
st.title("Reaction explorer")
st.caption("Pick an adverse reaction to see which drug pairs signal for it most strongly.")
ui.require_db()

rxn = ui.reaction_picker("Reaction", key="rxn_pick")
if not rxn:
    st.stop()

top = st.slider("How many top pairs to show", 20, 1000, 200, step=20)
df = D.reaction_pairs(rxn, top=top)

if df.empty:
    st.info("No signals for this reaction.")
    st.stop()

ui.metric_row([
    ("Pairs shown", f"{len(df):,}"),
    ("Max ROR", ui.fmt_ror(df["ror"].max())),
    ("Max co-reports (a)", f"{int(df['a'].max()):,}"),
])

df = df.copy()
df["pair"] = df["drug_a_name"].fillna(df["drug_a"]) + " + " + df["drug_b_name"].fillna(df["drug_b"])

st.subheader(f"Drug pairs signalling: {rxn}")
st.dataframe(
    df[["pair", "drug_a", "drug_b", "a", "b", "c", "d", "ror", "ci_low", "ci_high"]],
    width='stretch', hide_index=True,
)
ui.download_df(df, f"reaction_{rxn[:30].replace(' ', '_')}.csv")

plot = df.head(25).iloc[::-1]
fig = px.bar(plot, x="ror", y="pair", orientation="h", log_x=True,
             labels={"ror": "ROR (log scale)", "pair": ""},
             hover_data=["a", "ci_low"])
fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')
