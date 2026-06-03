"""F7 Drug x reaction signal heatmap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Heatmap")
st.title("Drug × reaction heatmap")
st.caption("For a chosen drug, how strongly each of its partners signals across that "
           "drug's most common reactions.")
ui.require_db()

focus = ui.drug_picker("Focus drug", key="hm_focus")
if not focus:
    st.stop()

n_partners = st.slider("Partners (rows)", 5, 30, 15)
n_rxn = st.slider("Reactions (columns)", 5, 25, 12)
metric = st.radio("Cell value", ["ror", "a"], horizontal=True,
                  format_func={"ror": "ROR (log)", "a": "Co-reports (a)"}.get)

partners = D.drug_partner_summary(focus, top=n_partners)
reactions = D.drug_reactions(focus, top=n_rxn)
if partners.empty or reactions.empty:
    st.info("Not enough data for a heatmap.")
    st.stop()

partner_ids = partners["partner"].tolist()
rxn_list = reactions["reaction"].tolist()

# Build matrix by querying pair signals for each partner, filtered to chosen reactions.
nm = dict(zip(partners["partner"], partners["partner_name"]))
rows = []
for pid in partner_ids:
    sig = D.pair_signals(focus, pid)
    sig = sig[sig["reaction"].isin(rxn_list)]
    vals = {"partner": nm.get(pid, pid)}
    for _, r in sig.iterrows():
        vals[r["reaction"]] = r[metric]
    rows.append(vals)

mat = pd.DataFrame(rows).set_index("partner")
mat = mat.reindex(columns=rxn_list)
if mat.dropna(how="all").empty:
    st.info("No overlapping signals to display.")
    st.stop()

z = mat.values.astype(float)
if metric == "ror":
    with np.errstate(divide="ignore"):
        z = np.log10(z)
    colorbar_title = "log10(ROR)"
else:
    colorbar_title = "a"

fig = px.imshow(
    z, x=rxn_list, y=mat.index.tolist(), aspect="auto",
    color_continuous_scale="Viridis", labels=dict(color=colorbar_title),
)
fig.update_xaxes(tickangle=40)
fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')
st.caption(f"Rows: partners of {D.name_for(focus)}. Columns: {focus}'s top reactions. "
           "Blank cells = no signal for that partner/reaction.")
ui.download_df(mat.reset_index(), f"heatmap_{focus}.csv", "Download matrix (CSV)")
