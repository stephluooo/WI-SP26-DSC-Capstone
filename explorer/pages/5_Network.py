"""F6 Drug interaction network graph (ego network around a focus drug)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui

ui.page_setup("Network")
st.title("Drug interaction network")
st.caption("An ego network around one drug: it sits at the centre, its most-signalled "
           "partner drugs surround it, and edges connect any two drugs that also "
           "co-report together.")
ui.require_db()

c1, c2 = st.columns([3, 1])
with c1:
    focus = ui.drug_picker("Focus drug", key="net_focus")
with c2:
    k = st.slider("Partners", 5, 40, 18)

if not focus:
    st.stop()

nodes, edges = D.subnetwork(focus, k=k)
if nodes.empty:
    st.info("No network for this drug.")
    st.stop()

# Circular layout: focus at centre, partners on a ring.
pos = {}
ids = nodes["dbid"].tolist()
pos[focus] = (0.0, 0.0)
ring = [i for i in ids if i != focus]
for j, dbid in enumerate(ring):
    ang = 2 * math.pi * j / max(len(ring), 1)
    pos[dbid] = (math.cos(ang), math.sin(ang))

# Edges
edge_x, edge_y = [], []
wmax = edges["weight"].max() if not edges.empty else 1
for _, e in edges.iterrows():
    if e["drug_a"] in pos and e["drug_b"] in pos and e["drug_a"] != e["drug_b"]:
        x0, y0 = pos[e["drug_a"]]
        x1, y1 = pos[e["drug_b"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                        line=dict(width=1, color="rgba(120,120,120,0.4)"),
                        hoverinfo="none")

nx_, ny_, text, size, color = [], [], [], [], []
nm = dict(zip(nodes["dbid"], nodes["name"]))
nsig = dict(zip(nodes["dbid"], nodes["n_signals"]))
for dbid in ids:
    x, y = pos[dbid]
    nx_.append(x); ny_.append(y)
    text.append(nm[dbid])
    is_focus = dbid == focus
    size.append(34 if is_focus else 16 + 14 * (nsig.get(dbid, 0) / max(nsig.values()) if nsig else 0))
    color.append("#d62728" if is_focus else "#1f77b4")

node_trace = go.Scatter(
    x=nx_, y=ny_, mode="markers+text", text=text, textposition="top center",
    textfont=dict(size=11),
    marker=dict(size=size, color=color, line=dict(width=1.5, color="white")),
    hovertext=[f"{nm[d]} ({d})<br>{nsig.get(d,0)} signals" for d in ids],
    hoverinfo="text",
)

fig = go.Figure([edge_trace, node_trace])
fig.update_layout(
    showlegend=False, height=680,
    xaxis=dict(visible=False), yaxis=dict(visible=False),
    margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
)
st.plotly_chart(fig, width='stretch')

st.caption(f"Centre (red) = {D.name_for(focus)}. Node size ∝ number of reaction signals. "
           f"{len(edges):,} edges among {len(nodes):,} drugs.")
ui.download_df(edges, f"network_{focus}_edges.csv", "Download edges (CSV)")
