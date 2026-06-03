"""F1 Drug-pair explorer + F9 contingency visualizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data as D
import ui

ui.page_setup("Drug Pair")
st.title("Drug-pair explorer")
st.caption("Pick two drugs to see every adverse reaction they are co-reported with, "
           "ranked by Reporting Odds Ratio.")
ui.require_db()

c1, c2 = st.columns(2)
with c1:
    a = ui.drug_picker("Drug A", key="pair_a", default_index=0)
with c2:
    b = ui.drug_picker("Drug B", key="pair_b", default_index=1)

if not a or not b:
    st.stop()
if a == b:
    st.warning("Pick two different drugs.")
    st.stop()

name_a, name_b = D.name_for(a), D.name_for(b)
df = D.pair_signals(a, b)

# Known-DDI banner
info = D.is_known_ddi(a, b)
if info is not None:
    if info["known"]:
        st.success(f"**{name_a} + {name_b}** is a **documented** DrugBank interaction.")
        if info["description"]:
            st.caption(info["description"])
    else:
        st.info(f"**{name_a} + {name_b}** is not in the DrugBank known-interaction list.")

if df.empty:
    st.warning(f"No FAERS co-report signals found for {name_a} + {name_b}.")
    st.stop()

ui.metric_row([
    ("Reaction signals", f"{len(df):,}"),
    ("Max ROR", ui.fmt_ror(df["ror"].max())),
    ("Max co-reports (a)", f"{int(df['a'].max()):,}"),
])

st.subheader(f"Reactions for {name_a} + {name_b}")
show = df.copy()
show["ror"] = show["ror"].round(2)
st.dataframe(show, width='stretch', hide_index=True)
ui.download_df(show, f"pair_{a}_{b}.csv")

# Top reactions bar chart (log x)
st.subheader("Top reactions by ROR")
topn = df.head(20).iloc[::-1]
fig = px.bar(topn, x="ror", y="reaction", orientation="h", log_x=True,
             labels={"ror": "ROR (log scale)", "reaction": ""},
             hover_data=["a", "ci_low", "ci_high"])
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, width='stretch')

# F9 contingency visualizer
st.divider()
st.subheader("Contingency table visualizer")
st.caption("Pick one reaction to see the 2×2 table behind its ROR.")
rxn = st.selectbox("Reaction", df["reaction"].tolist(), key="pair_rxn")
row = df[df["reaction"] == rxn].iloc[0]
a_, b_, c_, d_ = int(row["a"]), int(row["b"]), int(row["c"]), int(row["d"])

g1, g2 = st.columns([1, 1])
with g1:
    heat = go.Figure(data=go.Heatmap(
        z=[[a_, b_], [c_, d_]],
        x=["Reaction +", "Reaction −"],
        y=["Pair +", "Pair −"],
        text=[[f"a={a_:,}", f"b={b_:,}"], [f"c={c_:,}", f"d={d_:,}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False,
    ))
    heat.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10),
                       title="2×2 contingency")
    st.plotly_chart(heat, width='stretch')
with g2:
    st.markdown(
        f"""
**ROR formula**

$$\\mathrm{{ROR}} = \\dfrac{{a \\cdot d}}{{b \\cdot c}} = \\dfrac{{{a_:,} \\times {d_:,}}}{{{b_:,} \\times {c_:,}}}$$

- **a = {a_:,}** — reports with *both* the pair and this reaction
- **b = {b_:,}** — reports with the pair but *not* this reaction
- **c = {c_:,}** — reports with this reaction but *not* the pair
- **d = {d_:,}** — reports with neither

**ROR = {row['ror']:,.1f}**  ·  95% CI [{row['ci_low']:,.1f}, {row['ci_high']:,.1f}]
        """
    )
