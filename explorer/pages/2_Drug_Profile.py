"""F2 Single-drug profile."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Drug Profile")
st.title("Single-drug profile")
st.caption("Everything a drug is involved in: its most-signalled partner drugs and "
           "the reactions it appears with.")
ui.require_db()

dbid = ui.drug_picker("Drug", key="profile_drug")
if not dbid:
    st.stop()
name = D.name_for(dbid)
st.header(f"{name}  ·  `{dbid}`")

partners = D.drug_partner_summary(dbid, top=300)
reactions = D.drug_reactions(dbid, top=300)

ui.metric_row([
    ("Partner drugs", f"{len(partners):,}"),
    ("Distinct reactions", f"{len(reactions):,}"),
    ("Peak ROR", ui.fmt_ror(partners["max_ror"].max() if not partners.empty else 0)),
])

tab1, tab2 = st.tabs(["Top partner drugs", "Top reactions"])

with tab1:
    if partners.empty:
        st.info("No partner signals.")
    else:
        st.dataframe(
            partners[["partner_name", "partner", "n_signals", "max_ror", "total_a"]],
            width='stretch', hide_index=True,
        )
        ui.download_df(partners, f"partners_{dbid}.csv")
        top = partners.head(20).iloc[::-1]
        fig = px.bar(top, x="n_signals", y="partner_name", orientation="h",
                     labels={"n_signals": "# reaction signals", "partner_name": ""},
                     hover_data=["max_ror", "total_a"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')

with tab2:
    if reactions.empty:
        st.info("No reaction signals.")
    else:
        st.dataframe(reactions, width='stretch', hide_index=True)
        ui.download_df(reactions, f"reactions_{dbid}.csv")
        top = reactions.head(20).iloc[::-1]
        fig = px.bar(top, x="n_signals", y="reaction", orientation="h",
                     labels={"n_signals": "# pair signals", "reaction": ""},
                     hover_data=["max_ror", "total_a"])
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width='stretch')
