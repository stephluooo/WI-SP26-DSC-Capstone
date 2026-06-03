"""F12 Fingerprint (Tanimoto) similarity + F13 live prediction."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Similarity")
st.title("Fingerprint similarity & live prediction")
st.caption("Compare drugs by their ECFP4 molecular fingerprints, and score any drug "
           "pair on demand with the trained model.")
ui.require_db()

if not D.fp_available():
    st.error("Fingerprint file `phase2_fingerprints.npz` not found.")
    st.stop()

drugs, fps, index = D.fingerprints()
st.caption(f"{len(drugs):,} drugs have fingerprints (1,024-bit ECFP4).")

tab1, tab2 = st.tabs(["Most similar drugs", "Live pair prediction"])

with tab1:
    dbid = ui.drug_picker("Drug", key="sim_drug")
    if dbid:
        if dbid not in index:
            st.warning(f"{D.name_for(dbid)} has no fingerprint (likely a biologic "
                       "without SMILES).")
        else:
            top = st.slider("How many", 5, 50, 20)
            sim = D.most_similar(dbid, top=top)
            st.subheader(f"Most structurally similar to {D.name_for(dbid)}")
            st.dataframe(sim, width='stretch', hide_index=True)
            plot = sim.iloc[::-1]
            fig = px.bar(plot, x="tanimoto", y="name", orientation="h",
                         labels={"tanimoto": "Tanimoto similarity", "name": ""})
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width='stretch')
            ui.download_df(sim, f"similar_{dbid}.csv")

with tab2:
    if not D.model_available():
        st.error("Trained model `best_model.pt` not found.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            a = ui.drug_picker("Drug A", key="live_a", default_index=0)
        with c2:
            b = ui.drug_picker("Drug B", key="live_b", default_index=1)
        if a and b and a != b:
            if a not in index or b not in index:
                missing = D.name_for(a) if a not in index else D.name_for(b)
                st.warning(f"{missing} has no fingerprint, so the model cannot score "
                           "this pair.")
            elif st.button("Predict interaction probability", type="primary"):
                with st.spinner("Scoring with the trained DNN..."):
                    prob = D.predict_pair(a, b)
                if prob is None:
                    st.error("Could not score this pair.")
                else:
                    st.metric(f"P(interaction) for {D.name_for(a)} + {D.name_for(b)}",
                              f"{prob:.4f}")
                    tan = D.tanimoto(index[a], index[b], fps)
                    st.caption(f"Structural Tanimoto similarity between the two drugs: "
                               f"{tan:.3f}")
                    info = D.is_known_ddi(a, b)
                    if info is not None and info["known"]:
                        st.success("This pair is already documented in DrugBank.")
                        if info["description"]:
                            st.caption(info["description"])
