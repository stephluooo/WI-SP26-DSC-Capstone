"""DDI Explorer — home / overview.

Run from the project root:

    python -m streamlit run explorer/app.py
"""

from __future__ import annotations

import streamlit as st

import data as D
import ui

ui.page_setup("Overview", icon="💊")

st.title("💊 Drug–Drug Interaction Explorer")
st.caption(
    "Interactive exploration of FAERS disproportionality signals and the "
    "structure-based DDI model from the Plan C study."
)

ui.require_db()

m = D.meta()
ui.metric_row([
    ("Signals", f"{int(m.get('n_signals', 0)):,}"),
    ("Drug pairs", f"{int(m.get('n_pairs', 0)):,}"),
    ("Reactions", f"{int(m.get('n_reactions', 0)):,}"),
    ("Drugs", f"{int(m.get('n_drugs', 0)):,}"),
])

st.divider()

left, right = st.columns([3, 2])
with left:
    st.subheader("What is this?")
    st.markdown(
        """
This dashboard sits on top of the results in `results/ddi_study/`:

- **Signal detection (Phase 1)** — for every (drug pair, reaction) we computed a
  2×2 contingency table and a **Reporting Odds Ratio (ROR)** with a 95% confidence
  interval. A high ROR means the pair and reaction are co-reported far more often
  than chance would predict.
- **Molecular fingerprints (Phase 2)** — each drug is encoded as a 1,024-bit
  **ECFP4** fingerprint from its chemical structure.
- **Deep model (Phase 3)** — a neural network learns to tell interacting from
  non-interacting pairs using only those fingerprints.
- **Novel predictions (Phase 4)** — the model scores pairs never co-reported in
  FAERS to surface candidate undocumented interactions.

Use the pages in the sidebar to explore pairs, drugs, reactions, the global signal
leaderboard, network/heatmap/volcano visualizations, the model's predictions and
performance, fingerprint similarity, live scoring, and the canonicalization audit.
        """
    )

with right:
    st.subheader("Pipeline figures")
    tabs = st.tabs(["Signals", "ROC", "Precision@k"])
    with tabs[0]:
        if D.IMG["overview"].exists():
            st.image(str(D.IMG["overview"]), width='stretch')
    with tabs[1]:
        if D.IMG["roc"].exists():
            st.image(str(D.IMG["roc"]), width='stretch')
    with tabs[2]:
        if D.IMG["precision_at_k"].exists():
            st.image(str(D.IMG["precision_at_k"]), width='stretch')

st.divider()
status = []
status.append(("Known DrugBank DDIs", "loaded ✓" if D.known_ddi_available() else "not loaded"))
status.append(("Fingerprints", "loaded ✓" if D.fp_available() else "missing"))
status.append(("Trained model", "loaded ✓" if D.model_available() else "missing"))
ui.metric_row(status)
st.caption(f"Database built: {m.get('built_at', 'unknown')}")
