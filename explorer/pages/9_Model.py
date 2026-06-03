"""F11 Model performance dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

import data as D
import ui

ui.page_setup("Model")
st.title("Model performance")
st.caption("Cross-validation metrics and validation of the structure-based DDI classifier.")
ui.require_db()

metrics = D.metrics()
if metrics.empty:
    st.info("No metrics table found.")
else:
    mean = metrics[["auc", "accuracy", "precision", "recall", "f1"]].mean()
    ui.metric_row([
        ("Mean AUC", f"{mean['auc']:.4f}"),
        ("Accuracy", f"{mean['accuracy']:.3f}"),
        ("Precision", f"{mean['precision']:.3f}"),
        ("Recall", f"{mean['recall']:.3f}"),
        ("F1", f"{mean['f1']:.3f}"),
    ])
    st.subheader("Per-fold metrics")
    st.dataframe(metrics.round(4), width='stretch', hide_index=True)

    long = metrics.melt(id_vars="fold",
                        value_vars=["auc", "accuracy", "precision", "recall", "f1"],
                        var_name="metric", value_name="value")
    fig = px.line(long, x="fold", y="value", color="metric", markers=True,
                  labels={"fold": "CV fold", "value": "score"})
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

col1, col2 = st.columns(2)
with col1:
    st.subheader("ROC curve")
    if D.IMG["roc"].exists():
        st.image(str(D.IMG["roc"]), width='stretch')
with col2:
    st.subheader("Precision@k vs DrugBank")
    val = D.validation()
    if not val.empty:
        fig2 = px.line(val, x="k", y="precision", markers=True,
                       hover_data=["hits"], labels={"k": "top-k", "precision": "precision"})
        fig2.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width='stretch')
        st.dataframe(val, width='stretch', hide_index=True)
    elif D.IMG["precision_at_k"].exists():
        st.image(str(D.IMG["precision_at_k"]), width='stretch')
