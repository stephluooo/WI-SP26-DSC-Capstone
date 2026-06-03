"""Cached data-access layer for the DDI explorer.

All Streamlit pages import from here. SQLite reads are wrapped with
``st.cache_data`` / ``st.cache_resource`` so repeated queries are instant.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "ddi_study"
DB_PATH = RESULTS_DIR / "ddi.db"
FP_NPZ = RESULTS_DIR / "phase2_fingerprints.npz"
MODEL_PATH = RESULTS_DIR / "best_model.pt"

IMG = {
    "overview": RESULTS_DIR / "phase1_overview.png",
    "roc": RESULTS_DIR / "phase3_roc_curve.png",
    "cv_metrics": RESULTS_DIR / "phase4_cv_metrics.png",
    "precision_at_k": RESULTS_DIR / "phase4_precision_at_k.png",
    "top20": RESULTS_DIR / "phase4_top20_predictions.png",
    "heatmap": RESULTS_DIR / "phase4_signal_heatmap.png",
    "bootstrap": RESULTS_DIR / "phase1_bootstrap_comparison.png",
}


# --------------------------------------------------------------------------
# Connection / availability
# --------------------------------------------------------------------------
def db_exists() -> bool:
    return DB_PATH.exists()


@st.cache_resource(show_spinner=False)
def get_conn() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _table_exists(name: str) -> bool:
    con = get_conn()
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


@st.cache_data(show_spinner=False)
def has_table(name: str) -> bool:
    return _table_exists(name)


@st.cache_data(show_spinner=False)
def query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    con = get_conn()
    return pd.read_sql_query(sql, con, params=params)


@st.cache_data(show_spinner=False)
def meta() -> dict:
    if not _table_exists("meta"):
        return {}
    rows = get_conn().execute("SELECT key, value FROM meta").fetchall()
    return {r["key"]: r["value"] for r in rows}


# --------------------------------------------------------------------------
# Drug / reaction vocabularies (for autocomplete)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def drug_options() -> pd.DataFrame:
    """Distinct drugs appearing in signals: dbid + resolved name, sorted by name."""
    if _table_exists("signal_drugs"):
        df = query_df("SELECT dbid, name FROM signal_drugs ORDER BY name")
    else:
        df = query_df("SELECT dbid, name FROM drugs ORDER BY name")
    df["label"] = df["name"].fillna(df["dbid"]) + "  (" + df["dbid"] + ")"
    return df


@st.cache_data(show_spinner=False)
def name_for(dbid: str) -> str:
    row = get_conn().execute("SELECT name FROM drugs WHERE dbid=?", (dbid,)).fetchone()
    return row["name"] if row and row["name"] else dbid


@st.cache_data(show_spinner=False)
def reaction_options(limit: int = 20000) -> list[str]:
    df = query_df(
        "SELECT DISTINCT reaction FROM signals ORDER BY reaction LIMIT ?", (limit,)
    )
    return df["reaction"].tolist()


# --------------------------------------------------------------------------
# Feature queries
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def pair_signals(a: str, b: str) -> pd.DataFrame:
    """All reaction signals for a drug pair, either ordering."""
    sql = """SELECT reaction, a, b, c, d, ror, ci_low, ci_high
             FROM signals
             WHERE (drug_a=? AND drug_b=?) OR (drug_a=? AND drug_b=?)
             ORDER BY ror DESC"""
    return query_df(sql, (a, b, b, a))


@st.cache_data(show_spinner=False)
def drug_partner_summary(dbid: str, top: int = 200) -> pd.DataFrame:
    """Top partner drugs for a given drug, aggregated over reactions."""
    sql = """
        SELECT partner,
               COUNT(*)        AS n_signals,
               MAX(ror)        AS max_ror,
               SUM(a)          AS total_a
        FROM (
            SELECT drug_b AS partner, ror, a FROM signals WHERE drug_a=?
            UNION ALL
            SELECT drug_a AS partner, ror, a FROM signals WHERE drug_b=?
        )
        GROUP BY partner
        ORDER BY n_signals DESC, max_ror DESC
        LIMIT ?
    """
    df = query_df(sql, (dbid, dbid, top))
    if not df.empty:
        df["partner_name"] = df["partner"].map(_name_map())
    return df


@st.cache_data(show_spinner=False)
def drug_reactions(dbid: str, top: int = 200) -> pd.DataFrame:
    """Top reactions associated with any pair involving this drug."""
    sql = """
        SELECT reaction,
               COUNT(*)  AS n_signals,
               MAX(ror)  AS max_ror,
               SUM(a)    AS total_a
        FROM signals
        WHERE drug_a=? OR drug_b=?
        GROUP BY reaction
        ORDER BY n_signals DESC, max_ror DESC
        LIMIT ?
    """
    return query_df(sql, (dbid, dbid, top))


@st.cache_data(show_spinner=False)
def reaction_pairs(reaction: str, top: int = 500) -> pd.DataFrame:
    sql = """SELECT drug_a, drug_b, a, b, c, d, ror, ci_low, ci_high
             FROM signals WHERE reaction=?
             ORDER BY ror DESC LIMIT ?"""
    df = query_df(sql, (reaction, top))
    if not df.empty:
        nm = _name_map()
        df["drug_a_name"] = df["drug_a"].map(nm)
        df["drug_b_name"] = df["drug_b"].map(nm)
    return df


@st.cache_data(show_spinner=False)
def leaderboard(order_by: str = "ror", min_a: int = 3, min_ci_low: float = 0.0,
                reaction_like: str = "", limit: int = 1000) -> pd.DataFrame:
    allowed = {"ror": "ror DESC", "a": "a DESC", "ci_low": "ci_low DESC"}
    order_sql = allowed.get(order_by, "ror DESC")
    clauses = ["a >= ?", "ci_low >= ?"]
    params: list = [min_a, min_ci_low]
    if reaction_like.strip():
        clauses.append("reaction LIKE ?")
        params.append(f"%{reaction_like.strip().lower()}%")
    where = " AND ".join(clauses)
    sql = f"""SELECT drug_a, drug_b, reaction, a, b, c, d, ror, ci_low, ci_high
              FROM signals WHERE {where} ORDER BY {order_sql} LIMIT ?"""
    params.append(limit)
    df = query_df(sql, tuple(params))
    if not df.empty:
        nm = _name_map()
        df["drug_a_name"] = df["drug_a"].map(nm)
        df["drug_b_name"] = df["drug_b"].map(nm)
    return df


@st.cache_data(show_spinner=False)
def subnetwork(focus: str, k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ego network: focus drug + its top-k partners, with all edges among them.

    Returns (nodes_df[dbid,name,n_signals], edges_df[src,dst,weight,max_ror]).
    """
    partners = drug_partner_summary(focus, top=k)
    if partners.empty:
        return pd.DataFrame(), pd.DataFrame()
    ids = [focus] + partners["partner"].tolist()
    nm = _name_map()
    ph = ",".join("?" * len(ids))
    sql = f"""
        SELECT drug_a, drug_b, COUNT(*) AS weight, MAX(ror) AS max_ror
        FROM signals
        WHERE drug_a IN ({ph}) AND drug_b IN ({ph})
        GROUP BY drug_a, drug_b
    """
    edges = query_df(sql, tuple(ids) + tuple(ids))
    nodes = pd.DataFrame({"dbid": ids})
    nodes["name"] = nodes["dbid"].map(lambda d: nm.get(d, d))
    pn = dict(zip(partners["partner"], partners["n_signals"]))
    nodes["n_signals"] = nodes["dbid"].map(lambda d: pn.get(d, partners["n_signals"].sum()))
    return nodes, edges


@st.cache_data(show_spinner=False)
def ror_sample(max_a: int = 60, per_bucket: int = 4000) -> pd.DataFrame:
    """Sampled (a, ror, ci_low) points for the volcano/distribution plot.

    Bucketed by `a` so the scatter is not dominated by the millions of low-a points.
    """
    frames = []
    con = get_conn()
    for lo, hi in [(3, 5), (6, 10), (11, 25), (26, 60), (61, 10 ** 9)]:
        df = pd.read_sql_query(
            "SELECT a, ror, ci_low, drug_a, drug_b, reaction FROM signals "
            "WHERE a BETWEEN ? AND ? ORDER BY ror DESC LIMIT ?",
            con, params=(lo, hi, per_bucket),
        )
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    nm = _name_map()
    out["drug_a_name"] = out["drug_a"].map(nm)
    out["drug_b_name"] = out["drug_b"].map(nm)
    return out


@st.cache_data(show_spinner=False)
def top_signals(limit: int = 2000) -> pd.DataFrame:
    df = query_df(
        "SELECT drug_a, drug_b, reaction, a, ror, ci_low FROM signals "
        "ORDER BY ror DESC LIMIT ?", (limit,)
    )
    if not df.empty:
        nm = _name_map()
        df["drug_a_name"] = df["drug_a"].map(nm)
        df["drug_b_name"] = df["drug_b"].map(nm)
    return df


@st.cache_data(show_spinner=False)
def novel_predictions() -> pd.DataFrame:
    if not _table_exists("novel_predictions"):
        return pd.DataFrame()
    return query_df("SELECT * FROM novel_predictions ORDER BY predicted_probability DESC")


@st.cache_data(show_spinner=False)
def metrics() -> pd.DataFrame:
    if not _table_exists("metrics"):
        return pd.DataFrame()
    return query_df("SELECT * FROM metrics ORDER BY fold")


@st.cache_data(show_spinner=False)
def validation() -> pd.DataFrame:
    if not _table_exists("validation"):
        return pd.DataFrame()
    return query_df("SELECT * FROM validation ORDER BY k")


@st.cache_data(show_spinner=False)
def match_details(search: str = "", matched_only: bool | None = None,
                  limit: int = 1000) -> pd.DataFrame:
    if not _table_exists("match_details"):
        return pd.DataFrame()
    clauses, params = [], []
    if search.strip():
        clauses.append("(faers_name LIKE ? OR drugbank_id LIKE ?)")
        params += [f"%{search.strip().lower()}%", f"%{search.strip().upper()}%"]
    if matched_only is True:
        clauses.append("matched='True'")
    elif matched_only is False:
        clauses.append("matched='False'")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT faers_name, drugbank_id, matched FROM match_details{where} LIMIT ?"
    params.append(limit)
    return query_df(sql, tuple(params))


@st.cache_data(show_spinner=False)
def match_stats() -> dict:
    if not _table_exists("match_details"):
        return {}
    con = get_conn()
    total = con.execute("SELECT COUNT(*) FROM match_details").fetchone()[0]
    matched = con.execute(
        "SELECT COUNT(*) FROM match_details WHERE matched='True'"
    ).fetchone()[0]
    distinct_ids = con.execute(
        "SELECT COUNT(DISTINCT drugbank_id) FROM match_details WHERE matched='True'"
    ).fetchone()[0]
    return {"total": total, "matched": matched, "distinct_ids": distinct_ids,
            "rate": (matched / total) if total else 0.0}


# --------------------------------------------------------------------------
# Known DrugBank DDIs (optional)
# --------------------------------------------------------------------------
def known_ddi_available() -> bool:
    return _table_exists("known_ddi")


@st.cache_data(show_spinner=False)
def is_known_ddi(a: str, b: str) -> dict | None:
    if not _table_exists("known_ddi"):
        return None
    key = "|".join(sorted([a, b]))
    row = get_conn().execute(
        "SELECT description FROM known_ddi WHERE pair_key=? LIMIT 1", (key,)
    ).fetchone()
    return {"known": row is not None,
            "description": row["description"] if row else None}


@st.cache_data(show_spinner=False)
def known_keys_for(keys: tuple[str, ...]) -> set[str]:
    """Return subset of canonical pair_keys present in known_ddi (batched)."""
    if not _table_exists("known_ddi") or not keys:
        return set()
    con = get_conn()
    found: set[str] = set()
    CH = 800
    for i in range(0, len(keys), CH):
        chunk = keys[i:i + CH]
        ph = ",".join("?" * len(chunk))
        rows = con.execute(
            f"SELECT DISTINCT pair_key FROM known_ddi WHERE pair_key IN ({ph})", chunk
        ).fetchall()
        found.update(r[0] for r in rows)
    return found


# --------------------------------------------------------------------------
# Fingerprints + model (Tanimoto / live prediction)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fingerprints() -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (drug_ids, fps[N,1024] uint8, dbid->row index)."""
    z = np.load(FP_NPZ)
    drugs = z["drugs"].astype(str)
    fps = (z["fps"] > 0.5).astype(np.uint8)
    index = {d: i for i, d in enumerate(drugs)}
    return drugs, fps, index


def fp_available() -> bool:
    return FP_NPZ.exists()


def tanimoto(a_idx: int, b_idx: int, fps: np.ndarray) -> float:
    x, y = fps[a_idx], fps[b_idx]
    inter = int(np.bitwise_and(x, y).sum())
    union = int(np.bitwise_or(x, y).sum())
    return inter / union if union else 0.0


@st.cache_data(show_spinner=False)
def most_similar(dbid: str, top: int = 20) -> pd.DataFrame:
    drugs, fps, index = fingerprints()
    if dbid not in index:
        return pd.DataFrame()
    i = index[dbid]
    x = fps[i].astype(np.float32)
    inter = fps.astype(np.float32) @ x
    sums = fps.sum(axis=1).astype(np.float32)
    union = sums + x.sum() - inter
    sim = np.where(union > 0, inter / union, 0.0)
    order = np.argsort(-sim)
    rows = []
    nm = _name_map()
    for j in order:
        if j == i:
            continue
        rows.append({"dbid": drugs[j], "name": nm.get(drugs[j], drugs[j]),
                     "tanimoto": float(sim[j])})
        if len(rows) >= top:
            break
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load best_model.pt into the DDINet architecture from ddi_study.py."""
    import torch
    import torch.nn as nn

    class DDINet(nn.Module):
        def __init__(self, input_dim: int = 2048):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    model = DDINet()
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def model_available() -> bool:
    return MODEL_PATH.exists()


def predict_pair(a: str, b: str) -> float | None:
    """Score an arbitrary drug pair with the trained model. Returns probability or None."""
    import torch

    drugs, fps, index = fingerprints()
    if a not in index or b not in index:
        return None
    pair = sorted([a, b])
    vec = np.concatenate([fps[index[pair[0]]], fps[index[pair[1]]]]).astype(np.float32)
    model = load_model()
    with torch.no_grad():
        logit = model(torch.from_numpy(vec).unsqueeze(0))
        prob = torch.sigmoid(logit).item()
    return prob


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _name_map() -> dict:
    """dbid -> name for everything in the drugs table (fast in-memory map)."""
    con = get_conn()
    try:
        rows = con.execute("SELECT dbid, name FROM drugs").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {r["dbid"]: (r["name"] or r["dbid"]) for r in rows}
