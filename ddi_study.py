"""
Plan C: Molecular Fingerprint-Based Multi-Drug Interaction Prediction
=====================================================================
Detects drug-drug interaction (DDI) signals from FAERS via disproportionality
analysis, maps drugs to molecular fingerprints, and trains a deep neural
network to predict novel DDIs from chemical structure.

Architecture: DrugBank canonicalization happens BEFORE pair generation so
that brand/generic name variants (e.g. "fosamax" / "alendronate sodium")
collapse to a single DrugBank ID.

Usage:
    python ddi_study.py                       # run all phases
    python ddi_study.py --phase 1             # run only Phase 1
    python ddi_study.py --phase 4             # re-run Phase 4 from saved artifacts

Requires:
    pip install pandas numpy scipy rdkit torch scikit-learn matplotlib rapidfuzz
"""

import argparse
import csv
import io
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from textwrap import dedent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/ddi_study")
REPORTS_DIR = Path("reports")
FAERS_PARQUET = DATA_DIR / "faers_full.parquet"
FAERS_ZIP = DATA_DIR / "faers_full.csv.zip"

PHASE1_SIGNALS = RESULTS_DIR / "phase1_signals.csv"
PHASE2_STATS = RESULTS_DIR / "phase2_mapping_stats.txt"
PHASE3_METRICS = RESULTS_DIR / "phase3_metrics.csv"
PHASE3_ROC = RESULTS_DIR / "phase3_roc_curve.png"
PHASE4_NOVEL = RESULTS_DIR / "phase4_novel_predictions.csv"
PHASE4_VALIDATION = RESULTS_DIR / "phase4_validation.csv"
MODEL_PATH = RESULTS_DIR / "best_model.pt"
REPORT_MD = REPORTS_DIR / "ddi_molecular_study.md"

MIN_PAIR_EXPOSURE = 50  # Phase 4: minimum training pairs per drug


# ===================================================================
# DRUGBANK LOADING & CANONICALIZATION  (shared across phases)
# ===================================================================

def load_drugbank_vocabulary(path: str) -> pd.DataFrame:
    """Load DrugBank vocabulary CSV (plain or zipped)."""
    print(f"  Loading DrugBank vocabulary from {path} ...")
    if path.endswith(".zip"):
        vocab = pd.read_csv(path, compression="zip", dtype=str)
    else:
        vocab = pd.read_csv(path, dtype=str)
    print(f"    {len(vocab):,} entries.")
    return vocab


def parse_sdf_structures(path: str) -> tuple:
    """
    Parse DrugBank SDF and extract per-drug properties.
    Returns (id_to_smiles, id_to_name, name_to_id, synonym_to_id, product_to_id).
    """
    print(f"  Parsing DrugBank SDF from {path} ...")
    if path.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        raw = zf.read(zf.namelist()[0]).decode("utf-8", "replace")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

    entries = raw.split("$$$$")
    id_to_smiles = {}
    id_to_name = {}
    name_to_id = {}
    synonym_to_id = {}
    product_to_id = {}

    for entry in entries:
        if not entry.strip():
            continue
        props = {}
        lines = entry.split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("> <") and lines[i].endswith(">"):
                key = lines[i][3:-1]
                val = lines[i + 1].strip() if i + 1 < len(lines) else ""
                props[key] = val
                i += 2
            else:
                i += 1

        dbid = props.get("DRUGBANK_ID", "").strip()
        smiles = props.get("SMILES", "").strip()
        gname = props.get("GENERIC_NAME", "").strip()

        if not dbid:
            continue
        if smiles:
            id_to_smiles[dbid] = smiles
        if gname:
            id_to_name[dbid] = gname
            name_to_id[gname.lower()] = dbid

        for syn in props.get("SYNONYMS", "").split(";"):
            syn = syn.strip().lower()
            if syn:
                synonym_to_id[syn] = dbid
        for prod in props.get("PRODUCTS", "").split(";"):
            prod = prod.strip().lower()
            if prod:
                product_to_id[prod] = dbid

    print(f"    {len(id_to_smiles):,} SMILES, {len(name_to_id):,} names, "
          f"{len(synonym_to_id):,} synonyms, {len(product_to_id):,} products.")
    return id_to_smiles, id_to_name, name_to_id, synonym_to_id, product_to_id


def build_canonicalizer(
    vocab_path: str,
    sdf_path: str,
    fuzzy_threshold: float = 0.9,
) -> tuple:
    """
    Build a FAERS-drug-name -> DrugBank-ID lookup using tiered matching:
      1. Exact on SDF generic name
      2. Exact on SDF synonym
      3. Exact on SDF product name
      4. Exact on vocabulary CSV common name
      5. Exact on vocabulary CSV synonyms
      6. Fuzzy (Levenshtein > threshold)

    Returns (name_to_dbid, dbid_to_name, id_to_smiles, sdf_lookups).
    name_to_dbid maps lowercased FAERS strings to DrugBank IDs.
    """
    print("[Canonicalizer] Building drug name -> DrugBank ID mapping ...")
    vocab = load_drugbank_vocabulary(vocab_path)
    sdf_lookups = parse_sdf_structures(sdf_path)
    id_to_smiles, id_to_name, name_to_id, synonym_to_id, product_to_id = sdf_lookups

    vocab_name_to_id = {}
    vocab_syn_to_id = {}
    for _, row in vocab.iterrows():
        dbid = row["DrugBank ID"]
        cname = row.get("Common name", "")
        if pd.notna(cname) and cname.strip():
            vocab_name_to_id[cname.strip().lower()] = dbid
        syns = row.get("Synonyms", "")
        if pd.notna(syns):
            for s in str(syns).split("|"):
                s = s.strip().lower()
                if s:
                    vocab_syn_to_id[s] = dbid

    # Merge all exact lookups into one dict (priority order preserved by insert order)
    unified = {}
    unified.update(vocab_syn_to_id)
    unified.update(vocab_name_to_id)
    unified.update(product_to_id)
    unified.update(synonym_to_id)
    unified.update(name_to_id)  # highest priority = SDF generic name

    print(f"  Unified lookup: {len(unified):,} entries.")
    return unified, id_to_name, id_to_smiles


def canonicalize_drug(raw_name: str, lookup: dict) -> str:
    """Map a single FAERS drug name to a DrugBank ID, or return None."""
    return lookup.get(raw_name.strip().lower())


# ===================================================================
# PHASE 1 — Signal Detection via Disproportionality Analysis
# ===================================================================

def load_faers() -> pd.DataFrame:
    """Load FAERS data, preferring Parquet over CSV for speed and memory."""
    phase1_cols = [
        "safetyreportid", "drug_characterization",
        "drug_active_substance", "drug_name", "reactions",
    ]
    if FAERS_PARQUET.exists():
        print(f"[Phase 1] Loading FAERS from {FAERS_PARQUET} (Parquet) ...")
        df = pd.read_parquet(FAERS_PARQUET, columns=phase1_cols)
        df = df.astype(str)
        print(f"  Loaded {len(df):,} rows.")
        return df

    csv.field_size_limit(sys.maxsize)
    path = FAERS_ZIP if FAERS_ZIP.exists() else DATA_DIR / "faers_full.csv"
    print(f"[Phase 1] Loading FAERS from {path} (CSV) ...")
    df = pd.read_csv(path, dtype=str, low_memory=False, usecols=phase1_cols)
    print(f"  Loaded {len(df):,} rows.")
    return df


def reconstruct_reports(df: pd.DataFrame, lookup: dict) -> tuple:
    """
    Group by safetyreportid. For each report, canonicalize suspect drugs
    to DrugBank IDs. Drop drugs that can't be mapped (they can't be
    fingerprinted later anyway). Keep reports with >= 2 canonical drugs.

    Returns (reports_dict, match_details_list).
    """
    print("[Phase 1] Reconstructing reports with DrugBank canonicalization ...")
    suspects = df[df["drug_characterization"] == "1"].copy()
    suspects["raw_drug"] = (
        suspects["drug_active_substance"]
        .fillna(suspects["drug_name"])
        .str.strip()
        .str.lower()
    )
    suspects["reactions"] = suspects["reactions"].fillna("")

    match_counts = Counter()  # method tracking
    mapped_total = 0
    unmapped_total = 0
    match_details = []

    reports = {}
    for rid, grp in suspects.groupby("safetyreportid"):
        raw_drugs = set(grp["raw_drug"].dropna())
        canonical = set()
        for rd in raw_drugs:
            dbid = lookup.get(rd)
            if dbid:
                canonical.add(dbid)
                mapped_total += 1
            else:
                unmapped_total += 1

        if len(canonical) < 2:
            continue

        rxns_raw = grp["reactions"].iloc[0]
        rxns = sorted(set(r.strip().lower() for r in rxns_raw.split("|") if r.strip()))
        if rxns:
            reports[rid] = {"drugs": sorted(canonical), "reactions": rxns}

    print(f"  Drug mentions mapped: {mapped_total:,}, unmapped: {unmapped_total:,} "
          f"({100*mapped_total/(mapped_total+unmapped_total+1):.1f}% hit rate)")
    print(f"  {len(reports):,} multi-drug reports retained (2+ canonical drugs).")

    # Build match details for ALL unique raw drugs seen
    unique_raw = set(suspects["raw_drug"].dropna())
    for rd in sorted(unique_raw):
        dbid = lookup.get(rd)
        match_details.append({
            "faers_name": rd,
            "drugbank_id": dbid if dbid else "",
            "matched": bool(dbid),
        })

    return reports, match_details


def compute_pair_reaction_counts(reports: dict) -> pd.DataFrame:
    """
    For every (drug_pair, reaction), compute 2x2 contingency table and ROR.
    Filters: a >= 3 AND b >= 3 (both cells must have minimum support).
    """
    print("[Phase 1] Computing pair-reaction contingency tables ...")

    total_reports = len(reports)
    pair_rxn_counts = Counter()
    pair_counts = Counter()
    rxn_counts = Counter()

    for rec in reports.values():
        drugs = rec["drugs"]
        rxns = rec["reactions"]
        pairs = list(combinations(drugs, 2))
        for p in pairs:
            pair_counts[p] += 1
            for rxn in rxns:
                pair_rxn_counts[(p, rxn)] += 1
        for rxn in rxns:
            rxn_counts[rxn] += 1

    rows = []
    for (pair, rxn), a in pair_rxn_counts.items():
        if a < 3:
            continue
        b = pair_counts[pair] - a
        if b < 3:
            continue
        c = rxn_counts[rxn] - a
        d = total_reports - a - b - c
        if c <= 0 or d <= 0:
            continue
        ror = (a * d) / (b * c)
        ln_ror = math.log(ror)
        se = math.sqrt(1/a + 1/b + 1/c + 1/d)
        ci_low = math.exp(ln_ror - 1.96 * se)
        ci_high = math.exp(ln_ror + 1.96 * se)
        rows.append({
            "drug_a": pair[0],
            "drug_b": pair[1],
            "reaction": rxn,
            "a": a, "b": b, "c": c, "d": d,
            "ror": round(ror, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
        })

    signals = pd.DataFrame(rows)
    print(f"  {len(signals):,} (pair, reaction) combos with a>=3, b>=3.")
    return signals


def filter_signals(signals: pd.DataFrame, ci_threshold: float = 1.5) -> pd.DataFrame:
    sig = signals[(signals["ror"] > 2) & (signals["ci_low"] > ci_threshold)].copy()
    sig.sort_values("ror", ascending=False, inplace=True)
    print(f"  {len(sig):,} significant signals (ROR>2, CI_low>{ci_threshold}).")
    return sig


def label_pairs(signals: pd.DataFrame, all_pairs: set, ci_threshold: float = 1.5) -> pd.DataFrame:
    positive_pairs = set()
    for _, row in signals.iterrows():
        if row["ci_low"] > ci_threshold:
            positive_pairs.add((row["drug_a"], row["drug_b"]))

    rows = []
    for pair in all_pairs:
        rows.append({
            "drug_a": pair[0],
            "drug_b": pair[1],
            "label": 1 if pair in positive_pairs else 0,
        })
    labeled = pd.DataFrame(rows)
    n_pos = labeled["label"].sum()
    n_neg = len(labeled) - n_pos
    print(f"  Labeled pairs: {n_pos:,} positive, {n_neg:,} negative.")
    return labeled


def run_phase1(lookup: dict, dbid_to_name: dict) -> tuple:
    """Execute Phase 1 with DrugBank-canonicalized drug identifiers."""
    df = load_faers()
    reports, match_details = reconstruct_reports(df, lookup)
    del df

    # Save match details for verification
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(match_details).to_csv(RESULTS_DIR / "phase2_match_details.csv", index=False)
    print(f"  Match details saved ({len(match_details):,} unique drug names).")

    signals = compute_pair_reaction_counts(reports)
    sig_filtered = filter_signals(signals)
    sig_filtered.to_csv(PHASE1_SIGNALS, index=False)
    print(f"  Saved signals to {PHASE1_SIGNALS}")

    # Add human-readable names to signals for the saved CSV
    sig_readable = sig_filtered.copy()
    sig_readable["drug_a_name"] = sig_readable["drug_a"].map(lambda x: dbid_to_name.get(x, x))
    sig_readable["drug_b_name"] = sig_readable["drug_b"].map(lambda x: dbid_to_name.get(x, x))
    sig_readable.to_csv(RESULTS_DIR / "phase1_signals_named.csv", index=False)

    all_pairs = set()
    for rec in reports.values():
        for p in combinations(rec["drugs"], 2):
            all_pairs.add(p)

    labeled = label_pairs(sig_filtered, all_pairs)

    # Charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(np.log2(sig_filtered["ror"].clip(upper=1e4)), bins=60, edgecolor="black")
    axes[0].set_xlabel("log2(ROR)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Significant ROR Signals")

    top_rxns = sig_filtered["reaction"].value_counts().head(20)
    axes[1].barh(top_rxns.index[::-1], top_rxns.values[::-1], edgecolor="black")
    axes[1].set_xlabel("Number of DDI Signals")
    axes[1].set_title("Top 20 Reactions in DDI Signals")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "phase1_overview.png", dpi=150)
    plt.close()

    return sig_filtered, labeled


# ===================================================================
# PHASE 2 — ECFP Fingerprint Computation
# ===================================================================

def compute_ecfp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def run_phase2(labeled: pd.DataFrame, id_to_smiles: dict) -> dict:
    """Compute ECFP4 fingerprints for all DrugBank IDs in labeled pairs."""
    all_ids = sorted(set(labeled["drug_a"]) | set(labeled["drug_b"]))
    print(f"[Phase 2] Computing fingerprints for {len(all_ids):,} DrugBank IDs ...")

    fp_map = {}
    no_smiles = []
    parse_fail = []

    for dbid in all_ids:
        smiles = id_to_smiles.get(dbid)
        if not smiles:
            no_smiles.append(dbid)
            continue
        fp = compute_ecfp(smiles)
        if fp is None:
            parse_fail.append(dbid)
            continue
        fp_map[dbid] = fp

    print(f"  Fingerprints: {len(fp_map):,} OK, "
          f"{len(no_smiles)} no SMILES, {len(parse_fail)} parse failures.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_STATS, "w") as f:
        f.write(f"DrugBank IDs in labeled pairs: {len(all_ids)}\n")
        f.write(f"Fingerprints computed: {len(fp_map)}\n")
        f.write(f"No SMILES: {len(no_smiles)}\n")
        f.write(f"RDKit parse failures: {len(parse_fail)}\n")
        if no_smiles:
            f.write(f"\nNo SMILES ({len(no_smiles)}):\n")
            for dbid in no_smiles[:100]:
                f.write(f"  {dbid}\n")
        if parse_fail:
            f.write(f"\nParse failures ({len(parse_fail)}):\n")
            for dbid in parse_fail:
                f.write(f"  {dbid}\n")
    print(f"  Stats saved to {PHASE2_STATS}")

    return fp_map


# ===================================================================
# PHASE 3 — Deep Neural Network Training
# ===================================================================

def prepare_dataset(labeled: pd.DataFrame, fp_map: dict) -> tuple:
    """Build X (2048-dim) and y (0/1) arrays. Drops pairs missing fingerprints."""
    print("[Phase 3] Preparing dataset ...")
    X_rows = []
    y_rows = []
    skipped = 0

    for _, row in labeled.iterrows():
        fp_a = fp_map.get(row["drug_a"])
        fp_b = fp_map.get(row["drug_b"])
        if fp_a is None or fp_b is None:
            skipped += 1
            continue
        pair = sorted([row["drug_a"], row["drug_b"]])
        if pair[0] == row["drug_a"]:
            X_rows.append(np.concatenate([fp_a, fp_b]))
        else:
            X_rows.append(np.concatenate([fp_b, fp_a]))
        y_rows.append(row["label"])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    print(f"  Dataset: {X.shape[0]:,} pairs, {X.shape[1]} features. "
          f"Skipped {skipped:,} (missing FP).")
    return X, y


def build_model(input_dim: int = 2048):
    import torch
    import torch.nn as nn

    class DDINet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return DDINet()


def train_fold(model, X_train, y_train, X_val, y_val,
               epochs=50, batch_size=256, lr=1e-3, patience=5, device="cpu"):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    pos_weight_val = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    )

    model = build_model(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_auc = 0.0
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop epoch {epoch+1}, AUC={best_auc:.4f}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    preds = (final_probs >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_val, final_probs) if len(set(y_val)) > 1 else 0.5,
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
    }
    return best_state, metrics, final_probs


def run_phase3(labeled: pd.DataFrame, fp_map: dict,
               n_folds=5, epochs=50, batch_size=256, lr=1e-3, patience=5) -> tuple:
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve

    X, y = prepare_dataset(labeled, fp_map)
    if len(X) == 0:
        print("[Phase 3] No pairs with fingerprints. Aborting.")
        return None, None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Phase 3] Training on {device}, {n_folds}-fold CV ...")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []
    fold_roc_data = []
    global_best_auc = 0.0
    global_best_state = None

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold_i+1}/{n_folds} ...")
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = build_model(X.shape[1]).to(device)
        state, metrics, val_probs = train_fold(
            model, X_tr, y_tr, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr,
            patience=patience, device=device,
        )
        metrics["fold"] = fold_i + 1
        all_metrics.append(metrics)
        print(f"    AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}")

        fpr, tpr, _ = roc_curve(y_val, val_probs)
        fold_roc_data.append((fpr, tpr, metrics["auc"]))

        if metrics["auc"] > global_best_auc:
            global_best_auc = metrics["auc"]
            global_best_state = state

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(PHASE3_METRICS, index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    for fpr, tpr, auc_val in fold_roc_data:
        ax.plot(fpr, tpr, alpha=0.6, label=f"AUC={auc_val:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — 5-Fold Cross-Validation")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PHASE3_ROC, dpi=150)
    plt.close()

    if global_best_state:
        torch.save(global_best_state, MODEL_PATH)
        print(f"  Best model saved (AUC={global_best_auc:.4f}).")

    return global_best_state, metrics_df


# ===================================================================
# PHASE 4 — Novel DDI Prediction, Validation, Report
# ===================================================================

def score_unseen_pairs(
    fp_map: dict,
    labeled: pd.DataFrame,
    model_state: dict,
    input_dim: int = 2048,
    batch_size: int = 2048,
    top_k: int = 500,
    min_exposure: int = MIN_PAIR_EXPOSURE,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Score unseen drug pairs in streaming batches, retaining only top_k.
    Applies minimum-exposure filter: each drug must appear in at least
    `min_exposure` training pairs to be eligible for novel scoring.
    """
    import heapq
    import torch

    # Count training pair appearances per drug
    drug_pair_count = Counter()
    seen_pairs = set()
    for _, r in labeled.iterrows():
        a, b = r["drug_a"], r["drug_b"]
        key = tuple(sorted([a, b]))
        seen_pairs.add(key)
        drug_pair_count[a] += 1
        drug_pair_count[b] += 1

    # Filter to drugs with sufficient exposure AND a fingerprint
    eligible = sorted([
        d for d in fp_map
        if drug_pair_count.get(d, 0) >= min_exposure
    ])
    excluded = len(fp_map) - len(eligible)
    print(f"[Phase 4] Eligible drugs: {len(eligible)} "
          f"(excluded {excluded} with <{min_exposure} training pairs)")

    total_possible = len(eligible) * (len(eligible) - 1) // 2
    print(f"  Possible unseen pairs: ~{total_possible:,}")

    model = build_model(input_dim)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    heap = []
    batch_pairs = []
    batch_fps = []
    scored = 0

    def flush():
        nonlocal scored
        if not batch_fps:
            return
        X = np.array(batch_fps, dtype=np.float32)
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        for pair, prob in zip(batch_pairs, probs):
            p = float(prob)
            if len(heap) < top_k:
                heapq.heappush(heap, (p, pair))
            elif p > heap[0][0]:
                heapq.heapreplace(heap, (p, pair))
        scored += len(batch_fps)
        batch_pairs.clear()
        batch_fps.clear()

    for i, d_a in enumerate(eligible):
        for d_b in eligible[i+1:]:
            key = tuple(sorted([d_a, d_b]))
            if key in seen_pairs:
                continue
            batch_pairs.append(key)
            batch_fps.append(np.concatenate([fp_map[key[0]], fp_map[key[1]]]))
            if len(batch_fps) >= batch_size:
                flush()
                if scored % 1_000_000 == 0 and scored > 0:
                    print(f"    {scored:,} scored ...", flush=True)
    flush()

    print(f"  Scored {scored:,} unseen pairs. Retained top {len(heap)}.")

    rows = sorted(heap, key=lambda x: -x[0])
    result = pd.DataFrame({
        "drug_a": [r[1][0] for r in rows],
        "drug_b": [r[1][1] for r in rows],
        "predicted_probability": [r[0] for r in rows],
    })
    return result


def deduplicate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate DrugBank ID pairs (should be rare after canonicalization)."""
    seen = set()
    keep = []
    for idx, row in predictions.iterrows():
        key = tuple(sorted([row["drug_a"], row["drug_b"]]))
        if key[0] == key[1]:
            continue  # self-pair
        if key in seen:
            continue
        seen.add(key)
        keep.append(idx)
    deduped = predictions.loc[keep].reset_index(drop=True)
    dropped = len(predictions) - len(deduped)
    if dropped > 0:
        print(f"  Deduplicated: dropped {dropped} duplicate/self pairs.")
    return deduped


def validate_against_drugbank(
    predictions: pd.DataFrame,
    drugbank_interactions_path: str = "data/drugbank_all_drug_drug_interactions.csv",
) -> pd.DataFrame:
    """
    Validate predictions against known DrugBank DDIs.
    Expects CSV with DrugBank ID columns. Tries common column name patterns.
    """
    print("[Phase 4] Validating against DrugBank known DDIs ...")
    if not os.path.exists(drugbank_interactions_path):
        print(f"  WARNING: {drugbank_interactions_path} not found.")
        print(f"  Download from DrugBank -> Downloads -> Drug-Drug Interactions.")
        return pd.DataFrame({"k": [], "hits": [], "precision": []})

    known = pd.read_csv(drugbank_interactions_path, dtype=str)
    cols = known.columns.tolist()

    # Try to find the two DrugBank ID columns
    id_cols = [c for c in cols if "drugbank" in c.lower() and "id" in c.lower()]
    if len(id_cols) >= 2:
        col_a, col_b = id_cols[0], id_cols[1]
    elif len(cols) >= 2:
        col_a, col_b = cols[0], cols[1]
    else:
        print(f"  ERROR: Cannot identify ID columns in {cols}")
        return pd.DataFrame({"k": [], "hits": [], "precision": []})

    print(f"  Using columns: {col_a}, {col_b}")
    known_set = set()
    for _, row in known.iterrows():
        a = str(row[col_a]).strip()
        b = str(row[col_b]).strip()
        known_set.add(tuple(sorted([a, b])))
    print(f"  {len(known_set):,} known DDI pairs loaded.")

    ks = [10, 25, 50, 100, 200, 500]
    rows = []
    for k in ks:
        if k > len(predictions):
            break
        top_k = predictions.head(k)
        hits = sum(
            1 for _, r in top_k.iterrows()
            if tuple(sorted([r["drug_a"], r["drug_b"]])) in known_set
        )
        rows.append({"k": k, "hits": hits, "precision": round(hits / k, 4)})

    val_df = pd.DataFrame(rows)
    for _, r in val_df.iterrows():
        print(f"  P@{int(r['k']):>3d} = {r['precision']:.3f} ({int(r['hits'])} hits)")
    return val_df


def generate_visualizations(signals, predictions, validation, metrics, dbid_to_name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if predictions is not None and len(predictions) > 0:
        top20 = predictions.head(20).copy()
        top20["pair"] = (
            top20["drug_a"].map(lambda x: dbid_to_name.get(x, x)[:20]) + " + " +
            top20["drug_b"].map(lambda x: dbid_to_name.get(x, x)[:20])
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top20["pair"][::-1], top20["predicted_probability"][::-1], edgecolor="black")
        ax.set_xlabel("Predicted Interaction Probability")
        ax.set_title("Top 20 Predicted Novel DDIs")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_top20_predictions.png", dpi=150)
        plt.close()

    if validation is not None and len(validation) > 0 and validation["precision"].sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(validation["k"], validation["precision"], "o-", linewidth=2)
        ax.set_xlabel("k (top-k predictions)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision@k — Validation Against DrugBank")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_precision_at_k.png", dpi=150)
        plt.close()

    if signals is not None and len(signals) > 0:
        top_pairs = signals.groupby(["drug_a", "drug_b"])["ror"].max().nlargest(15)
        top_rxns = signals["reaction"].value_counts().head(10).index.tolist()
        hm_pairs = list(top_pairs.index)

        matrix = np.zeros((len(hm_pairs), len(top_rxns)))
        for i, (da, db) in enumerate(hm_pairs):
            for j, rxn in enumerate(top_rxns):
                match = signals[
                    (signals["drug_a"] == da) & (signals["drug_b"] == db) &
                    (signals["reaction"] == rxn)
                ]
                if len(match) > 0:
                    matrix[i, j] = np.log2(match["ror"].values[0] + 1)

        pair_labels = [
            f"{dbid_to_name.get(a, a)[:15]} + {dbid_to_name.get(b, b)[:15]}"
            for a, b in hm_pairs
        ]
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(top_rxns)))
        ax.set_xticklabels(top_rxns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(hm_pairs)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_title("Top DDI Signals: log2(ROR+1) Heatmap")
        plt.colorbar(im, ax=ax, label="log2(ROR+1)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_signal_heatmap.png", dpi=150)
        plt.close()

    if metrics is not None and len(metrics) > 0:
        metric_cols = ["auc", "accuracy", "precision", "recall", "f1"]
        means = [metrics[c].mean() for c in metric_cols]
        stds = [metrics[c].std() for c in metric_cols]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(metric_cols)), means, yerr=stds, capsize=4, edgecolor="black")
        ax.set_xticks(range(len(metric_cols)))
        ax.set_xticklabels([c.upper() for c in metric_cols])
        ax.set_ylabel("Score")
        ax.set_title("5-Fold Cross-Validation Metrics (Mean +/- Std)")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_cv_metrics.png", dpi=150)
        plt.close()

    print("[Phase 4] Charts saved.")


def generate_report(signals, labeled, metrics, predictions, validation, dbid_to_name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    n_signals = len(signals) if signals is not None else 0
    n_pairs = len(labeled) if labeled is not None else 0
    n_pos = int(labeled["label"].sum()) if labeled is not None else 0
    n_neg = n_pairs - n_pos
    mean_auc = metrics["auc"].mean() if metrics is not None and len(metrics) else 0
    std_auc = metrics["auc"].std() if metrics is not None and len(metrics) else 0

    def name(dbid):
        return dbid_to_name.get(dbid, dbid)

    top_signals_md = ""
    if signals is not None and len(signals) > 0:
        top10 = signals.head(10)
        top_signals_md = "| Drug A | Drug B | Reaction | ROR | CI Low | Cases |\n"
        top_signals_md += "|---|---|---|---|---|---|\n"
        for _, r in top10.iterrows():
            top_signals_md += (f"| {name(r['drug_a'])} | {name(r['drug_b'])} | "
                              f"{r['reaction']} | {r['ror']:.1f} | {r['ci_low']:.1f} | {r['a']} |\n")

    top_novel_md = ""
    if predictions is not None and len(predictions) > 0:
        top10p = predictions.head(10)
        top_novel_md = "| Drug A | Drug B | Probability |\n|---|---|---|\n"
        for _, r in top10p.iterrows():
            top_novel_md += (f"| {name(r['drug_a'])} | {name(r['drug_b'])} | "
                            f"{r['predicted_probability']:.4f} |\n")

    metrics_md = ""
    if metrics is not None and len(metrics) > 0:
        metrics_md = "| Fold | AUC | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n"
        for _, r in metrics.iterrows():
            metrics_md += (f"| {int(r['fold'])} | {r['auc']:.4f} | {r['accuracy']:.4f} | "
                          f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n")
        means = metrics[["auc", "accuracy", "precision", "recall", "f1"]].mean()
        metrics_md += (f"| **Mean** | **{means['auc']:.4f}** | **{means['accuracy']:.4f}** | "
                      f"**{means['precision']:.4f}** | **{means['recall']:.4f}** | **{means['f1']:.4f}** |\n")

    precision_md = ""
    if validation is not None and len(validation) > 0:
        precision_md = "| k | Hits | Precision |\n|---|---|---|\n"
        for _, r in validation.iterrows():
            precision_md += f"| {int(r['k'])} | {int(r['hits'])} | {r['precision']:.3f} |\n"

    md = dedent(f"""\
    # Molecular Fingerprint DDI Prediction — Study Report

    ## Overview

    This study applies a molecular fingerprint-based deep learning pipeline to
    the FDA Adverse Event Reporting System (FAERS) to detect and predict
    multi-drug interactions (DDIs). Drug names are canonicalized to DrugBank IDs
    before pair generation, eliminating false signals from brand/generic name variants.

    **Dataset**: FAERS (full database, all available years).

    ---

    ## Phase 1: Signal Detection

    Pairwise disproportionality analysis (ROR) on DrugBank-canonicalized drug
    pairs identified **{n_signals:,}** statistically significant signals.
    Contingency table filters: a >= 3, b >= 3, ROR > 2, 95% CI lower > 1.5.

    **Labeled pairs**: {n_pos:,} positive / {n_neg:,} negative.

    ### Top 10 Strongest Signals

    {top_signals_md}

    ![Phase 1 Overview](../results/ddi_study/phase1_overview.png)

    ---

    ## Phase 2: ECFP4 Fingerprints

    DrugBank IDs mapped to 1024-bit ECFP4 molecular fingerprints via RDKit.
    See `results/ddi_study/phase2_mapping_stats.txt` for statistics.

    ---

    ## Phase 3: Deep Neural Network

    4-layer DNN (2048->512->256->128->1) with 5-fold stratified CV.

    **Mean AUC**: {mean_auc:.4f} +/- {std_auc:.4f}

    ### Per-Fold Metrics

    {metrics_md}

    ![ROC](../results/ddi_study/phase3_roc_curve.png)
    ![Metrics](../results/ddi_study/phase4_cv_metrics.png)

    ---

    ## Phase 4: Novel DDI Predictions

    Scored unseen drug pairs (minimum {MIN_PAIR_EXPOSURE} training-pair
    exposure per drug). Deduplicated by DrugBank ID.

    ### Top 10 Predicted Novel DDIs

    {top_novel_md}

    ![Top Predictions](../results/ddi_study/phase4_top20_predictions.png)

    ### Signal Heatmap

    ![Heatmap](../results/ddi_study/phase4_signal_heatmap.png)

    ### Validation Against DrugBank

    {precision_md}

    ---

    ## Limitations

    - FAERS is spontaneous reporting; does not establish causation.
    - Drugs not in DrugBank are excluded from analysis.
    - Model learns substructure-reporting correlations, not pharmacokinetic mechanisms.
    - Minimum-exposure filter reduces false positives but may exclude rare-but-real DDIs.

    ## References

    1. Schreier, T. et al. (2024). Integration of FAERS, DrugBank and SIDER
       for ML-based ADR Detection. *Datenbank-Spektrum*, 24, 233-242.
    2. Zhang, X. et al. (2025). Identifying Drug Combinations Associated with
       Acute Kidney Injury. *Biomed J Sci & Tech Res*, 64(1).
    3. Shen, Y. et al. (2020). Mining High-Order Drug Interaction Effects.
       *BMC Med Inform Decis Mak*, 20, 48.
    """)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Phase 4] Report saved to {REPORT_MD}")


def run_phase4(signals, labeled, fp_map, model_state, metrics, dbid_to_name):
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictions = score_unseen_pairs(fp_map, labeled, model_state, device=device)
    predictions = deduplicate_predictions(predictions)

    # Add human-readable names
    predictions["drug_a_name"] = predictions["drug_a"].map(lambda x: dbid_to_name.get(x, x))
    predictions["drug_b_name"] = predictions["drug_b"].map(lambda x: dbid_to_name.get(x, x))
    predictions.to_csv(PHASE4_NOVEL, index=False)
    print(f"  Predictions saved to {PHASE4_NOVEL}")

    validation = validate_against_drugbank(predictions)
    if len(validation) > 0:
        validation.to_csv(PHASE4_VALIDATION, index=False)

    generate_visualizations(signals, predictions, validation, metrics, dbid_to_name)
    generate_report(signals, labeled, metrics, predictions, validation, dbid_to_name)


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Plan C: Molecular DDI Prediction")
    parser.add_argument("--phase", type=int, default=0,
                        help="Run a specific phase (1-4). 0 = all.")
    parser.add_argument("--drugbank-vocab",
                        default="data/drugbank_all_drugbank_vocabulary.csv.zip")
    parser.add_argument("--drugbank-sdf",
                        default="data/drugbank_all_structures.sdf.zip")
    parser.add_argument("--min-exposure", type=int, default=MIN_PAIR_EXPOSURE,
                        help=f"Min training pairs per drug for Phase 4 (default: {MIN_PAIR_EXPOSURE})")
    args = parser.parse_args()

    global MIN_PAIR_EXPOSURE
    MIN_PAIR_EXPOSURE = args.min_exposure

    run_all = args.phase == 0
    signals, labeled, fp_map, model_state, metrics = None, None, None, None, None

    # Always need the canonicalizer (DrugBank lookups)
    print("=" * 60)
    lookup, dbid_to_name, id_to_smiles = build_canonicalizer(
        args.drugbank_vocab, args.drugbank_sdf
    )
    print("=" * 60)

    # Save ID-to-name mapping for later phases
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name_map_path = RESULTS_DIR / "dbid_to_name.json"
    if run_all or args.phase == 1:
        import json as jsonmod
        with open(name_map_path, "w") as f:
            jsonmod.dump(dbid_to_name, f)

    if run_all or args.phase == 1:
        signals, labeled = run_phase1(lookup, dbid_to_name)
        labeled.to_csv(RESULTS_DIR / "phase1_labeled_pairs.csv", index=False)

    if run_all or args.phase >= 2:
        if labeled is None:
            labeled = pd.read_csv(RESULTS_DIR / "phase1_labeled_pairs.csv", dtype=str)
            labeled["label"] = labeled["label"].astype(int)
        if signals is None and PHASE1_SIGNALS.exists():
            signals = pd.read_csv(PHASE1_SIGNALS)
        if name_map_path.exists() and not dbid_to_name:
            import json as jsonmod
            with open(name_map_path) as f:
                dbid_to_name = jsonmod.load(f)

    if run_all or args.phase == 2:
        fp_map = run_phase2(labeled, id_to_smiles)
        np.savez_compressed(
            RESULTS_DIR / "phase2_fingerprints.npz",
            drugs=np.array(list(fp_map.keys())),
            fps=np.array(list(fp_map.values())),
        )

    if run_all or args.phase >= 3:
        if fp_map is None:
            data = np.load(RESULTS_DIR / "phase2_fingerprints.npz", allow_pickle=True)
            fp_map = dict(zip(data["drugs"], data["fps"]))

    if run_all or args.phase == 3:
        model_state, metrics = run_phase3(labeled, fp_map)

    if run_all or args.phase == 4:
        import torch
        if model_state is None and MODEL_PATH.exists():
            model_state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if metrics is None and PHASE3_METRICS.exists():
            metrics = pd.read_csv(PHASE3_METRICS)
        if model_state is None:
            print("[Phase 4] No trained model found. Run Phase 3 first.")
            return
        run_phase4(signals, labeled, fp_map, model_state, metrics, dbid_to_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
