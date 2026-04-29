"""
Plan C: Molecular Fingerprint-Based Multi-Drug Interaction Prediction
=====================================================================
Detects drug-drug interaction (DDI) signals from FAERS via disproportionality
analysis, maps drugs to molecular fingerprints, and trains a deep neural
network to predict novel DDIs from chemical structure.

Usage:
    python ddi_study.py                       # run all phases
    python ddi_study.py --phase 1             # run only Phase 1
    python ddi_study.py --phase 2             # run Phase 2 with default DrugBank paths

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
FAERS_ZIP = DATA_DIR / "faers_full.csv.zip"

PHASE1_SIGNALS = RESULTS_DIR / "phase1_signals.csv"
PHASE2_STATS = RESULTS_DIR / "phase2_mapping_stats.txt"
PHASE3_METRICS = RESULTS_DIR / "phase3_metrics.csv"
PHASE3_ROC = RESULTS_DIR / "phase3_roc_curve.png"
PHASE4_NOVEL = RESULTS_DIR / "phase4_novel_predictions.csv"
PHASE4_VALIDATION = RESULTS_DIR / "phase4_validation.csv"
MODEL_PATH = RESULTS_DIR / "best_model.pt"
REPORT_MD = REPORTS_DIR / "ddi_molecular_study.md"


# ===================================================================
# PHASE 1 — Signal Detection via Disproportionality Analysis
# ===================================================================

def load_faers(path: Path = FAERS_ZIP) -> pd.DataFrame:
    """Load FAERS CSV from zip, applying minimal type coercion."""
    csv.field_size_limit(sys.maxsize)
    print(f"[Phase 1] Loading FAERS data from {path} ...")
    df = pd.read_csv(path, dtype=str, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")
    return df


def reconstruct_reports(df: pd.DataFrame) -> dict:
    """
    Group by safetyreportid and return dict:
        report_id -> {"drugs": [str], "reactions": [str]}
    Only retains suspect drugs (drug_characterization == '1').
    """
    print("[Phase 1] Reconstructing reports (suspect drugs only) ...")
    suspects = df[df["drug_characterization"] == "1"].copy()
    suspects["drug_active_substance"] = (
        suspects["drug_active_substance"]
        .fillna(suspects["drug_name"])
        .str.strip()
        .str.lower()
    )
    suspects["reactions"] = suspects["reactions"].fillna("")

    reports = {}
    for rid, grp in suspects.groupby("safetyreportid"):
        drugs = sorted(set(grp["drug_active_substance"].dropna()))
        rxns_raw = grp["reactions"].iloc[0]
        rxns = sorted(set(r.strip().lower() for r in rxns_raw.split("|") if r.strip()))
        if len(drugs) >= 2 and rxns:
            reports[rid] = {"drugs": drugs, "reactions": rxns}

    print(f"  {len(reports):,} multi-drug reports with reactions retained.")
    return reports


def compute_pair_reaction_counts(reports: dict) -> pd.DataFrame:
    """
    For every (drug_pair, reaction) in the reports, compute:
        a = pair+reaction, b = pair+no_reaction, c = no_pair+reaction, d = no_pair+no_reaction
    Returns DataFrame with ROR and 95% CI.
    """
    print("[Phase 1] Computing pair-reaction contingency tables ...")

    total_reports = len(reports)
    pair_rxn_counts = Counter()     # (pair, rxn) -> count of reports
    pair_counts = Counter()         # pair -> count of reports
    rxn_counts = Counter()          # rxn -> count of reports

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
        c = rxn_counts[rxn] - a
        d = total_reports - a - b - c
        if b <= 0 or c <= 0 or d <= 0:
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
    print(f"  {len(signals):,} (pair, reaction) combinations with a >= 3.")
    return signals


def filter_signals(signals: pd.DataFrame, ci_threshold: float = 1.5) -> pd.DataFrame:
    """Keep signals where ROR > 2 and lower CI > ci_threshold."""
    sig = signals[(signals["ror"] > 2) & (signals["ci_low"] > ci_threshold)].copy()
    sig.sort_values("ror", ascending=False, inplace=True)
    print(f"  {len(sig):,} significant signals (ROR>2, CI_low>{ci_threshold}).")
    return sig


def label_pairs(signals: pd.DataFrame, all_pairs: set, ci_threshold: float = 1.5) -> pd.DataFrame:
    """
    Label drug pairs: positive if at least one significant signal,
    negative otherwise.
    """
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


def run_phase1() -> tuple:
    """Execute Phase 1 and return (signals_df, labeled_pairs_df)."""
    df = load_faers()
    reports = reconstruct_reports(df)

    signals = compute_pair_reaction_counts(reports)
    sig_filtered = filter_signals(signals)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sig_filtered.to_csv(PHASE1_SIGNALS, index=False)
    print(f"  Saved signals to {PHASE1_SIGNALS}")

    all_pairs = set()
    for rec in reports.values():
        for p in combinations(rec["drugs"], 2):
            all_pairs.add(p)

    labeled = label_pairs(sig_filtered, all_pairs)

    # --- Charts ---
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
    chart_path = RESULTS_DIR / "phase1_overview.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved chart to {chart_path}")

    return sig_filtered, labeled


# ===================================================================
# PHASE 2 — Chemical Structure Mapping
# ===================================================================

def load_drugbank_vocabulary(path: str) -> pd.DataFrame:
    """
    Load DrugBank vocabulary CSV (plain or zipped).
    Expected columns: DrugBank ID, Common name, Synonyms, ...
    """
    print(f"[Phase 2] Loading DrugBank vocabulary from {path} ...")
    if path.endswith(".zip"):
        vocab = pd.read_csv(path, compression="zip", dtype=str)
    else:
        vocab = pd.read_csv(path, dtype=str)
    print(f"  {len(vocab):,} DrugBank entries loaded.")
    return vocab


def parse_sdf_structures(path: str) -> dict:
    """
    Parse DrugBank SDF file (plain or zipped) and extract DRUGBANK_ID -> SMILES.
    Also builds secondary lookup dicts for GENERIC_NAME, SYNONYMS, and PRODUCTS.
    Returns (id_to_smiles, id_to_name, name_to_id, synonym_to_id, product_to_id).
    """
    print(f"[Phase 2] Parsing DrugBank SDF from {path} ...")
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

    print(f"  {len(id_to_smiles):,} drugs with SMILES, {len(name_to_id):,} generic names, "
          f"{len(synonym_to_id):,} synonyms, {len(product_to_id):,} product names.")
    return id_to_smiles, id_to_name, name_to_id, synonym_to_id, product_to_id


def match_faers_to_drugbank(
    faers_drugs: list,
    vocab: pd.DataFrame,
    sdf_lookups: tuple,
    fuzzy_threshold: float = 0.9,
) -> tuple:
    """
    Map FAERS drug names to DrugBank IDs using a tiered strategy:
      1. Exact match on generic name (from SDF)
      2. Exact match on synonym (from SDF)
      3. Exact match on product name (from SDF)
      4. Exact match on vocabulary CSV common name
      5. Exact match on vocabulary CSV synonyms
      6. Fuzzy match (Levenshtein ratio > threshold) as fallback

    Returns (matched_dict, match_details_list).
    match_details_list has one entry per matched drug with the method used,
    for verification purposes.
    """
    from rapidfuzz import fuzz, process as rfprocess

    print("[Phase 2] Matching FAERS drug names to DrugBank ...")

    _, id_to_name, name_to_id, synonym_to_id, product_to_id = sdf_lookups

    # Build vocabulary CSV lookups
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

    matched = {}
    details = []
    unmatched = []

    for drug in faers_drugs:
        d_lower = drug.strip().lower()
        if d_lower in name_to_id:
            dbid = name_to_id[d_lower]
            matched[drug] = dbid
            details.append({"faers_name": drug, "drugbank_id": dbid,
                            "drugbank_name": id_to_name.get(dbid, ""),
                            "method": "sdf_generic_name", "score": 100})
        elif d_lower in synonym_to_id:
            dbid = synonym_to_id[d_lower]
            matched[drug] = dbid
            details.append({"faers_name": drug, "drugbank_id": dbid,
                            "drugbank_name": id_to_name.get(dbid, ""),
                            "method": "sdf_synonym", "score": 100})
        elif d_lower in product_to_id:
            dbid = product_to_id[d_lower]
            matched[drug] = dbid
            details.append({"faers_name": drug, "drugbank_id": dbid,
                            "drugbank_name": id_to_name.get(dbid, ""),
                            "method": "sdf_product", "score": 100})
        elif d_lower in vocab_name_to_id:
            dbid = vocab_name_to_id[d_lower]
            matched[drug] = dbid
            details.append({"faers_name": drug, "drugbank_id": dbid,
                            "drugbank_name": id_to_name.get(dbid, ""),
                            "method": "vocab_common_name", "score": 100})
        elif d_lower in vocab_syn_to_id:
            dbid = vocab_syn_to_id[d_lower]
            matched[drug] = dbid
            details.append({"faers_name": drug, "drugbank_id": dbid,
                            "drugbank_name": id_to_name.get(dbid, ""),
                            "method": "vocab_synonym", "score": 100})
        else:
            unmatched.append(drug)

    if unmatched:
        all_names = {}
        all_names.update({n: name_to_id[n] for n in name_to_id})
        all_names.update({n: synonym_to_id[n] for n in synonym_to_id})
        all_names.update({n: vocab_name_to_id[n] for n in vocab_name_to_id})
        all_names.update({n: vocab_syn_to_id[n] for n in vocab_syn_to_id})
        all_keys = list(all_names.keys())
        print(f"  Attempting fuzzy match for {len(unmatched):,} unmatched drugs ...")
        for drug in unmatched:
            result = rfprocess.extractOne(
                drug.lower(), all_keys, scorer=fuzz.ratio,
                score_cutoff=fuzzy_threshold * 100,
            )
            if result:
                best_name, score, _ = result
                dbid = all_names[best_name]
                matched[drug] = dbid
                details.append({"faers_name": drug, "drugbank_id": dbid,
                                "drugbank_name": id_to_name.get(dbid, ""),
                                "method": "fuzzy", "score": round(score, 1),
                                "fuzzy_matched_to": best_name})

    n_total = len(faers_drugs)
    n_matched = len(matched)
    print(f"  Matched {n_matched}/{n_total} ({100*n_matched/n_total:.1f}%) FAERS drugs to DrugBank.")

    # Save full match details for verification
    details_df = pd.DataFrame(details)
    details_path = RESULTS_DIR / "phase2_match_details.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    details_df.to_csv(details_path, index=False)
    print(f"  Match details saved to {details_path}")

    return matched, details_df


def compute_ecfp(smiles: str, radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    """Compute ECFP fingerprint from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def build_fingerprint_map(
    matched: dict,
    id_to_smiles: dict,
    radius: int = 2,
    n_bits: int = 1024,
) -> tuple:
    """
    Build {faers_drug_name: np.array(1024)} from matched DrugBank IDs and SMILES.
    """
    print("[Phase 2] Computing ECFP4 fingerprints ...")
    fp_map = {}
    failed = []
    for drug, dbid in matched.items():
        smiles = id_to_smiles.get(dbid)
        if not smiles:
            failed.append((drug, dbid, "no SMILES in SDF"))
            continue
        fp = compute_ecfp(smiles, radius=radius, n_bits=n_bits)
        if fp is None:
            failed.append((drug, dbid, "RDKit parse failure"))
            continue
        fp_map[drug] = fp

    print(f"  Fingerprints computed for {len(fp_map):,} drugs; {len(failed)} failed.")
    return fp_map, failed


def run_phase2(
    labeled: pd.DataFrame,
    drugbank_vocab_path: str = "data/drugbank_all_drugbank_vocabulary.csv.zip",
    drugbank_sdf_path: str = "data/drugbank_all_structures.sdf.zip",
) -> tuple:
    """Execute Phase 2 and return (fingerprint_map, match_details_df)."""
    all_drugs = sorted(set(labeled["drug_a"]) | set(labeled["drug_b"]))
    print(f"[Phase 2] {len(all_drugs):,} unique drugs in labeled pairs.")

    vocab = load_drugbank_vocabulary(drugbank_vocab_path)
    sdf_lookups = parse_sdf_structures(drugbank_sdf_path)
    id_to_smiles = sdf_lookups[0]

    matched, details_df = match_faers_to_drugbank(all_drugs, vocab, sdf_lookups)
    fp_map, failed = build_fingerprint_map(matched, id_to_smiles)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PHASE2_STATS, "w") as f:
        f.write(f"Total unique FAERS drugs: {len(all_drugs)}\n")
        f.write(f"DrugBank matched: {len(matched)}\n")
        f.write(f"Fingerprints computed: {len(fp_map)}\n")
        f.write(f"Failed fingerprints: {len(failed)}\n\n")
        method_counts = details_df["method"].value_counts()
        f.write("Match method breakdown:\n")
        for method, count in method_counts.items():
            f.write(f"  {method}: {count}\n")
        f.write(f"\nFailed drugs ({len(failed)}):\n")
        for drug, dbid, reason in failed:
            f.write(f"  {drug} ({dbid}): {reason}\n")
    print(f"  Stats saved to {PHASE2_STATS}")

    return fp_map, details_df


# ===================================================================
# PHASE 3 — Deep Neural Network Training
# ===================================================================

def prepare_dataset(
    labeled: pd.DataFrame,
    fp_map: dict,
    n_bits: int = 1024,
) -> tuple:
    """
    Build X (2048-dim concatenated fingerprints) and y (0/1 labels) arrays.
    Drops pairs where either drug lacks a fingerprint.
    """
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
        X_rows.append(np.concatenate([fp_a, fp_b]))
        y_rows.append(row["label"])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    print(f"  Dataset: {X.shape[0]:,} pairs, {X.shape[1]} features. Skipped {skipped:,} (missing FP).")
    return X, y


def build_model(input_dim: int = 2048):
    """Define the DDI prediction DNN in PyTorch."""
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
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    return DDINet()


def train_fold(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
):
    """
    Train for one CV fold.  Returns (best_model_state, metrics_dict, train_losses, val_aucs).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

    pos_weight_val = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    model_raw = build_model(X_train.shape[1]).to(device)
    model_raw.net[-1] = nn.Identity()

    optimizer = torch.optim.Adam(model_raw.parameters(), lr=lr, weight_decay=1e-5)

    ds_train = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    best_auc = 0.0
    best_state = None
    wait = 0
    train_losses = []
    val_aucs = []

    for epoch in range(epochs):
        model_raw.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model_raw(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(ds_train)
        train_losses.append(epoch_loss)

        model_raw.eval()
        with torch.no_grad():
            val_logits = model_raw(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            auc = 0.5
        val_aucs.append(auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model_raw.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}, best AUC={best_auc:.4f}")
                break

    model_raw.load_state_dict(best_state)
    model_raw.eval()
    with torch.no_grad():
        final_logits = model_raw(X_val_t)
        final_probs = torch.sigmoid(final_logits).cpu().numpy()
    preds = (final_probs >= 0.5).astype(int)

    metrics = {
        "auc": roc_auc_score(y_val, final_probs) if len(set(y_val)) > 1 else 0.5,
        "accuracy": accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, zero_division=0),
        "recall": recall_score(y_val, preds, zero_division=0),
        "f1": f1_score(y_val, preds, zero_division=0),
    }
    return best_state, metrics, train_losses, val_aucs, final_probs


def run_phase3(
    labeled: pd.DataFrame,
    fp_map: dict,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple:
    """Run 5-fold stratified cross-validation. Returns (best_state, all_metrics)."""
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
        state, metrics, _, _, val_probs = train_fold(
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
    print(f"  Metrics saved to {PHASE3_METRICS}")

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
    print(f"  ROC chart saved to {PHASE3_ROC}")

    if global_best_state:
        torch.save(global_best_state, MODEL_PATH)
        print(f"  Best model saved to {MODEL_PATH}")

    return global_best_state, metrics_df


# ===================================================================
# PHASE 4 — Novel DDI Prediction, Validation, and Report Generation
# ===================================================================

def score_unseen_pairs(
    fp_map: dict,
    seen_pairs: set,
    model_state: dict,
    input_dim: int = 2048,
    batch_size: int = 1024,
    device: str = "cpu",
) -> pd.DataFrame:
    """Score all possible drug pairs not seen in training."""
    import torch

    drugs = sorted(fp_map.keys())
    print(f"[Phase 4] Generating unseen pairs from {len(drugs)} drugs ...")

    model = build_model(input_dim)
    model.net[-1] = torch.nn.Identity()
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    pairs = []
    X_batch = []

    for i, d_a in enumerate(drugs):
        for d_b in drugs[i+1:]:
            key = (d_a, d_b) if d_a < d_b else (d_b, d_a)
            if key in seen_pairs:
                continue
            fp = np.concatenate([fp_map[d_a], fp_map[d_b]])
            pairs.append(key)
            X_batch.append(fp)

    if not X_batch:
        print("  No unseen pairs to score.")
        return pd.DataFrame()

    X_all = np.array(X_batch, dtype=np.float32)
    probs = []
    with torch.no_grad():
        for start in range(0, len(X_all), batch_size):
            end = min(start + batch_size, len(X_all))
            xb = torch.tensor(X_all[start:end], dtype=torch.float32).to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p.tolist())

    result = pd.DataFrame({
        "drug_a": [p[0] for p in pairs],
        "drug_b": [p[1] for p in pairs],
        "predicted_probability": probs,
    })
    result.sort_values("predicted_probability", ascending=False, inplace=True)
    print(f"  Scored {len(result):,} unseen pairs.")
    return result


def validate_against_drugbank(
    predictions: pd.DataFrame,
    drugbank_interactions_path: str = "data/drugbank_interactions.csv",
) -> pd.DataFrame:
    """
    Check predicted DDIs against known DrugBank interactions.
    Expects CSV with columns: drug_a, drug_b (DrugBank names or IDs).
    Returns precision@k table.
    """
    print("[Phase 4] Validating against DrugBank known DDIs ...")
    if not os.path.exists(drugbank_interactions_path):
        print(f"  WARNING: {drugbank_interactions_path} not found. Skipping validation.")
        return pd.DataFrame({"k": [], "precision": []})

    known = pd.read_csv(drugbank_interactions_path, dtype=str)
    known_set = set()
    for _, row in known.iterrows():
        a = str(row.iloc[0]).strip().lower()
        b = str(row.iloc[1]).strip().lower()
        key = tuple(sorted([a, b]))
        known_set.add(key)

    ks = [10, 25, 50, 100]
    rows = []
    for k in ks:
        top_k = predictions.head(k)
        hits = 0
        for _, r in top_k.iterrows():
            key = tuple(sorted([r["drug_a"], r["drug_b"]]))
            if key in known_set:
                hits += 1
        rows.append({"k": k, "hits": hits, "precision": hits / k if k else 0})

    val_df = pd.DataFrame(rows)
    print(f"  Precision@k: {dict(zip(val_df['k'], val_df['precision']))}")
    return val_df


def generate_visualizations(
    signals: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
):
    """Generate all Phase 4 charts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Top 20 predicted novel DDI pairs
    if len(predictions) > 0:
        top20 = predictions.head(20).copy()
        top20["pair"] = top20["drug_a"] + " + " + top20["drug_b"]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(top20["pair"][::-1], top20["predicted_probability"][::-1], edgecolor="black")
        ax.set_xlabel("Predicted Interaction Probability")
        ax.set_title("Top 20 Predicted Novel DDIs")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_top20_predictions.png", dpi=150)
        plt.close()

    # 2. Precision@k curve
    if len(validation) > 0 and validation["precision"].sum() > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(validation["k"], validation["precision"], "o-", linewidth=2)
        ax.set_xlabel("k (top-k predictions)")
        ax.set_ylabel("Precision")
        ax.set_title("Precision@k — Validation Against DrugBank")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_precision_at_k.png", dpi=150)
        plt.close()

    # 3. Heatmap: Top DDI signals (drug x reaction)
    if len(signals) > 0:
        top_pairs = signals.groupby(["drug_a", "drug_b"])["ror"].max().nlargest(15)
        top_rxns = signals["reaction"].value_counts().head(10).index.tolist()
        hm_pairs = [(a, b) for (a, b) in top_pairs.index]

        matrix = np.zeros((len(hm_pairs), len(top_rxns)))
        for i, (da, db) in enumerate(hm_pairs):
            for j, rxn in enumerate(top_rxns):
                match = signals[
                    (signals["drug_a"] == da) &
                    (signals["drug_b"] == db) &
                    (signals["reaction"] == rxn)
                ]
                if len(match) > 0:
                    matrix[i, j] = np.log2(match["ror"].values[0] + 1)

        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(top_rxns)))
        ax.set_xticklabels(top_rxns, rotation=45, ha="right", fontsize=8)
        pair_labels = [f"{a[:15]} + {b[:15]}" for a, b in hm_pairs]
        ax.set_yticks(range(len(hm_pairs)))
        ax.set_yticklabels(pair_labels, fontsize=8)
        ax.set_title("Top DDI Signals: log2(ROR+1) Heatmap")
        plt.colorbar(im, ax=ax, label="log2(ROR+1)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_signal_heatmap.png", dpi=150)
        plt.close()

    # 4. Cross-validation summary bar chart
    if metrics is not None and len(metrics) > 0:
        metric_cols = ["auc", "accuracy", "precision", "recall", "f1"]
        means = [metrics[c].mean() for c in metric_cols]
        stds = [metrics[c].std() for c in metric_cols]
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(metric_cols))
        ax.bar(x, means, yerr=stds, capsize=4, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in metric_cols])
        ax.set_ylabel("Score")
        ax.set_title("5-Fold Cross-Validation Metrics (Mean +/- Std)")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "phase4_cv_metrics.png", dpi=150)
        plt.close()

    print("[Phase 4] All charts saved to results/ddi_study/.")


def generate_report(
    signals: pd.DataFrame,
    labeled: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    validation: pd.DataFrame,
):
    """Write the final markdown report with inline images."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    n_signals = len(signals) if signals is not None else 0
    n_pairs = len(labeled) if labeled is not None else 0
    n_pos = int(labeled["label"].sum()) if labeled is not None else 0
    n_neg = n_pairs - n_pos

    mean_auc = metrics["auc"].mean() if metrics is not None and len(metrics) else 0
    std_auc = metrics["auc"].std() if metrics is not None and len(metrics) else 0

    top_signals_md = ""
    if signals is not None and len(signals) > 0:
        top10 = signals.head(10)
        top_signals_md = "| Drug A | Drug B | Reaction | ROR | 95% CI Lower | Cases |\n"
        top_signals_md += "|---|---|---|---|---|---|\n"
        for _, r in top10.iterrows():
            top_signals_md += f"| {r['drug_a']} | {r['drug_b']} | {r['reaction']} | {r['ror']:.1f} | {r['ci_low']:.1f} | {r['a']} |\n"

    top_novel_md = ""
    if predictions is not None and len(predictions) > 0:
        top10p = predictions.head(10)
        top_novel_md = "| Drug A | Drug B | Predicted Probability |\n"
        top_novel_md += "|---|---|---|\n"
        for _, r in top10p.iterrows():
            top_novel_md += f"| {r['drug_a']} | {r['drug_b']} | {r['predicted_probability']:.4f} |\n"

    metrics_md = ""
    if metrics is not None and len(metrics) > 0:
        metrics_md = "| Fold | AUC | Accuracy | Precision | Recall | F1 |\n"
        metrics_md += "|---|---|---|---|---|---|\n"
        for _, r in metrics.iterrows():
            metrics_md += f"| {int(r['fold'])} | {r['auc']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n"
        means = metrics[["auc", "accuracy", "precision", "recall", "f1"]].mean()
        metrics_md += f"| **Mean** | **{means['auc']:.4f}** | **{means['accuracy']:.4f}** | **{means['precision']:.4f}** | **{means['recall']:.4f}** | **{means['f1']:.4f}** |\n"

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
    multi-drug interactions (DDIs).

    **Dataset**: FAERS 2025 Q1 — 1,318,424 rows (report x drug).

    ---

    ## Phase 1: Signal Detection

    Pairwise disproportionality analysis (ROR) identified **{n_signals:,}**
    statistically significant (drug pair, adverse reaction) signals from
    multi-drug reports.

    **Labeled pairs**: {n_pos:,} positive (DDI signal) / {n_neg:,} negative.

    ### Top 10 Strongest Signals

    {top_signals_md}

    ![Phase 1 Overview](../results/ddi_study/phase1_overview.png)

    ---

    ## Phase 2: Chemical Structure Mapping

    FAERS drug names were matched to DrugBank entries and converted to 1024-bit
    ECFP4 molecular fingerprints via RDKit. See
    `results/ddi_study/phase2_mapping_stats.txt` for full mapping statistics.

    ---

    ## Phase 3: Deep Neural Network

    A 4-layer DNN (2048 -> 512 -> 256 -> 128 -> 1) was trained with 5-fold
    stratified cross-validation.

    **Mean AUC**: {mean_auc:.4f} +/- {std_auc:.4f}

    ### Per-Fold Metrics

    {metrics_md}

    ![ROC Curves](../results/ddi_study/phase3_roc_curve.png)
    ![CV Metrics](../results/ddi_study/phase4_cv_metrics.png)

    ---

    ## Phase 4: Novel DDI Predictions

    The trained model scored all unseen drug pairs (never co-prescribed in FAERS).
    Top predictions represent chemically plausible but unreported combinations
    with the highest predicted interaction risk.

    ### Top 10 Predicted Novel DDIs

    {top_novel_md}

    ![Top Predictions](../results/ddi_study/phase4_top20_predictions.png)

    ### Signal Heatmap

    ![Heatmap](../results/ddi_study/phase4_signal_heatmap.png)

    ### Validation Against DrugBank

    {precision_md}

    ---

    ## Limitations

    - FAERS is a spontaneous reporting system and does not establish causation.
    - Drug name matching is approximate; ~20% of FAERS substances may lack
      DrugBank SMILES.
    - The model learns correlation between molecular substructure co-occurrence
      and reporting patterns, not pharmacokinetic mechanisms.
    - Single-quarter data may miss rare combinations.

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


def run_phase4(
    signals: pd.DataFrame,
    labeled: pd.DataFrame,
    fp_map: dict,
    model_state: dict,
    metrics: pd.DataFrame,
):
    """Execute Phase 4: score unseen pairs, validate, chart, report."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seen_pairs = set()
    for _, r in labeled.iterrows():
        seen_pairs.add((r["drug_a"], r["drug_b"]))

    predictions = score_unseen_pairs(fp_map, seen_pairs, model_state, device=device)
    predictions.head(100).to_csv(PHASE4_NOVEL, index=False)
    print(f"  Top 100 novel predictions saved to {PHASE4_NOVEL}")

    validation = validate_against_drugbank(predictions)
    if len(validation) > 0:
        validation.to_csv(PHASE4_VALIDATION, index=False)

    generate_visualizations(signals, predictions, validation, metrics)
    generate_report(signals, labeled, metrics, predictions, validation)


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Plan C: Molecular DDI Prediction")
    parser.add_argument("--phase", type=int, default=0, help="Run a specific phase (1-4). 0 = all.")
    parser.add_argument("--drugbank-vocab", default="data/drugbank_all_drugbank_vocabulary.csv.zip")
    parser.add_argument("--drugbank-sdf", default="data/drugbank_all_structures.sdf.zip")
    args = parser.parse_args()

    run_all = args.phase == 0

    signals, labeled, fp_map, model_state, metrics = None, None, None, None, None

    if run_all or args.phase == 1:
        signals, labeled = run_phase1()
        labeled.to_csv(RESULTS_DIR / "phase1_labeled_pairs.csv", index=False)

    if run_all or args.phase >= 2:
        if labeled is None:
            labeled = pd.read_csv(RESULTS_DIR / "phase1_labeled_pairs.csv", dtype=str)
            labeled["label"] = labeled["label"].astype(int)
        if signals is None and PHASE1_SIGNALS.exists():
            signals = pd.read_csv(PHASE1_SIGNALS)

    if run_all or args.phase == 2:
        fp_map, _ = run_phase2(labeled, args.drugbank_vocab, args.drugbank_sdf)
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
            model_state = torch.load(MODEL_PATH, map_location="cpu")
        if metrics is None and PHASE3_METRICS.exists():
            metrics = pd.read_csv(PHASE3_METRICS)
        if model_state is None:
            print("[Phase 4] No trained model found. Run Phase 3 first.")
            return
        run_phase4(signals, labeled, fp_map, model_state, metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()
