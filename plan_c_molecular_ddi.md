# Plan C: Molecular Fingerprint-Based Multi-Drug Interaction Prediction

A study that detects multi-drug interaction (DDI) signals from the FDA Adverse Event Reporting System (FAERS) and builds a deep neural network to predict adverse drug interactions from molecular structure.

**References**:
- Zhang, X. et al. (2025). Identifying Drug Combinations Associated with Acute Kidney Injury. *Biomed J Sci & Tech Res*, 64(1).
- Schreier, T. et al. (2024). [Integration of FAERS, DrugBank and SIDER Data for ML-based Detection of ADRs](https://link.springer.com/article/10.1007/s13222-024-00486-1). *Datenbank-Spektrum*, 24, 233--242.
- Shen, Y. et al. (2020). [Mining and Visualizing High-Order Directional Drug Interaction Effects](https://link.springer.com/article/10.1186/s12911-020-1053-z). *BMC Med Inform Decis Mak*, 20, 48.

**Dataset**: `data/faers_full.csv.zip` -- full FAERS database (all available years), downloaded via `download_faers.py`.

---

## Architecture

Drug name canonicalization via DrugBank happens **before** pair generation. This eliminates false signals from brand/generic name variants (e.g. "Fosamax" and "alendronate sodium" both resolve to DrugBank ID `DB00710`) and prevents self-pair artifacts in downstream predictions.

```
                  ┌──────────────┐
                  │  DrugBank    │
                  │  Vocab + SDF │
                  └──────┬───────┘
                         │
                  build_canonicalizer()
                         │
            ┌────────────▼────────────┐
            │  name_to_dbid lookup    │
            │  dbid_to_name mapping   │
            │  dbid_to_smiles mapping │
            └────────────┬────────────┘
                         │
     ┌───────────────────┼───────────────────┐
     ▼                   ▼                   ▼
  Phase 1            Phase 2            Phase 4
  (uses lookup       (uses SMILES)      (uses lookup
   for pairing)                          for display)
```

---

## Phase 1: Signal Detection via Disproportionality Analysis

### Objective

Identify drug pairs reported together with specific adverse events more frequently than expected. These statistically significant (drug_pair, reaction) associations become labeled training data.

### Method

1. **Load FAERS** (Parquet preferred, falls back to CSV). Read only the 5 columns needed: `safetyreportid`, `drug_characterization`, `drug_active_substance`, `drug_name`, `reactions`.

2. **Canonicalize drug names**: Map each `drug_active_substance` (falling back to `drug_name`) to a DrugBank ID via the unified lookup. Unmapped drugs are dropped -- they cannot be fingerprinted later.

3. **Reconstruct reports**: Group by `safetyreportid`, retain only suspect drugs (`drug_characterization = 1`). Discard reports with fewer than 2 canonical drugs.

4. **Generate pairwise combinations**: For each multi-drug report, generate all 2-drug combinations using DrugBank IDs.

5. **Build contingency tables**: For each (drug_pair, reaction):

   |                  | Reaction present | Reaction absent |
   |------------------|-----------------|-----------------|
   | **Pair present** | a               | b               |
   | **Pair absent**  | c               | d               |

6. **Compute Reporting Odds Ratio (ROR)**:
   - ROR = (a * d) / (b * c)
   - 95% CI = exp(ln(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d))
   - **Filters**: a >= 3, b >= 3, ROR > 2, CI lower bound > 1.5

7. **Label drug pairs**: Pairs with at least one significant signal are labeled **positive**. Remaining pairs are **negative**.

### Output

- `results/ddi_study/phase1_signals.csv` -- all significant DDI signals
- `results/ddi_study/phase1_signals_named.csv` -- same, with human-readable drug names
- `results/ddi_study/phase1_labeled_pairs.csv` -- binary-labeled pair dataset
- `results/ddi_study/phase1_overview.png` -- ROR distribution + top reactions chart
- `results/ddi_study/phase2_match_details.csv` -- per-drug canonicalization audit trail

---

## Phase 2: ECFP4 Fingerprint Computation

### Objective

Compute 1024-bit Extended Connectivity Fingerprints (ECFP4) for each DrugBank ID that appears in the labeled pair dataset.

### Method

1. For each DrugBank ID in the labeled pairs, retrieve its SMILES string from the SDF.
2. Parse SMILES with RDKit and compute ECFP4 (Morgan fingerprint, radius=2, 1024 bits).
3. Drugs without SMILES or with unparseable structures are excluded.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles(smiles)
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
```

### Output

- `results/ddi_study/phase2_fingerprints.npz` -- DrugBank ID -> 1024-bit vector mapping
- `results/ddi_study/phase2_mapping_stats.txt` -- success/failure statistics

---

## Phase 3: Deep Neural Network Training

### Objective

Train a binary classifier: given two drugs' concatenated fingerprints, predict whether their combination is a DDI signal.

### Input Representation

For each labeled drug pair (drug_A, drug_B):
- Alphabetically sort by DrugBank ID (ensures (A,B) == (B,A))
- Concatenate ECFP4(drug_A) + ECFP4(drug_B) = 2048-dimensional binary vector

### Architecture

```
Input (2048)
  -> Linear(2048, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(512, 256)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(256, 128)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(128, 1)    [logits, BCEWithLogitsLoss]
```

### Training Protocol

- **Loss**: BCEWithLogitsLoss with positive class weight (n_neg / n_pos)
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Evaluation**: 5-fold stratified cross-validation
- **Epochs**: 50 per fold, early stopping (patience=5) on validation AUC
- **Batch size**: 256

### Output

- `results/ddi_study/best_model.pt` -- best model weights
- `results/ddi_study/phase3_roc_curve.png` -- ROC curves for all 5 folds
- `results/ddi_study/phase3_metrics.csv` -- per-fold AUC, accuracy, precision, recall, F1

---

## Phase 4: Novel DDI Prediction, Validation, and Report

### Objective

Score drug pairs never seen together in FAERS reports, validate against known interactions, and generate a full study report.

### Safeguards Against False Positives

1. **Minimum exposure filter**: Each drug must appear in at least 50 training pairs to be eligible for novel scoring. This prevents the model from assigning high probability to drugs it barely observed.

2. **Deduplication by DrugBank ID**: Self-pairs and duplicate ID pairs are removed. Because canonicalization already collapsed brand/generic variants, this should be a no-op -- but serves as a safety net.

3. **Streaming top-k scoring**: Pairs are scored in batches of 2048. A min-heap retains only the top 500 predictions at any time, keeping memory constant.

### Validation

If `data/drugbank_all_drug_drug_interactions.csv` is present (download from DrugBank), the top predictions are checked against known DDI pairs. Precision@k is computed for k = 10, 25, 50, 100, 200, 500.

### Output

- `results/ddi_study/phase4_novel_predictions.csv` -- top 500 predicted novel DDIs (with DrugBank IDs and names)
- `results/ddi_study/phase4_validation.csv` -- precision@k results
- `results/ddi_study/phase4_top20_predictions.png` -- bar chart of top 20 pairs
- `results/ddi_study/phase4_signal_heatmap.png` -- drug x reaction heatmap
- `results/ddi_study/phase4_cv_metrics.png` -- CV performance summary
- `reports/ddi_molecular_study.md` -- full markdown report with tables and charts

---

## Dependencies

| Package | Purpose | Install |
|---|---|---|
| pandas, numpy | Data manipulation | `pip install pandas numpy` |
| scipy | ROR computation | `pip install scipy` |
| rdkit | SMILES parsing, ECFP fingerprints | `pip install rdkit` |
| torch | Neural network | `pip install torch` |
| scikit-learn | Cross-validation, metrics | `pip install scikit-learn` |
| matplotlib | Charts | `pip install matplotlib` |
| rapidfuzz | Fuzzy string matching | `pip install rapidfuzz` |
| pyarrow | Parquet I/O | `pip install pyarrow` |

---

## Usage

```bash
# Full pipeline (all phases)
python ddi_study.py

# Individual phases (artifacts persist between runs)
python ddi_study.py --phase 1
python ddi_study.py --phase 2
python ddi_study.py --phase 3
python ddi_study.py --phase 4

# Custom minimum exposure threshold
python ddi_study.py --phase 4 --min-exposure 100

# Verify drug name canonicalization
python verify_drugbank_matches.py
python verify_drugbank_matches.py --search metformin
python verify_drugbank_matches.py --unmatched
```

---

## File Structure

```
ddi_study.py                              # Main pipeline (all 4 phases)
verify_drugbank_matches.py                # Canonicalization audit tool
download_faers.py                         # FAERS bulk downloader (--resume support)
convert_to_parquet.py                     # CSV -> Parquet conversion (chunked)
aws_setup.sh                              # EC2 environment setup
plan_c_molecular_ddi.md                   # This document
data/
  faers_full.csv.zip                      # FAERS data (LFS)
  faers_full.parquet                      # FAERS Parquet (local, not tracked)
  drugbank_all_drugbank_vocabulary.csv.zip # DrugBank vocabulary (LFS)
  drugbank_all_structures.sdf.zip         # DrugBank structures + SMILES
  drugbank_all_drug_drug_interactions.csv  # DrugBank known DDIs (optional)
results/ddi_study/                        # All phase outputs
reports/ddi_molecular_study.md            # Final study report
```
