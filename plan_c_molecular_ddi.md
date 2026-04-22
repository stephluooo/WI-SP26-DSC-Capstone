# Molecular Fingerprint-Based Multi-Drug Interaction Prediction

A study that detects multi-drug interaction (DDI) signals from the FDA Adverse Event Reporting System (FAERS) and builds a deep neural network to predict adverse drug interactions from molecular structure.

**References**:
- Zhang, X. et al. (2025). Identifying Drug Combinations Associated with Acute Kidney Injury. *Biomed J Sci & Tech Res*, 64(1).
- Schreier, T. et al. (2024). [Integration of FAERS, DrugBank and SIDER Data for ML-based Detection of ADRs](https://link.springer.com/article/10.1007/s13222-024-00486-1). *Datenbank-Spektrum*, 24, 233--242.
- Shen, Y. et al. (2020). [Mining and Visualizing High-Order Directional Drug Interaction Effects](https://link.springer.com/article/10.1186/s12911-020-1053-z). *BMC Med Inform Decis Mak*, 20, 48.

**Dataset**: `data/faers_full.csv.zip` -- 1,318,424 rows (report x drug), 2025 Q1.

---

## Phase 1: Signal Detection via Disproportionality Analysis

### Objective

Identify drug pairs that are reported together with specific adverse events more frequently than expected. These statistically significant (drug_pair, reaction) associations become the labeled training data for the neural network.

### Method

1. **Load and reconstruct reports**: Unzip `faers_full.csv.zip`, read into a DataFrame. Group rows by `safetyreportid` to reconstruct per-patient drug lists and reaction lists.

2. **Filter to suspect drugs**: Retain only rows where `drug_characterization = 1` (primary suspect). This removes concomitant medications that are unlikely to be causally related.

3. **Normalize drug names**: Lowercase and strip whitespace from `drug_active_substance`. Discard reports with fewer than 2 suspect drugs (single-drug reports cannot produce DDI signals).

4. **Generate pairwise combinations**: For each multi-drug report, generate all 2-drug combinations from the suspect drug list. Record which reactions co-occur with each pair.

5. **Build contingency tables**: For each unique (drug_pair, reaction) combination, compute a 2x2 table:

   |                  | Reaction present | Reaction absent |
   |------------------|-----------------|-----------------|
   | **Pair present** | a               | b               |
   | **Pair absent**  | c               | d               |

6. **Compute Reporting Odds Ratio (ROR)**:
   - ROR = (a * d) / (b * c)
   - 95% CI = exp(ln(ROR) +/- 1.96 * sqrt(1/a + 1/b + 1/c + 1/d))
   - Signal threshold: ROR > 2, lower 95% CI > 1, a >= 3

7. **Label drug pairs**: Pairs with at least one significant (ROR lower CI > 1.5) signal are labeled **positive**. Pairs below threshold are labeled **negative**.

### Output

- `results/ddi_study/phase1_signals.csv` -- all significant DDI signals with ROR, CI, reaction, and case count.
- Labeled pair dataset for Phase 3 training.

---

## Phase 2: Chemical Structure Mapping

### Objective

Map FAERS drug names to molecular structures (SMILES) via DrugBank, then compute Extended Connectivity Fingerprints (ECFP4) as fixed-length numerical representations of each drug.

### Method

1. **Parse DrugBank**: Download the DrugBank XML (academic license, free). Extract for each drug: DrugBank ID, name, synonyms, product names, SMILES string, and ATC codes.

2. **Fuzzy match FAERS names to DrugBank**: For each unique `drug_active_substance` in the labeled pair dataset:
   - Exact match on drug name (case-insensitive)
   - Exact match on synonyms
   - Exact match on product names
   - Fuzzy match (Levenshtein ratio > 0.9) as fallback
   - Expected match rate: ~80% based on Schreier et al. (2024)

3. **Retrieve SMILES**: For matched drugs, extract the SMILES string. Discard pairs where either drug cannot be mapped.

4. **Compute ECFP4 fingerprints**: Using RDKit:
   ```python
   from rdkit import Chem
   from rdkit.Chem import AllChem
   mol = Chem.MolFromSmiles(smiles)
   fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
   ```
   Each drug becomes a 1024-bit binary vector encoding molecular substructure presence/absence within a 2-bond radius.

### Output

- Drug-to-fingerprint mapping dictionary (drug_name -> 1024-bit vector).
- `results/ddi_study/phase2_mapping_stats.txt` -- mapping success rate, unmapped drugs list.

---

## Phase 3: Deep Neural Network Training

### Objective

Train a binary classifier that takes two drugs' molecular fingerprints as input and predicts whether their combination is a DDI signal (positive) or not (negative).

### Input Representation

For each labeled drug pair (drug_A, drug_B):
- Concatenate ECFP4(drug_A) + ECFP4(drug_B) = 2048-dimensional binary vector
- Ensure consistent ordering: always alphabetically sort the pair to avoid (A,B) != (B,A)

### Labels

From Phase 1:
- **Positive**: Drug pair has at least one (pair, reaction) signal with ROR lower 95% CI > 1.5
- **Negative**: Drug pair appears in reports but has no significant signal

### Architecture

```
Input (2048)
  -> Linear(2048, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(512, 256)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(256, 128)  -> BatchNorm -> ReLU -> Dropout(0.3)
  -> Linear(128, 1)    -> Sigmoid
```

### Training Protocol

- **Loss**: Binary cross-entropy
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Evaluation**: 5-fold stratified cross-validation
- **Epochs**: 50 per fold with early stopping (patience=5) on validation AUC
- **Batch size**: 256
- **Class imbalance**: Apply positive class weight = n_negative / n_positive

### Metrics

- AUC-ROC (primary)
- Accuracy, precision, recall, F1
- Per-fold and mean +/- std across folds

### Output

- Best model checkpoint (`.pt` file)
- `results/ddi_study/phase3_roc_curve.png` -- ROC curves for all 5 folds
- `results/ddi_study/phase3_metrics.csv` -- per-fold metrics table

---

## Phase 4: Analysis, Prediction on Unseen Pairs, and Validation

### Objective

Apply the trained model to drug pairs that have never co-occurred in FAERS reports to discover novel potential DDIs. Validate predictions against known interaction databases.

### Method

1. **Generate unseen pairs**: From all drugs with ECFP fingerprints, generate all possible pairwise combinations. Exclude pairs already present in the training data (seen in FAERS reports).

2. **Score unseen pairs**: Run each pair through the trained model. Output a probability (0-1) of interaction risk.

3. **Rank and filter**: Sort by predicted probability descending. Report the top 100 highest-risk novel drug pairs.

4. **Validate against DrugBank DDI list**: DrugBank contains a curated list of known drug-drug interactions. Check how many of the top-100 predicted pairs are confirmed DDIs in DrugBank. Compute precision@k for k = 10, 25, 50, 100.

5. **Substructure analysis**: For the top predicted pairs, identify which ECFP bits (molecular substructures) are most activated. This provides chemical interpretability -- which structural motifs drive predicted interactions.

### Visualizations

- **Bar chart**: Top 20 predicted novel DDI pairs with risk scores
- **ROC curve**: 5-fold cross-validation performance
- **Precision@k curve**: Validation against DrugBank known DDIs
- **Heatmap**: Top DDI signals from Phase 1 (drug x reaction matrix)
- **Distribution plot**: ROR distribution of Phase 1 signals
- **Scatter plot**: Model confidence vs ROR for known pairs (calibration check)

### Output

- `results/ddi_study/phase4_novel_predictions.csv` -- top 100 predicted novel DDIs
- `results/ddi_study/phase4_validation.csv` -- precision@k results
- All chart PNGs in `results/ddi_study/`
- `reports/ddi_molecular_study.md` -- full markdown report with inline images, tables, statistical summaries, and interpretation

---

## Dependencies

| Package | Purpose | Install |
|---|---|---|
| pandas, numpy | Data manipulation | `pip install pandas numpy` |
| scipy | ROR computation, statistics | `pip install scipy` |
| rdkit | SMILES parsing, ECFP fingerprints | `pip install rdkit` |
| torch | Neural network | `pip install torch` |
| scikit-learn | Cross-validation, metrics | `pip install scikit-learn` |
| matplotlib | Charts | `pip install matplotlib` |
| rapidfuzz | Fuzzy string matching for drug names | `pip install rapidfuzz` |

---

## File Structure

```
ddi_study.py                          # Main script (all 4 phases)
plan_c_molecular_ddi.md               # This document
data/faers_full.csv.zip               # Input FAERS data (already present)
data/drugbank_vocabulary.csv          # DrugBank drug names + SMILES (to download)
results/ddi_study/                    # Output charts and CSVs
reports/ddi_molecular_study.md        # Final markdown report
```
