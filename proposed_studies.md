# Proposed Studies Using the FAERS Dataset

Ten study proposals for the FDA Adverse Event Reporting System (FAERS) dataset, informed by recent pharmacovigilance research:

- Schreier, T., Tropmann-Frick, M. & Böhm, R. (2024). [Integration of FAERS, DrugBank and SIDER Data for Machine Learning-based Detection of Adverse Drug Reactions](https://link.springer.com/article/10.1007/s13222-024-00486-1). *Datenbank-Spektrum*, 24, 233–242.
- Davis, R., Dang, O., De, S. & Ball, R. (2026). [Characterizing the FDA Adverse Event Reporting System (FAERS) as a Network to Improve Pattern Discovery](https://link.springer.com/article/10.1007/s40264-025-01609-7). *Drug Safety*, 49, 239–251.

---

## 1. Disproportionality Analysis of Top-Prescribed Drugs in 2025

**Method**: Compute ROR, PRR, and IC for the most frequently reported drugs in 2025 Q1 data. Compare detected signals against current drug labels.

**Rationale**: 2025 Q1 is recent enough to potentially catch signals not yet reflected in drug labels. Davis et al. (2026) analyzed 2016–2023; this extends the timeline. Standard DPA is the baseline method the FDA itself uses, making results directly comparable to regulatory practice.

**Data**: FAERS only. **Complexity**: Low.

---

## 2. ML-Based Signal Detection (Schreier et al. Replication)

**Method**: Replicate the Schreier et al. (2024) pipeline — integrate FAERS data with DrugBank and SIDER, build ML classifiers (XGBoost, KNN) using contingency table counts plus drug/event features, and compare to DPA baselines.

**Rationale**: Their study used data through Q2 2022. Testing whether ML outperforms DPA (recall 0.81 vs 0.75) and whether event name, drug name, and ATC code remain the top SHAP features on newer data provides direct temporal validation of their findings.

**Data**: FAERS + DrugBank + SIDER. **Complexity**: High.

---

## 3. Network Analysis of Drug–Adverse Event Co-occurrence

**Method**: Build a bipartite network (drugs and MedDRA reaction terms as nodes, weighted by co-occurrence). Compute degree distributions, clustering coefficients, and community detection. Compare structural properties to Davis et al. (2026).

**Rationale**: Davis et al. found FAERS networks are heavy-tailed and highly clustered with log-normal degree distributions. Verifying this on a single-quarter snapshot tests whether network topology is stable over time. Community detection could reveal drug classes sharing ADR profiles.

**Data**: FAERS only. **Complexity**: Medium.

---

## 4. Polypharmacy Risk: Drug Combination Adverse Event Enrichment

**Method**: For reports listing 3+ drugs, compute which drug *pairs* are disproportionately associated with serious outcomes (death, hospitalization) compared to the same drugs appearing alone. Use odds ratios with confidence intervals.

**Rationale**: Drug-drug interactions are a major clinical concern. Schreier et al. noted that incorporating additional features improves ADR detection — drug co-administration is one such feature. Directly actionable for clinical decision support.

**Data**: FAERS only. **Complexity**: Medium.

---

## 5. Demographic Disparities in Adverse Drug Reactions

**Method**: Stratify by patient sex and age bins. For the top 50 drugs, compute sex- and age-specific RORs for their most common reactions. Test for significant differences in reporting rates across demographics.

**Rationale**: Davis et al. focused on structural network properties but did not examine demographics. The FAERS schema includes age, sex, and reporter country, enabling population-specific frequency analysis. Pharmacogenomic differences across populations make this clinically relevant.

**Data**: FAERS only. **Complexity**: Low–Medium.

---

## 6. Reporter Qualification Impact on Signal Quality

**Method**: Compare signals detected from physician-reported events (`reporter_qualification = 1`) vs consumer-reported events (`5`). Compute precision of signals against known ADR labels from SIDER for each reporter type.

**Rationale**: There is an open debate about whether consumer reports add noise or genuine signals. If physician reports yield higher-precision signals, that validates filtering strategies. If consumer reports detect signals earlier, that argues for inclusive approaches.

**Data**: FAERS + SIDER. **Complexity**: Medium.

---

## 7. Geographic Variation in Adverse Event Profiles

**Method**: Group by `occurcountry`. For the top 20 drugs reported globally, compare reaction profiles across US, EU, Japan, and other regions using chi-squared tests and network visualization.

**Rationale**: Drug metabolism varies by population genetics (e.g., CYP2D6 polymorphisms). Geographic reporting differences could reflect genuine pharmacogenomic variation or regulatory/cultural reporting biases. Either finding is publishable.

**Data**: FAERS only. **Complexity**: Medium.

---

## 8. Temporal Signal Detection: Early Warning System Prototype

**Method**: Split Q1 2025 into weekly bins. Track the emergence of drug-reaction pairs week-over-week using cumulative ROR. Identify pairs where the signal strengthens over time vs those that are transient.

**Rationale**: Wang et al. (2020), cited in Schreier et al., showed CNN-based methods detect signals earlier than statistical methods. A temporal analysis on weekly data could prototype a real-time surveillance dashboard with immediate clinical relevance.

**Data**: FAERS only. **Complexity**: Medium.

---

## 9. Suspect vs. Concomitant Drug Misclassification

**Method**: For reports where `drug_characterization = 2` (concomitant), check if that drug has known ADRs matching the reported reactions via SIDER lookup. Estimate the rate of potential misclassification where concomitant drugs may actually be the suspect.

**Rationale**: Davis et al. noted that FAERS data quality is a known concern. Misclassification of suspect vs concomitant drugs could suppress real signals. Quantifying this would inform FDA data quality improvement and is a novel contribution.

**Data**: FAERS + SIDER. **Complexity**: Medium.

---

## 10. Serious Outcome Prediction from Drug and Patient Features

**Method**: Build a binary classifier predicting `serious = 1` from drug name, route, dosage form, patient age, sex, and number of concomitant drugs. Compare logistic regression, random forest, and XGBoost. Use SHAP for feature importance.

**Rationale**: This flips the typical FAERS study (which focuses on *which* ADR occurs) to focus on *severity* prediction. Schreier et al. showed SHAP analysis is effective on FAERS-derived models. Predicting severity at report submission could help prioritize FDA review queues.

**Data**: FAERS only. **Complexity**: Medium.

---

## Feasibility Summary

| # | Study | External Data | Complexity |
|---|---|---|---|
| 1 | DPA of top drugs | None | Low |
| 2 | ML replication | DrugBank + SIDER | High |
| 3 | Network analysis | None | Medium |
| 4 | Polypharmacy risk | None | Medium |
| 5 | Demographic disparities | None | Low–Medium |
| 6 | Reporter quality impact | SIDER | Medium |
| 7 | Geographic variation | None | Medium |
| 8 | Temporal signal detection | None | Medium |
| 9 | Suspect misclassification | SIDER | Medium |
| 10 | Severity prediction | None | Medium |

Studies 1, 3, 4, 5, 7, 8, and 10 can be completed entirely with the FAERS data already downloaded. Studies 2, 6, and 9 require DrugBank and/or SIDER (both freely available for academic use).
