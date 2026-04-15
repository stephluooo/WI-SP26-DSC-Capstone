# FAERS Dataset Schema

**Source**: FDA Adverse Event Reporting System (FAERS), accessed via [OpenFDA bulk download files](https://open.fda.gov/apis/drug/event/download/).

**Coverage**: 2025 Q1 (partial — through August 2025 based on available partition files at time of download).

**Granularity**: One row per **(report × drug)**. A single adverse event report may involve multiple drugs (suspect, concomitant, interacting), producing multiple rows sharing the same `safetyreportid`.

**Size**: 1,318,424 rows, 23 columns. Compressed to ~30 MB (ZIP).

---

## Columns

| Column | Type | Description |
|---|---|---|
| `safetyreportid` | string | Unique FDA safety report identifier. Shared across rows when a report involves multiple drugs. |
| `receivedate` | string | Date FDA received the report, `YYYYMMDD`. |
| `receiptdate` | string | Date the report was first received by the sender, `YYYYMMDD`. |
| `serious` | string | `1` = serious event, `2` = non-serious. |
| `seriousnessdeath` | string | `1` = patient died, `2` = no death reported. |
| `seriousnesshospitalization` | string | `1` = hospitalized, `2` = not. |
| `seriousnessdisabling` | string | `1` = disabling outcome, `2` = not. |
| `seriousnesslifethreatening` | string | `1` = life-threatening, `2` = not. |
| `seriousnessother` | string | `1` = other serious outcome, `2` = not. |
| `occurcountry` | string | Two-letter country code where the event occurred (e.g., `US`, `JP`, `FR`). |
| `reportercountry` | string | Country of the primary reporter. |
| `reporter_qualification` | string | Reporter type: `1` = physician, `2` = pharmacist, `3` = other health professional, `4` = lawyer, `5` = consumer/non-health professional. |
| `patient_age` | string | Patient age at onset. Units in `patient_age_unit`. |
| `patient_age_unit` | string | Age unit code: `800` = decade, `801` = year, `802` = month, `803` = week, `804` = day, `805` = hour. |
| `patient_sex` | string | `1` = male, `2` = female, `0` = unknown. |
| `patient_death_date` | string | Date of patient death, `YYYYMMDD`. Empty if patient did not die. |
| `drug_characterization` | string | Drug's role: `1` = suspect, `2` = concomitant, `3` = interacting. |
| `drug_name` | string | Drug name as reported (brand or generic, free-text). |
| `drug_dosage_form` | string | Dosage form (e.g., `TABLET`, `CAPSULE`, `INJECTION`). |
| `drug_route` | string | Route of administration code (e.g., `048` = oral, `065` = subcutaneous). |
| `drug_indication` | string | Reason the drug was prescribed (MedDRA term). |
| `drug_active_substance` | string | Active ingredient name (standardized). |
| `reactions` | string | Pipe-delimited (`\|`) list of adverse reactions using MedDRA preferred terms (e.g., `NAUSEA\|VOMITING\|HEADACHE`). |

---

## Notes

- **Suspect vs Concomitant**: For causal analysis, filter to `drug_characterization = 1` (suspect drug). `2` = concomitant (taken alongside but not suspected), `3` = interacting.
- **Seriousness fields**: Multiple seriousness flags can be `1` simultaneously (e.g., both hospitalization and life-threatening).
- **Pipe-delimited reactions**: Split on `|` to analyze individual adverse events.
- **Reporter qualification**: Physician-reported events (`1`) are generally considered higher quality than consumer reports (`5`).
- **Deduplication**: Group by `safetyreportid` for report-level analysis. Multiple rows per report represent different drugs involved in the same event.
- **Common analytical methods**: Reporting Odds Ratio (ROR), Proportional Reporting Ratio (PRR), Information Component (IC), Empirical Bayes Geometric Mean (EBGM).
