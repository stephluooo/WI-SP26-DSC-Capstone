# HFCS Full Database Schema

**Source**: FDA CFSAN Adverse Event Reporting System (CAERS), formerly the Human Foods Complaint System (HFCS), accessed via the [OpenFDA Food Adverse Event API](https://open.fda.gov/apis/food/event/).

**Filter**: None — this is the complete, unfiltered HFCS database covering all food and supplement categories.

**Granularity**: One row per **(report × product)**. A single adverse event report may involve multiple products (e.g., a consumer taking a supplement alongside a conventional food), producing multiple rows sharing the same `report_number`.

**Size**: 231,552 rows from 148,156 unique reports (data from 2004 through early 2026).

---

## Columns

| Column | Type | Nulls | Description |
|---|---|---|---|
| `report_number` | string | 0 | Unique FDA report identifier. Shared across rows when a report involves multiple products. |
| `date_created` | int | 0 | Date the report was entered into the system, formatted as `YYYYMMDD` (e.g., `20091029`). |
| `date_started` | float | ~36K | Date the adverse event began, formatted as `YYYYMMDD`. Null when the consumer did not provide a start date. |
| `outcomes` | string | 0 | Pipe-delimited (`\|`) list of medical outcomes. Values include: `Death`, `Life Threatening`, `Hospitalization`, `Disability`, `Required Intervention`, `Visited Emergency Room`, `Visited a Health Care Provider`, `Other Serious or Important Medical Event`, `Other Serious Outcome`, `Other Outcome`, among others. A single report can have multiple outcomes. |
| `consumer_age` | float | ~22K | Age of the affected consumer at the time of the event. Units specified in `consumer_age_unit`. |
| `consumer_age_unit` | string | ~22K | Unit for `consumer_age`. Values: `year(s)`, `month(s)`, `week(s)`, `day(s)`, `decade(s)`. |
| `consumer_gender` | string | ~5K | Gender of the consumer. Values: `Male`, `Female`. |
| `product_role` | string | 0 | Role of the product in the adverse event. `SUSPECT` = believed to have caused the event. `CONCOMITANT` = taken concurrently but not suspected. |
| `product_name_brand` | string | ~1 | Brand name of the product as reported. Free-text, typically all-caps (e.g., `NATURE'S BOUNTY BIOTIN 5000MCG`). |
| `product_industry_code` | string | 0 | FDA industry code classifying the product. Common codes: `54` = Dietary Supplements, `29` = Baby Food, `41` = Dietary Conventional Foods, `20` = Fruit/Fruit Products, `53` = Cosmetics, etc. |
| `product_industry_name` | string | 0 | Human-readable label for `product_industry_code` (e.g., `Vit/Min/Prot/Unconv Diet(Human/Animal)`). |
| `reactions` | string | 0 | Pipe-delimited (`\|`) list of adverse reactions using MedDRA terminology (e.g., `CHOKING`, `PANCREATITIS`, `NAUSEA`). A single report can list many reactions. |

---

## Notes

- **Pipe-delimited fields**: `outcomes` and `reactions` contain multiple values separated by `|`. Split on `|` to analyze individual values.
- **SUSPECT vs CONCOMITANT**: For causal analysis, filter to `product_role = 'SUSPECT'`.
- **Industry codes**: The dataset spans all FDA food/supplement categories. Filter on `product_industry_code` to narrow scope (e.g., `54` for dietary supplements only).
- **Null demographics**: ~15-22% of reports are missing consumer age or gender.
- **Deduplication**: Rows share a `report_number` when multiple products are involved in a single event. Group by `report_number` for report-level analysis.
