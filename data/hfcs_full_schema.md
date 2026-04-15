# HFCS Full Database Schema

**Source**: FDA CFSAN Adverse Event Reporting System (CAERS), formerly the Human Foods Complaint System (HFCS), accessed via the [OpenFDA Food Adverse Event API](https://open.fda.gov/apis/food/event/).

**Filter**: `products.industry_code = 54` (Vit/Min/Prot/Unconv Diet — dietary supplements for humans and animals).

**Granularity**: One row per **(report × product)**. A single adverse event report may involve multiple products (e.g., a consumer taking three supplements simultaneously), producing multiple rows sharing the same `report_number`.

**Size**: 102,916 rows from 53,603 unique reports (data through early 2026).

---

## Columns

| Column | Type | Nulls | Description |
|---|---|---|---|
| `report_number` | string | 0 | Unique FDA report identifier. Shared across rows when a report involves multiple products. |
| `date_created` | int | 0 | Date the report was entered into the system, formatted as `YYYYMMDD` (e.g., `20091029`). |
| `date_started` | float | 36,257 | Date the adverse event began, formatted as `YYYYMMDD`. Null when the consumer did not provide a start date. |
| `outcomes` | string | 0 | Pipe-delimited (`\|`) list of medical outcomes for the event. Values include: `Death`, `Life Threatening`, `Hospitalization`, `Disability`, `Required Intervention`, `Visited Emergency Room`, `Visited a Health Care Provider`, `Other Serious or Important Medical Event`, `Other Serious Outcome`, `Other Outcome`, among others. A single report can have multiple outcomes. |
| `consumer_age` | float | 22,210 | Age of the affected consumer at the time of the event. Units are specified in `consumer_age_unit`. |
| `consumer_age_unit` | string | 22,210 | Unit for `consumer_age`. Values: `year(s)`, `month(s)`, `week(s)`, `day(s)`, `decade(s)`. |
| `consumer_gender` | string | 4,561 | Gender of the consumer. Values: `Male`, `Female`. |
| `product_role` | string | 0 | Role of the product in the adverse event. Values: `SUSPECT` (product believed to have caused the event) or `CONCOMITANT` (product was being taken concurrently but is not suspected). |
| `product_name_brand` | string | 1 | Brand name of the product as reported. All-caps, free-text (e.g., `NATURE'S BOUNTY BIOTIN 5000MCG`). |
| `product_industry_code` | string | 0 | FDA industry code for the product. Code `54` is the primary filter (dietary supplements), but concomitant products may have other codes (e.g., `53` for cosmetics, `29` for baby food). 47 unique codes in the dataset. |
| `product_industry_name` | string | 0 | Human-readable label for `product_industry_code` (e.g., `Vit/Min/Prot/Unconv Diet(Human/Animal)`). |
| `reactions` | string | 0 | Pipe-delimited (`\|`) list of adverse reactions reported. Uses MedDRA terminology (e.g., `CHOKING`, `PANCREATITIS`, `INFECTION`). A single report can list many reactions. 32,117 unique combinations across the dataset. |

---

## Notes

- **Pipe-delimited fields**: `outcomes` and `reactions` contain multiple values separated by `|`. Split on `|` to analyze individual outcomes or reactions.
- **Null ages**: ~21% of reports are missing consumer age data.
- **SUSPECT vs CONCOMITANT**: For causal analysis, filter to `product_role = 'SUSPECT'`. Concomitant products were being used at the same time but are not implicated.
- **Industry codes**: While the pull filters on code `54`, each report can list products from other industries (e.g., a conventional food consumed alongside a supplement). Non-54 rows represent concomitant products.
