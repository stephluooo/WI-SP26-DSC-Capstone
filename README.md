# Amazon Supplement Reviews + DSLD Label Data
## Overview

We are building a dataset that connects **Amazon customer reviews** of dietary supplements with **official FDA/NIH label data** from the Dietary Supplement Label Database (DSLD). The goal is to analyze consumer feedback relative to product labels -- such as ingredient accuracy, dosage expectations, adverse reactions, and brand trust.

The **current database** is a single merged CSV where every row is an Amazon review joined to its corresponding DSLD label record (62 columns, ~16K rows, 1,647 unique products matched to 1,308 DSLD entries).

## Next Steps (By Week 2)

* Our **hypothesis** is that there is a relationship between features (ingredients, number of ingredients, brand name) and the average rating of the product. We intend to investigate this using methods including XGBoost (Karina) and neural net based regression (Liyun).
* We also intend to **join the Human Food Complaint System (HFCS) side effect database** to our current database and investigate relationships between ingredients, side effects, and ratings (Michael).

---

## Data Files
- Amazon Review 23' from McAuley Lab: https://amazon-reviews-2023.github.io/index.html
- iHerb: https://www.kaggle.com/datasets/crawlfeeds/iherb-products-dataset
- DSLD: https://dsld.od.nih.gov/

## Tools
- Cursor
- Github

---

## Repository Structure

```
├── README.md                          # This file
├── merge_amazon_dsld.py               # Main pipeline: merge Amazon + DSLD
├── load_reviews.py                    # Amazon review loader with supplement filtering
├── dsld_example.py                    # DSLD API usage example (10 labels)
├── openfda_example.py                 # OpenFDA adverse event API example
├── sample_merged.py                   # Sampling + noise injection for validation
├── show_merged.py                     # Quick viewer for merged CSV
├── analyze_gaps.py                    # Match quality diagnostics
├── cnn_rating_regression.py           # CNN regression: ingredients/brand → avg rating
├── .gitattributes                     # Git LFS tracking rules
├── data/
│   ├── amazon_dsld_merged.csv.zip     # Full merged dataset (LFS, 17 MB zipped)
│   ├── amazon_dsld_merged_sample_1k.csv   # 1,000-row random sample
│   ├── amazon_dsld_merged_sample_10k.csv  # 10,000-row random sample
│   ├── amazon_dsld_sample_210.csv     # 210-row sample (200 real + 10 noise)
│   ├── iherb_best_selling_products_clean_dataset.csv
│   ├── meta_Health_and_Personal_Care.jsonl.gz  # Amazon product metadata (LFS)
│   └── DSLD-full-database-CSV/        # Raw DSLD bulk export (CSV format)
└── results/
    └── cnn_regression/
        ├── metrics.json               # Final validation metrics (MAE, RMSE, R²)
        ├── training_history.csv       # Per-epoch train/val loss and MAE
        ├── val_predictions.csv        # Actual vs predicted ratings for val set
        └── model.pt                   # Trained PyTorch model weights
```

---

## Pipeline

### `merge_amazon_dsld.py`

End-to-end pipeline that produces the merged dataset. Runs in five steps:

1. **Load Amazon metadata** from `meta_Health_and_Personal_Care.jsonl.gz` (60K products). Apply a supplement pre-filter using regex signals (`supplement`, `capsule`, `vitamin`, `mg`, etc.) and an exclusion list (`shampoo`, `pillow`, `massager`, etc.) to keep only supplement products (~6.5K remain).
2. **Load Amazon reviews** from `Health_and_Personal_Care.jsonl.gz` (~494K reviews).
3. **Load DSLD** from `DSLD-full-database-JSON.zip` (215K supplement labels with full ingredient, dosage, claim, and contact data).
4. **Match** Amazon products to DSLD records using multi-strategy name matching:
   - Token-set overlap (order-independent word matching)
   - Substring matching (exact phrase containment)
   - Brand alignment enforcement (brand conflict penalty of 0.7x)
   - Generic name blocklist to prevent false matches on vague terms
   - Minimum score threshold of 0.7
5. **Merge** reviews, metadata, and DSLD records into one DataFrame; write to CSV.

### `load_reviews.py`

Standalone script to filter Amazon reviews for a specific supplement keyword (default: Magnesium). Applies the same supplement-signal and exclusion regexes used in the main pipeline. Useful for quick sampling.

### `dsld_example.py` / `openfda_example.py`

Minimal working examples for the NIH DSLD API and OpenFDA Food Adverse Event API, respectively. Pull 10 records each; useful for understanding the data sources.

### `sample_merged.py`

Samples 200 unique Amazon-DSLD match pairs and injects 10 synthetic false rows with garbled names and randomized IDs. Intended for match-quality validation and classifier testing.

### `analyze_gaps.py`

Diagnostics script that inspects unmatched Amazon products to identify gaps in the matching strategy (single-word names, brand variations, missing tokens).

### `show_merged.py`

Prints the first 10 rows of the merged CSV with selected columns for quick inspection.

### `cnn_rating_regression.py`

CNN-based regression pipeline that tests whether label features (ingredient names, ingredient count, brand name) predict a product's average Amazon rating. Deduplicates the 10K-row sample to unique products, builds a vocabulary from combined brand + ingredient text, and trains a 1D-CNN with an auxiliary numeric input (ingredient count). Outputs metrics, training history, predictions, and model weights to `results/cnn_regression/`.

---

## Merged Dataset Schema (62 columns)

### Amazon Review Fields

| Column | Type | Description |
|---|---|---|
| `parent_asin` | string | Parent product ASIN |
| `asin` | string | Specific variant ASIN |
| `review_rating` | float | Star rating (1.0--5.0) |
| `review_title` | string | Review headline |
| `review_text` | string | Full review body |
| `helpful_vote` | int | Helpful vote count |
| `verified_purchase` | bool | Verified purchase flag |
| `timestamp` | int | Unix timestamp (ms) |
| `user_id` | string | Anonymized reviewer ID |
| `amazon_title` | string | Product title on Amazon |
| `amazon_store` | string | Brand/store name on Amazon |
| `main_category` | string | Amazon product category |
| `average_rating` | float | Product-level average rating |
| `rating_number` | int | Total rating count |

### Match Fields

| Column | Type | Description |
|---|---|---|
| `dsld_id` | int | DSLD label ID |
| `match_score` | float | Match confidence (0.7--1.0) |
| `match_reason` | string | Match method: `substring+brand`, `brand+token_overlap`, `substring`, `high_token_overlap` |

### DSLD Product Fields

| Column | Type | Description |
|---|---|---|
| `dsld_full_name` | string | Product name in DSLD |
| `dsld_brand_name` | string | Brand name |
| `dsld_brand_ip_symbol` | string | Brand trademark symbol |
| `dsld_upc` | string | UPC/SKU code |
| `dsld_product_version_code` | string | Product version identifier |
| `dsld_nhanes_id` | string | NHANES cross-reference ID |
| `dsld_bundle_name` | string | Bundle name (if applicable) |
| `dsld_form` | string | Physical form (Capsule, Powder, Liquid, etc.) |
| `dsld_form_code` | string | LanguaL code for form |
| `dsld_product_type` | string | Product type (Vitamin, Mineral, Botanical, etc.) |
| `dsld_product_type_code` | string | LanguaL code for type |

### DSLD Serving Info

| Column | Type | Description |
|---|---|---|
| `dsld_servings_per_container` | float | Servings per container |
| `dsld_serving_size` | string | Display serving size |
| `dsld_serving_min_qty` / `max_qty` | float | Min/max serving quantity |
| `dsld_serving_unit` | string | Serving unit |
| `dsld_serving_daily_min` / `max` | float | Recommended daily servings |
| `dsld_net_contents` | string | Net contents display |

### DSLD Market Status

| Column | Type | Description |
|---|---|---|
| `dsld_off_market` | int | 0 = on market, 1 = off market |
| `dsld_entry_date` | string | Date added to DSLD |
| `dsld_off_market_date` | string | Date removed from market |

### DSLD Ingredients

| Column | Type | Description |
|---|---|---|
| `dsld_ingredient_count` | int | Number of active ingredients |
| `dsld_ingredient_names` | list | Ingredient name list |
| `dsld_ingredient_groups` | list | Ingredient group categories |
| `dsld_ingredient_doses` | list | Dose display strings (e.g., "Vitamin C: 500 mg (556% DV)") |
| `dsld_ingredients_detailed` | JSON | Full ingredient detail: name, group, dose, forms, alt names, UNII codes |
| `dsld_other_ingredients` | list | Inactive/excipient ingredient names |
| `dsld_other_ingredients_text` | string | Raw other-ingredients text from label |

### DSLD Claims and Statements

| Column | Type | Description |
|---|---|---|
| `dsld_claims` | list | Product claims (Nutrient, Structure/Function, etc.) |
| `dsld_target_groups` | list | Target demographics |
| `dsld_dv_user_groups` | list | Daily value target groups |
| `dsld_directions` | string | Usage/dosage directions |
| `dsld_fda_identity` | string | FDA statement of identity |
| `dsld_formula_contains` | string | Formula contains statement |
| `dsld_precautions_children` | string | Precautions for children |
| `dsld_precautions_pregnant` | string | Precautions for pregnancy/nursing |
| `dsld_precautions_other` | string | General precautions |
| `dsld_storage` | string | Storage instructions |
| `dsld_all_statements` | list | All label statements (raw) |

### DSLD Metadata

| Column | Type | Description |
|---|---|---|
| `dsld_contact_names` | list | Manufacturer/distributor contact names |
| `dsld_contact_info` | list | Contact details (city, state, phone, web) |
| `dsld_related_label_ids` | list | Related DSLD label IDs |
| `dsld_percent_dv_footnote` | string | Daily value footnote text |
| `dsld_has_outer_carton` | bool | Whether product has outer carton labeling |

---

## Analysis

### CNN Regression: Ingredients & Brand → Average Rating

**Hypothesis:** There is a relationship between label features (ingredient names, number of ingredients, brand name) and the average Amazon rating of a product.

**Method:** A 1D Convolutional Neural Network (Conv1D) regression model trained on the `amazon_dsld_merged_sample_10k.csv` dataset (PyTorch). The pipeline:

1. **Deduplicates** the 10K-row sample to **1,395 unique products** (one row per `parent_asin`, since `average_rating` is constant per product).
2. **Constructs a text feature** by combining brand name and ingredient names into a single lowercase string (e.g., `"brand nature made ingredients l-lysine"`).
3. **Tokenizes and encodes** the text into integer sequences (vocabulary size: ~1,800 tokens, max sequence length: 256).
4. **Scales** ingredient count with a standard scaler as a separate numeric input.
5. **Trains a CNN** with two Conv1D layers (128 → 64 filters, kernel size 5) over the embedded text, global max-pooled, concatenated with the ingredient count, and passed through dense layers to predict a single rating value.
6. Uses **80/20 train/val split**, MSE loss, Adam optimizer (lr=1e-3), and **early stopping** (patience=4).

**Results** (held-out 20% validation set, 280 products):

| Metric | Value |
|---|---|
| MAE | 0.504 |
| RMSE | 0.707 |
| R² | −0.025 |
| Naive baseline MAE (predict mean) | 0.509 |

**Interpretation:** The CNN's MAE (0.504) is only marginally better than always predicting the mean rating (0.509), and R² ≈ 0 indicates that ingredient names, ingredient count, and brand name alone explain almost none of the variance in average product ratings. This is consistent with the understanding that consumer ratings are driven by many factors beyond label content — taste, shipping experience, expectations, price, and individual biology — that are not captured in DSLD label data. The result serves as a useful baseline showing that label-derived features alone are insufficient predictors of product satisfaction.

Output files in `results/cnn_regression/`:

| File | Description |
|---|---|
| `metrics.json` | Final validation MAE, RMSE, R², and naive baseline MAE |
| `training_history.csv` | Per-epoch training MSE, validation MSE, and validation MAE |
| `val_predictions.csv` | Actual vs predicted average ratings for every validation product |
| `model.pt` | Trained PyTorch model weights |

---

## Notes

- **List columns** (`dsld_ingredient_names`, `dsld_claims`, etc.) are stored as Python list literals. Parse with `ast.literal_eval()`.
- **JSON column** (`dsld_ingredients_detailed`) contains structured ingredient data. Parse with `json.loads()`.
- **Timestamps** are Unix milliseconds. Divide by 1000 for standard epoch seconds.
- The full dataset is stored via **Git LFS** as `data/amazon_dsld_merged.csv.zip`. Run `git lfs pull` after cloning to retrieve it.

## Tasks and Division

### Liyun Luo
Done:
- Searched for different database
- EDA

Goals:
- CNN based rating regression on 10k sample datasets

### Michael Kroyan
Done:
- Data Access
- Data Cleaning
- Entity Resolution
- Merging Datasets

Goals:
- EDA
- Consult on potential AWS use (Dask)

### Karina Shah
Done: 
- Conducting EDA on the merged dataset to explore the data and find out what kind of questions would be interesting to ask. 

Goals:
- Make more exploratory charts and finish EDA so that we can narrow in on a question.
