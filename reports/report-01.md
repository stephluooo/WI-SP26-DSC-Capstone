# Report 01

## Data Files
- Amazon Review 23' from McAuley Lab: https://amazon-reviews-2023.github.io/index.html
- iHerb: https://www.kaggle.com/datasets/crawlfeeds/iherb-products-dataset
- DSLD: https://dsld.od.nih.gov/

## Tools
- Cursor
- Github

## Tasks and Division

### Liyun Luo
Done:
- Searched for different datasets

Goals:
- EDA

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

## Overview
This section describes all columns in the `amazon_dsld_merged.csv` dataset and their data types.

The dataset is a merged collection combining Amazon product review data with the Dietary Supplement Label Database (DSLD). Each row represents a single review that has been matched to a corresponding supplement product in the DSLD.

---

## Amazon Review Columns


| Column              | Data Type | Description                                                                                                                                                              |
| ------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `parent_asin`       | string    | Amazon parent product ASIN (Amazon Standard Identification Number). Identifies the product being reviewed; may differ from `asin` when reviews are for product variants. |
| `asin`              | string    | Amazon Standard Identification Number for the specific product variant.                                                                                                  |
| `review_rating`     | float     | The star rating given in the review (typically 1.0–5.0).                                                                                                                 |
| `review_title`      | string    | Title of the customer review.                                                                                                                                            |
| `review_text`       | string    | Full text content of the customer review.                                                                                                                                |
| `helpful_vote`      | integer   | Number of "helpful" votes the review received from other customers.                                                                                                      |
| `verified_purchase` | boolean   | Indicates whether the review was from a verified purchase (True/False).                                                                                                  |
| `timestamp`         | integer   | Unix timestamp in milliseconds representing when the review was posted.                                                                                                  |
| `user_id`           | string    | Anonymized identifier for the user who wrote the review.                                                                                                                 |
| `amazon_title`      | string    | Product title as displayed on Amazon.                                                                                                                                    |
| `amazon_store`      | string    | Brand or store name as listed on Amazon.                                                                                                                                 |
| `main_category`     | string    | Primary product category on Amazon (e.g., "Health & Personal Care").                                                                                                     |
| `average_rating`    | float     | Overall average star rating of the product.                                                                                                                              |
| `rating_number`     | integer   | Total number of ratings/reviews for the product.                                                                                                                         |


---

## DSLD Match Columns


| Column         | Data Type | Description                                                                                               |
| -------------- | --------- | --------------------------------------------------------------------------------------------------------- |
| `dsld_id`      | integer   | Unique identifier for the product in the Dietary Supplement Label Database.                               |
| `match_score`  | float     | Similarity score (0.0–1.0) indicating how well the Amazon product matched the DSLD record.                |
| `match_reason` | string    | Method used for matching (e.g., `substring`, `high_token_overlap`, `short_substring`, `substring+brand`). |


---

## DSLD Product Columns


| Column                        | Data Type | Description                                                                                                                                                                              |
| ----------------------------- | --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dsld_full_name`              | string    | Full product name as recorded in the DSLD.                                                                                                                                               |
| `dsld_brand_name`             | string    | Brand name of the supplement.                                                                                                                                                            |
| `dsld_upc`                    | string    | Universal Product Code (UPC) for the supplement.                                                                                                                                         |
| `dsld_form`                   | string    | Physical form of the supplement (e.g., Capsule, Powder, Liquid, Tablet or Pill, Softgel Capsule, Other).                                                                                 |
| `dsld_product_type`           | string    | Classification of the supplement type (e.g., Mineral, Botanical, Botanical with Nutrients, Other Combinations, Non-Nutrient/Non-Botanical, Amino acid/Protein, Fat/Fatty Acid, Vitamin). |
| `dsld_servings_per_container` | integer   | Number of servings per container; may be null/empty.                                                                                                                                     |
| `dsld_off_market`             | integer   | Indicates whether the product is off the market (0 = on market, 1 = off market).                                                                                                         |
| `dsld_entry_date`             | string    | Date the product was added to the DSLD (YYYY-MM-DD format).                                                                                                                              |


---

## DSLD Ingredient and Label Columns


| Column                   | Data Type | Description                                                                                                           |
| ------------------------ | --------- | --------------------------------------------------------------------------------------------------------------------- |
| `dsld_ingredient_names`  | string    | List of ingredient names; stored as a string representation of a Python list (e.g., `['Vitamin C', 'Rose Hips']`).    |
| `dsld_ingredient_groups` | string    | Grouped or categorized ingredient names; stored as a string representation of a Python list.                          |
| `dsld_ingredient_count`  | integer   | Number of ingredients in the product.                                                                                 |
| `dsld_claims`            | string    | Product claims (e.g., Nutrient, Structure/Function, All Other); stored as a string representation of a Python list.   |
| `dsld_target_groups`     | string    | Target demographic groups (e.g., Vegetarian, Adult, Gluten Free); stored as a string representation of a Python list. |
| `dsld_directions`        | string    | Usage directions for the supplement.                                                                                  |
| `dsld_warnings`          | string    | Warning text and safety information for the product.                                                                  |


---

## Notes

1. **List-like columns** (`dsld_ingredient_names`, `dsld_ingredient_groups`, `dsld_claims`, `dsld_target_groups`) are stored as string representations of Python lists. Parsing with `ast.literal_eval()` or similar may be needed for structured analysis.
2. **Nullable fields**: `dsld_servings_per_container` may contain empty values when not specified on the label.
3. **Timestamp**: The `timestamp` column uses Unix time in milliseconds; convert by dividing by 1000 for standard Unix epoch seconds.

