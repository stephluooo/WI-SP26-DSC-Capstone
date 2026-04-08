# EDA Chart Explanations — `karina_eda.ipynb`

All charts use a **10% random sample** (`SAMPLE_FRAC = 0.10`, `random_state = 42`) of the full `data/amazon_dsld_merged.csv` dataset (~218K rows → ~21.9K rows) to keep computation fast while still being representative.

---

## Section 1 — Basic Data Overview

### Chart 1: Missing Values by Column
- **Type:** Horizontal bar chart (`ax.barh`)
- **Columns used:** All 32 columns (checks `.isnull().mean()` for each)
- **What it does:** For every column that has at least one missing value, it plots the fraction of rows that are null. Bars are sorted from most missing to least missing.
- **Why it was made:** Before doing any analysis we need to know which columns are incomplete. Columns like `dsld_upc`, `dsld_servings_per_container`, `dsld_warnings`, and `dsld_directions` have notable missingness, which affects whether we can reliably use them in downstream analysis. This chart makes it easy to spot data-quality issues at a glance.

---

## Section 2 — Review Rating Distribution

### Chart 2a: Distribution of Review Ratings (Bar Chart)
- **Type:** Vertical bar chart (`axes[0].bar`) with annotated counts
- **Column used:** `review_rating` (float, 1.0–5.0)
- **What it does:** Counts the number of reviews at each star level (1–5) and displays the totals above each bar.
- **Why it was made:** Understanding the overall sentiment skew is the most fundamental piece of EDA for a reviews dataset. Dietary supplement reviews tend to skew positive (lots of 5-star ratings), and this chart quantifies exactly how imbalanced the rating distribution is — important for any future modeling or sentiment analysis work.

### Chart 2b: Review Rating Share (Pie Chart)
- **Type:** Pie chart (`axes[1].pie`) with percentage labels
- **Column used:** `review_rating`
- **What it does:** Shows the same rating counts as proportions of the whole, making it easy to read percentages (e.g., "X% of all reviews are 5-star").
- **Why it was made:** Complements the bar chart by emphasizing relative share rather than absolute count, giving a quick sense of how dominant the top rating category is.

---

## Section 3 — Top Brands & Amazon Stores

### Chart 3a: Top 15 DSLD Brands by Review Count
- **Type:** Horizontal bar chart (`axes[0].barh`)
- **Column used:** `dsld_brand_name`
- **What it does:** Counts reviews per DSLD brand name and shows the 15 brands with the most reviews.
- **Why it was made:** Identifies which supplement brands dominate the dataset. A small number of brands may account for a large share of reviews, which is important context — analyses could be skewed by a few heavily-reviewed brands. This also helps identify major market players in the dietary supplement space.

### Chart 3b: Top 15 Amazon Stores by Review Count
- **Type:** Horizontal bar chart (`axes[1].barh`)
- **Column used:** `amazon_store`
- **What it does:** Counts reviews per Amazon store (seller) and shows the top 15.
- **Why it was made:** The Amazon store name represents the seller/storefront, which is different from the DSLD brand. Comparing the two charts reveals whether the top sellers align with the top DSLD brands or if there are resellers/multi-brand stores dominating the review volume.

---

## Section 4 — Product Form & Product Type

### Chart 4a: Top 10 Product Forms
- **Type:** Horizontal bar chart (`axes[0].barh`)
- **Column used:** `dsld_form` (e.g., Capsule, Tablet, Softgel, Liquid, Powder, Gummy)
- **What it does:** Shows the 10 most common physical forms that supplements come in, ranked by review count.
- **Why it was made:** Product form can influence consumer experience and satisfaction (e.g., gummies may get higher ratings than large tablets). Understanding the form distribution is a prerequisite to investigating whether product form correlates with review ratings.

### Chart 4b: Product Type Distribution (Pie Chart)
- **Type:** Pie chart (`axes[1].pie`)
- **Column used:** `dsld_product_type` (e.g., Vitamin, Mineral, Botanical, Amino Acid)
- **What it does:** Shows the proportion of reviews by DSLD product type category.
- **Why it was made:** Reveals what kinds of supplements are most represented in the dataset. If one category (e.g., Vitamin) dominates, it affects the generalizability of any findings to the broader supplement market.

### Chart 4c: Average Rating by Product Form (min 50 reviews)
- **Type:** Horizontal bar chart (`ax.barh`) with sample-size annotations
- **Columns used:** `dsld_form` grouped against `review_rating` (aggregated with `.agg(["mean", "count"])`)
- **What it does:** For each product form with at least 50 reviews, shows the average star rating. Each bar is annotated with the sample size (n=).
- **Why it was made:** Directly tests whether certain product forms receive systematically higher or lower ratings. The minimum threshold of 50 reviews filters out forms with too little data to be meaningful. This is a key finding for understanding consumer satisfaction patterns.

---

## Section 5 — Ingredient Analysis

### Chart 5a: Top 20 Most Common Ingredients
- **Type:** Horizontal bar chart (`ax.barh`)
- **Column used:** `dsld_ingredient_names` (string representation of a Python list, parsed with `ast.literal_eval`)
- **What it does:** Parses the ingredient list for every product, flattens all ingredients into a single counter, and displays the 20 most frequently occurring ingredients across all reviews.
- **Why it was made:** Identifies the most popular supplement ingredients in the marketplace (e.g., Vitamin D, Vitamin C, Zinc). This is valuable for understanding which ingredients consumers are most commonly purchasing and could be cross-referenced with ratings to see if products containing certain ingredients are rated higher or lower.

### Chart 5b: Distribution of Ingredient Count per Product
- **Type:** Histogram (`ax.hist`) with a median line
- **Column used:** `dsld_ingredient_count` (integer)
- **What it does:** Plots how many ingredients each product contains, with a red dashed line marking the median.
- **Why it was made:** Shows whether most supplements are single-ingredient (e.g., just Vitamin D) or multi-ingredient blends. This distribution matters because product complexity could relate to consumer confusion, label accuracy, and review sentiment.

---

## Section 6 — Verified Purchase & Helpful Votes

### Chart 6a: Review Rating by Verified Purchase (Box Plot)
- **Type:** Box plot (`sns.boxplot`)
- **Columns used:** `verified_purchase` (boolean, x-axis) vs. `review_rating` (float, y-axis)
- **What it does:** Compares the distribution of star ratings between verified purchasers and non-verified reviewers, showing median, quartiles, and outliers.
- **Why it was made:** Verified-purchase status is a key credibility signal. If non-verified reviews skew significantly higher or lower, it may indicate fake/incentivized reviews. This chart tests that assumption directly.

### Chart 6b: Distribution of Helpful Votes (non-zero, log scale)
- **Type:** Histogram (`axes[1].hist`) with log-scaled y-axis
- **Column used:** `helpful_vote` (integer, filtered to > 0)
- **What it does:** Shows how helpful votes are distributed among reviews that received at least one vote. The y-axis is log-scaled because the distribution is heavily right-skewed.
- **Why it was made:** Most reviews get zero helpful votes, and of those that do, the distribution is extremely long-tailed. This chart reveals how concentrated "helpfulness" is — a small number of reviews get a disproportionate share of votes, which is useful for identifying high-impact reviews.

### Chart 6c: Average Rating — Verified vs Non-Verified by Product Type
- **Type:** Grouped bar chart (`sns.barplot` with `hue="verified_purchase"`)
- **Columns used:** `dsld_product_type` (x-axis), `review_rating` (y-axis, averaged), `verified_purchase` (hue)
- **What it does:** For each product type, shows the average rating split by verified vs. non-verified purchase.
- **Why it was made:** Extends the box plot analysis by breaking it down per product category. If the verified/non-verified rating gap varies by product type, it could indicate that certain supplement categories are more susceptible to inauthentic reviews.

---

## Section 7 — Reviews Over Time

### Chart 7a: Number of Reviews per Month (Time Series)
- **Type:** Line chart with shaded fill (`axes[0].plot` + `fill_between`)
- **Column used:** `timestamp` (Unix ms, converted to datetime → grouped by month via `dt.to_period("M")`)
- **What it does:** Plots the total number of reviews submitted each month over the entire time span of the dataset.
- **Why it was made:** Reveals temporal trends — whether review volume is growing, seasonal, or has spikes (e.g., around Prime Day or holidays). Understanding review volume over time is essential context for any time-dependent analysis.

### Chart 7b: Average Review Rating per Month (Time Series)
- **Type:** Line chart (`axes[1].plot`)
- **Columns used:** `timestamp` (converted to monthly periods) grouped against `review_rating` (mean)
- **What it does:** Shows how the average star rating has changed month-over-month.
- **Why it was made:** Tests whether there is rating inflation or deflation over time. If average ratings are trending upward, it could indicate changing consumer behavior, stricter Amazon review policies, or survivorship bias (poorly-rated products get delisted).

---

## Section 8 — Target Groups & Claims

### Chart 8a: Top 15 Target Groups
- **Type:** Horizontal bar chart (`axes[0].barh`)
- **Column used:** `dsld_target_groups` (string representation of a list, parsed with `ast.literal_eval`)
- **What it does:** Parses the target-group labels for each product (e.g., "Adult (18–50 Years)", "Vegetarian", "Gluten Free"), flattens them, and counts the 15 most common.
- **Why it was made:** Shows who these supplements are marketed to. Knowing the dominant target demographics helps contextualize the review data — e.g., if most products target adults 18–50, the reviews likely reflect that demographic's preferences and language.

### Chart 8b: Top 10 Product Claims
- **Type:** Horizontal bar chart (`axes[1].barh`)
- **Column used:** `dsld_claims` (string representation of a list, parsed with `ast.literal_eval`)
- **What it does:** Parses the health/regulatory claims on each product's label (e.g., "Structure/Function", "Nutrient", "All Other") and shows the 10 most common.
- **Why it was made:** The type of claim a supplement makes is regulated by the FDA and directly impacts consumer expectations. Understanding which claim types dominate the dataset sets the stage for analyzing whether certain claim types correlate with higher or lower ratings.

---

## Section 9 — Correlation Heatmap & Numeric Summary

### Chart 9a: Correlation Heatmap of Numeric Features
- **Type:** Lower-triangle heatmap (`sns.heatmap` with `np.triu` mask)
- **Columns used:** `review_rating`, `helpful_vote`, `average_rating`, `rating_number`, `match_score`, `dsld_servings_per_container`, `dsld_ingredient_count`
- **What it does:** Computes pairwise Pearson correlation coefficients for all numeric columns and displays them as an annotated color-coded matrix.
- **Why it was made:** Quickly identifies which numeric features move together. For example, `review_rating` and `average_rating` should correlate positively; `rating_number` and `average_rating` may show an interesting relationship (products with more reviews might regress toward the mean). The `match_score` column (how well Amazon products matched to DSLD records) is also included to check whether match quality relates to any review metrics.

### Chart 9b: Average Rating vs. Number of Ratings (Scatter Plot)
- **Type:** Scatter plot (`ax.scatter`) with log-scaled x-axis
- **Columns used:** `rating_number` (x-axis, log scale), `average_rating` (y-axis)
- **What it does:** Plots each product's aggregate Amazon rating against how many ratings it has received. A random sub-sample of up to 5,000 points is used for clarity.
- **Why it was made:** Visualizes the "wisdom of crowds" effect — products with very few ratings can have extreme averages (1.0 or 5.0), while products with thousands of ratings tend to cluster around 4.0–4.5. This funnel shape is a classic pattern in review data and is important to understand before drawing conclusions from average ratings.
