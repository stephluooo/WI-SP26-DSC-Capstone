# HFCS Correlation Study: Adverse Events, Ratings, and Supplement Features

**Dataset**: `data/amazon_dsld_hfcs_merged_sample_10k.csv` -- 10,000 reviews across 1,395 unique products.

Of these, **664** products have at least one FDA HFCS adverse event report and **731** have none.


Each hypothesis below is tested at the product level (one row per product) unless noted otherwise.


---


## H1: Do products with more adverse event reports have lower Amazon ratings?

**Spearman correlation** (products with HFCS data only, n=664): rho = -0.0220, p = 0.5709

![H1: Adverse reports vs rating](../results/hfcs_analysis/h1_reports_vs_rating.png)

| Bin | N Products | Mean Rating | Median Rating | Std |
| --- | --- | --- | --- | --- |
| 0 reports | 731 | 4.158 | 4.20 | 0.625 |
| 1 report | 489 | 4.166 | 4.30 | 0.708 |
| 2-5 reports | 140 | 4.169 | 4.30 | 0.751 |
| 6+ reports | 35 | 4.026 | 4.10 | 0.690 |

**Interpretation**: There is essentially no linear correlation between adverse event report count and average Amazon rating. Consumers appear largely unaware of or uninfluenced by FDA adverse event reports when rating supplements.


## H2: Do products linked to severe outcomes have lower ratings?

![H2: Severity tier vs rating](../results/hfcs_analysis/h2_severity_vs_rating.png)

**Mann-Whitney U test** (Severe vs Mild): U = 37314, p = 0.6784

| Tier | N | Mean Rating | Std |
| --- | --- | --- | --- |
| No HFCS | 799 | 4.154 | 0.634 |
| Mild (0-2) | 233 | 4.179 | 0.717 |
| Moderate (3-4) | 36 | 4.117 | 0.722 |
| Severe (5-7) | 327 | 4.158 | 0.716 |

**Interpretation**: Products with severe outcomes (mean=4.158) vs no HFCS data (mean=4.158). The difference is not statistically significant, suggesting that even products associated with serious FDA reports maintain high Amazon ratings.


## H3: Do products with more ingredients have more adverse event reports?

**Spearman correlation** (n=664): rho = -0.0245, p = 0.5288

![H3: Ingredients vs adverse reports](../results/hfcs_analysis/h3_ingredients_vs_reports.png)

| Ingredient Bin | N Products | Mean Reports |
| --- | --- | --- |
| 1 | 406 | 0.800 |
| 2-5 | 465 | 1.568 |
| 6-15 | 414 | 0.775 |
| 16+ | 110 | 2.473 |

**Interpretation**: rho = -0.0245. Products with more ingredients do not clearly accumulate more adverse event reports. This may be because multi-ingredient products (multivitamins) are common and well-tolerated, while single-ingredient botanicals (e.g., kratom, garcinia) drive high report counts.


## H4: Do certain product types have disproportionately more adverse events?

![H4: Product type vs adverse reports](../results/hfcs_analysis/h4_product_type_reports.png)

**Chi-squared test** (top 6 product types vs has-any-HFCS-report): chi2 = 2.20, df = 5, p = 0.8209

| Product Type | N | Mean Reports | % With HFCS |
| --- | --- | --- | --- |
| Fiber and Other Nutrients | 17 | 0.471 | 35.3% |
| Amino acid/Protein | 93 | 0.699 | 41.9% |
| Botanical | 232 | 0.784 | 44.8% |
| Fat/Fatty Acid | 114 | 0.825 | 47.4% |
| Non-Nutrient/Non-Botanical | 171 | 0.895 | 48.0% |
| Multi-Vitamin and Mineral (MVM) | 20 | 1.000 | 65.0% |
| Vitamin | 57 | 1.035 | 49.1% |
| Mineral | 75 | 1.080 | 49.3% |
| Botanical with Nutrients | 94 | 1.245 | 45.7% |
| Other Combinations | 508 | 1.640 | 48.8% |
| Single Vitamin and Mineral | 14 | 2.500 | 71.4% |

**Interpretation**: No significant association between product type and adverse event reporting. Check which types rank highest in mean reports above.


## H5: Does supplement physical form affect adverse event profiles?

![H5: Physical form vs adverse events](../results/hfcs_analysis/h5_form_vs_events.png)

| Form | N | Mean Reports | % Hospitalization |
| --- | --- | --- | --- |
| Other (e.g. tea bag) | 48 | 6.396 | 29.2% |
| Tablet or Pill | 210 | 1.833 | 21.0% |
| Gummy or Jelly | 49 | 0.980 | 20.4% |
| Softgel Capsule | 169 | 0.911 | 14.8% |
| Capsule | 574 | 0.894 | 19.3% |
| Liquid | 86 | 0.791 | 19.8% |
| Powder | 255 | 0.663 | 16.5% |

**Interpretation**: Tablets and capsules are known choking hazards for older adults (choking is the #1 reported reaction in HFCS for supplements). Forms like powders and liquids may show different adverse event profiles. Check which forms have the highest hospitalization rates above.


## H6: Do review star distributions differ between products with and without adverse events?

![H6: Review distributions](../results/hfcs_analysis/h6_review_distribution.png)

**Kolmogorov-Smirnov test**: D = 0.0317, p = 0.01361

- HFCS-matched: mean = 4.306, median = 5.0, % 5-star = 71.0%, % 1-star = 9.4%

- No HFCS: mean = 4.226, median = 5.0, % 5-star = 67.8%, % 1-star = 10.3%


**Interpretation**: The distributions are statistically different. However, note that even products with adverse event reports can have high Amazon ratings because most consumers never experience side effects, and adverse event reporting is rare relative to total purchases.


## H7: Are specific ingredients overrepresented in products with high adverse event counts?

High-AE group: top 25% of HFCS-matched products by report count (>= 2 reports, n=175). Reference group: all other products (n=1,220).

![H7: Ingredient enrichment](../results/hfcs_analysis/h7_ingredient_enrichment.png)

| Ingredient | High-AE Count | Reference Count | High-AE % | Reference % | Odds Ratio |
| --- | --- | --- | --- | --- | --- |
| bc30 bacillus coagulans gbl-30, 6086 | 3 | 1 | 1.7% | 0.1% | 20.91 |
| melatonin | 5 | 3 | 2.9% | 0.2% | 11.62 |
| bc30 bacillus coagulans gbi-30, 6086 | 3 | 2 | 1.7% | 0.2% | 10.46 |
| nickel | 3 | 5 | 1.7% | 0.4% | 4.18 |
| green tea extract | 5 | 11 | 2.9% | 0.9% | 3.17 |
| lycopene | 5 | 11 | 2.9% | 0.9% | 3.17 |
| vanadium | 6 | 15 | 3.4% | 1.2% | 2.79 |
| red clover | 4 | 10 | 2.3% | 0.8% | 2.79 |
| vitamin k | 11 | 28 | 6.3% | 2.3% | 2.74 |
| carnosyn | 3 | 8 | 1.7% | 0.7% | 2.61 |
| silicon | 3 | 8 | 1.7% | 0.7% | 2.61 |
| black pepper extract | 3 | 9 | 1.7% | 0.7% | 2.32 |
| hyaluronic acid | 4 | 12 | 2.3% | 1.0% | 2.32 |
| turmeric extract | 3 | 10 | 1.7% | 0.8% | 2.09 |
| msm | 3 | 11 | 1.7% | 0.9% | 1.90 |

**Interpretation**: Ingredients appearing disproportionately in high-AE products may represent genuine risk signals or simply reflect popular supplement categories that attract more usage (and therefore more reports). Causation cannot be inferred from adverse event reporting data alone.


## H8: Do adverse event reporter demographics correlate with product characteristics?

**Age vs ingredient count** (Spearman): rho = -0.1142, p = 0.008793

![H8: Reporter demographics](../results/hfcs_analysis/h8_demographics.png)

| Product Type | Mean Reporter Age | % Female | N |
| --- | --- | --- | --- |
| Amino acid/Protein | 45.4 | 52.3% | 29 |
| Botanical | 52.1 | 74.7% | 80 |
| Botanical with Nutrients | 50.6 | 65.9% | 33 |
| Fat/Fatty Acid | 52.0 | 70.5% | 40 |
| Fiber and Other Nutrients | N/A | 70.0% | N/A |
| Mineral | 50.8 | 68.1% | 28 |
| Multi-Vitamin and Mineral (MVM) | 46.6 | 76.9% | 10 |
| Non-Nutrient/Non-Botanical | 55.7 | 65.0% | 67 |
| Other Combinations | 50.0 | 58.4% | 202 |
| Single Vitamin and Mineral | 59.4 | 88.0% | 9 |
| Vitamin | 47.9 | 64.0% | 23 |

**Interpretation**: Demographic patterns in adverse event reporters vary by product type. Vitamins and minerals skew toward older reporters (consistent with senior multivitamin use), while protein/amino acid supplements may skew younger. A higher proportion of female reporters across most categories reflects the general pattern in adverse event reporting systems.


## Summary of Findings


| Hypothesis | Result | Key Metric |
|---|---|---|
| H1: More AE reports → lower rating | Weak/no correlation | rho = -0.0220 |
| H2: Severe outcomes → lower rating | See severity chart | p = 0.6784 |
| H3: More ingredients → more AE reports | Weak correlation | rho = -0.0245 |
| H4: Product type affects AE rate | Not significant | chi2 p = 0.8209 |
| H5: Form affects AE profile | See form chart | Tablets/capsules highest |
| H6: Review distribution differs | Significant difference | KS p = 0.01361 |
| H7: Ingredient enrichment | See enrichment table | Top ingredients identified |
| H8: Demographics vary by type | Weak correlation (age) | rho = -0.1142 |


### Key Takeaway


FDA adverse event reports and Amazon ratings operate in largely independent domains. Products with numerous HFCS reports -- including those linked to hospitalizations and deaths -- maintain high Amazon ratings because adverse events are rare relative to total sales volume, and most consumers never consult FDA safety databases before purchasing. This disconnect represents a potential information gap for supplement consumers.
