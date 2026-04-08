# XGBoost Analysis: Do Label Features Predict Amazon Ratings?

## The Question

> Can what's on a supplement label — ingredients, ingredient count, brand name, product form, product type — predict how well it's rated on Amazon?

Liyun tested this with a CNN. Here we test the same question with XGBoost (gradient-boosted decision trees) to see if a fundamentally different model reaches the same conclusion.

---

## Setup

| Detail | Value |
|---|---|
| Total reviews | 218,678 |
| Unique products (after dedup) | 24,017 |
| Train / Test split | 19,213 / 4,804 (80/20) |
| Features used | 54 (all from label data only) |
| Target variable | `average_rating` per product (mean 3.96, std 0.80) |

**Features breakdown:**

| Feature group | Count | What it captures |
|---|---|---|
| Ingredient count | 1 | How many active ingredients the product has |
| Brand (encoded) | 1 | Which of the 1,684 brands makes it |
| Product form (encoded) | 1 | Capsule, powder, liquid, etc. (10 forms) |
| Product type (encoded) | 1 | Vitamin, mineral, botanical, etc. (11 types) |
| Top-50 ingredient indicators | 50 | Binary flags for the 50 most common ingredients (covering 67% of all mentions) |

---

## Bottom Line

**Label features explain ~3% of the variance in product ratings. That is essentially nothing.**

| Model | MAE | R² | Takeaway |
|---|---|---|---|
| **Naive baseline** (always guess 3.96) | 0.603 | 0.000 | The bar to beat |
| **CNN** (Liyun) | 0.504 | -0.025 | No better than guessing |
| **XGBoost** (this analysis) | 0.586 | 0.029 | Barely better than guessing |

- **MAE** = mean absolute error (average number of stars the prediction is off). XGBoost's 0.586 vs the naive 0.603 is a **2.8% improvement** — trivial.
- **R²** = proportion of variance explained. 0.029 means the model captures **less than 3%** of what makes one product rated higher than another. An R² of 0 means "no better than the mean"; a perfect model gets 1.0.
- The 5-fold cross-validation confirms this is stable, not a fluke: **CV R² = 0.034 +/- 0.007**.

---

## What Features Mattered Most (and Why It Doesn't Change the Story)

Even within a weak model, we can see which label features XGBoost leaned on the most:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Organic food blend (ingredient) | 0.098 |
| 2 | Enzogenol (ingredient) | 0.041 |
| 3 | Proprietary blend (ingredient) | 0.039 |
| 4 | Calories (ingredient) | 0.038 |
| 5 | Biotin (ingredient) | 0.037 |
| 6 | Boron (ingredient) | 0.025 |
| 7 | Product type | 0.023 |
| 8 | Hyaluronic acid (ingredient) | 0.023 |
| 9 | Copper (ingredient) | 0.023 |
| 10 | Thiamine (ingredient) | 0.021 |

Specific ingredients dominate the top spots, while brand (#20, importance 0.017) and ingredient count (#14, importance 0.019) contribute less. But even the "best" feature only has 0.098 importance inside a model that explains almost nothing overall — so none of these are meaningful predictors.

---

## Ablation: Testing Each Feature Group Alone

We trained separate models on each feature group to isolate their contributions:

| Feature group | CV MAE | CV R² |
|---|---|---|
| All label features combined | 0.590 | 0.033 |
| Top-50 ingredients (multi-hot) | 0.595 | 0.021 |
| Brand only | 0.597 | 0.017 |
| Brand + ingredient count | 0.597 | 0.017 |
| Ingredient count only | 0.603 | 0.005 |
| Product form only | 0.604 | 0.004 |
| Product type only | 0.604 | 0.005 |
| **Naive baseline (predict mean)** | **0.603** | **0.000** |

None of them move the needle. The best group (all features combined) explains 3.3% of variance. Brand alone gets 1.7%. Ingredient count, form, and type are indistinguishable from guessing the mean.

---

## Why This Matters

Two completely different model architectures — a neural network (CNN) and a tree ensemble (XGBoost), trained on different dataset sizes (1,395 vs 24,017 products) — agree:

**What's on the label does not determine how consumers rate a product.**

This makes intuitive sense. Ratings are shaped by things labels can't capture:
- Whether the product actually works for that person
- Taste, smell, pill size
- Shipping and packaging experience
- Price relative to expectations
- Placebo effects and brand loyalty beyond the name itself

---

## What This Means for Next Steps

The hypothesis (label features → average rating) is **rejected**. This is a useful negative result because it tells us where *not* to look. The group's updated direction — investigating the relationship between label features and **reported side effects** from the HFCS database — is a stronger hypothesis, since ingredient composition has a more direct causal link to adverse reactions than to subjective star ratings.
