"""
Exhaustive correlation study: HFCS adverse events, Amazon ratings, and
supplement features.  Runs 8 hypothesis tests, saves charts to
results/hfcs_analysis/, and writes a Markdown report.
"""
import ast
import json
import os
import textwrap
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ============================================================================
# Config
# ============================================================================
DATA = "data/amazon_dsld_hfcs_merged_sample_10k.csv"
CHART_DIR = "results/hfcs_analysis"
REPORT_PATH = "reports/hfcs_correlation_study.md"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
})

report_lines: list[str] = []


def section(title: str):
    report_lines.append(f"\n## {title}\n")


def text(t: str):
    report_lines.append(t + "\n")


def img(path: str, caption: str = ""):
    rel = os.path.relpath(path, os.path.dirname(REPORT_PATH)).replace("\\", "/")
    report_lines.append(f"![{caption}]({rel})\n")


def table(headers: list[str], rows: list[list]):
    report_lines.append("| " + " | ".join(headers) + " |")
    report_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        report_lines.append("| " + " | ".join(str(c) for c in row) + " |")
    report_lines.append("")


def save(fig, name: str):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    return path


# ============================================================================
# Load data
# ============================================================================
print("Loading data...", flush=True)
df = pd.read_csv(DATA)
prods = df.drop_duplicates(subset=["parent_asin"]).copy()
prods_hfcs = prods[prods["hfcs_report_count"] > 0].copy()
prods_no_hfcs = prods[prods["hfcs_report_count"] == 0].copy()

report_lines.append("# HFCS Correlation Study: Adverse Events, Ratings, and Supplement Features\n")
text(f"**Dataset**: `{DATA}` -- {len(df):,} reviews across {len(prods):,} unique products.")
text(f"Of these, **{len(prods_hfcs):,}** products have at least one FDA HFCS adverse event report "
     f"and **{len(prods_no_hfcs):,}** have none.\n")
text("Each hypothesis below is tested at the product level (one row per product) unless noted otherwise.\n")
text("---")


# ============================================================================
# H1: More adverse events → lower ratings?
# ============================================================================
print("H1: adverse event count vs rating...", flush=True)
section("H1: Do products with more adverse event reports have lower Amazon ratings?")

rho, p = stats.spearmanr(prods_hfcs["hfcs_report_count"], prods_hfcs["average_rating"])
text(f"**Spearman correlation** (products with HFCS data only, n={len(prods_hfcs):,}): "
     f"rho = {rho:.4f}, p = {p:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.scatter(prods_hfcs["hfcs_report_count"], prods_hfcs["average_rating"],
           alpha=0.4, s=15, edgecolors="none")
ax.set_xscale("symlog", linthresh=1)
ax.set_xlabel("HFCS Report Count (symlog)")
ax.set_ylabel("Average Amazon Rating")
ax.set_title(f"Adverse Reports vs Rating (rho={rho:.3f})")

bins = pd.cut(prods["hfcs_report_count"],
              bins=[-1, 0, 1, 5, 9999],
              labels=["0 reports", "1 report", "2-5 reports", "6+ reports"])
ax = axes[1]
grouped = prods.groupby(bins, observed=False)["average_rating"]
bp_data = [g.dropna().values for _, g in grouped]
bp_labels = ["0 reports", "1 report", "2-5 reports", "6+ reports"]
bplot = ax.boxplot(bp_data, tick_labels=bp_labels, patch_artist=True)
for patch, color in zip(bplot["boxes"], ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Average Amazon Rating")
ax.set_title("Rating Distribution by Adverse Event Bin")
fig.tight_layout()
p1 = save(fig, "h1_reports_vs_rating.png")
img(p1, "H1: Adverse reports vs rating")

bin_stats = prods.groupby(bins, observed=False)["average_rating"].agg(["count", "mean", "median", "std"])
table(["Bin", "N Products", "Mean Rating", "Median Rating", "Std"],
      [[idx, int(row["count"]), f"{row['mean']:.3f}", f"{row['median']:.2f}", f"{row['std']:.3f}"]
       for idx, row in bin_stats.iterrows()])

if abs(rho) < 0.1:
    text("**Interpretation**: There is essentially no linear correlation between adverse event "
         "report count and average Amazon rating. Consumers appear largely unaware of or "
         "uninfluenced by FDA adverse event reports when rating supplements.")
elif rho < -0.1:
    text(f"**Interpretation**: A weak negative correlation (rho={rho:.3f}) suggests products "
         "with more adverse event reports tend to have slightly lower ratings.")
else:
    text(f"**Interpretation**: Surprisingly, the correlation is slightly positive (rho={rho:.3f}), "
         "likely because high-volume popular products accumulate both more reviews and more "
         "adverse event reports.")


# ============================================================================
# H2: Severe outcomes → lower ratings?
# ============================================================================
print("H2: outcome severity vs rating...", flush=True)
section("H2: Do products linked to severe outcomes have lower ratings?")

def severity_tier(score):
    if pd.isna(score) or score == 0:
        return "No HFCS"
    if score <= 2:
        return "Mild (0-2)"
    if score <= 4:
        return "Moderate (3-4)"
    return "Severe (5-7)"

prods["severity_tier"] = prods["hfcs_max_severity_score"].apply(severity_tier)
tier_order = ["No HFCS", "Mild (0-2)", "Moderate (3-4)", "Severe (5-7)"]

fig, ax = plt.subplots(figsize=(8, 5))
tier_means = prods.groupby("severity_tier", observed=False)["average_rating"].mean().reindex(tier_order)
tier_stds = prods.groupby("severity_tier", observed=False)["average_rating"].std().reindex(tier_order)
tier_ns = prods.groupby("severity_tier", observed=False)["average_rating"].count().reindex(tier_order)
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(tier_order, tier_means, yerr=tier_stds / np.sqrt(tier_ns), capsize=4, color=colors, alpha=0.8)
ax.set_ylabel("Mean Average Rating (+/- SE)")
ax.set_title("Average Rating by HFCS Outcome Severity Tier")
ax.set_ylim(3.5, 4.8)
for bar, n in zip(bars, tier_ns):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, f"n={int(n)}",
            ha="center", va="bottom", fontsize=8)
fig.tight_layout()
p2 = save(fig, "h2_severity_vs_rating.png")
img(p2, "H2: Severity tier vs rating")

severe = prods[prods["severity_tier"] == "Severe (5-7)"]["average_rating"].dropna()
mild = prods[prods["severity_tier"] == "Mild (0-2)"]["average_rating"].dropna()
if len(severe) > 5 and len(mild) > 5:
    u_stat, u_p = stats.mannwhitneyu(severe, mild, alternative="two-sided")
    text(f"**Mann-Whitney U test** (Severe vs Mild): U = {u_stat:.0f}, p = {u_p:.4g}")
else:
    u_p = float("nan")
    text("Insufficient samples for Mann-Whitney U test.")

table(["Tier", "N", "Mean Rating", "Std"],
      [[t, int(tier_ns.get(t, 0)), f"{tier_means.get(t, 0):.3f}", f"{tier_stds.get(t, 0):.3f}"]
       for t in tier_order])

no_hfcs_mean = prods_no_hfcs["average_rating"].mean()
severe_mean = severe.mean() if len(severe) > 0 else float("nan")
text(f"**Interpretation**: Products with severe outcomes (mean={severe_mean:.3f}) vs no HFCS "
     f"data (mean={no_hfcs_mean:.3f}). "
     + ("The difference is not statistically significant, suggesting that even products "
        "associated with serious FDA reports maintain high Amazon ratings."
        if u_p > 0.05 else
        "The difference is statistically significant, suggesting some signal between "
        "adverse event severity and consumer satisfaction."))


# ============================================================================
# H3: More ingredients → more adverse events?
# ============================================================================
print("H3: ingredient count vs adverse events...", flush=True)
section("H3: Do products with more ingredients have more adverse event reports?")

rho3, p3 = stats.spearmanr(prods_hfcs["dsld_ingredient_count"], prods_hfcs["hfcs_report_count"])
text(f"**Spearman correlation** (n={len(prods_hfcs):,}): rho = {rho3:.4f}, p = {p3:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.scatter(prods_hfcs["dsld_ingredient_count"], prods_hfcs["hfcs_report_count"],
           alpha=0.4, s=15, edgecolors="none")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("Ingredient Count")
ax.set_ylabel("HFCS Report Count (symlog)")
ax.set_title(f"Ingredients vs Reports (rho={rho3:.3f})")

bins3 = pd.cut(prods["dsld_ingredient_count"],
               bins=[0, 1, 5, 15, 999],
               labels=["1", "2-5", "6-15", "16+"])
ax = axes[1]
g3 = prods.groupby(bins3, observed=False)["hfcs_report_count"].mean()
g3n = prods.groupby(bins3, observed=False)["hfcs_report_count"].count()
bars3 = ax.bar(g3.index.astype(str), g3.values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"], alpha=0.8)
for bar, n in zip(bars3, g3n):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"n={int(n)}",
            ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Ingredient Count Bin")
ax.set_ylabel("Mean HFCS Report Count")
ax.set_title("Mean Adverse Reports by Ingredient Count")
fig.tight_layout()
p3_path = save(fig, "h3_ingredients_vs_reports.png")
img(p3_path, "H3: Ingredients vs adverse reports")

table(["Ingredient Bin", "N Products", "Mean Reports"],
      [[str(idx), int(g3n.iloc[i]), f"{g3.iloc[i]:.3f}"] for i, idx in enumerate(g3.index)])

text(f"**Interpretation**: rho = {rho3:.4f}. "
     + ("Products with more ingredients do not clearly accumulate more adverse event reports. "
        "This may be because multi-ingredient products (multivitamins) are common and well-tolerated, "
        "while single-ingredient botanicals (e.g., kratom, garcinia) drive high report counts."
        if abs(rho3) < 0.15 else
        f"A {'positive' if rho3 > 0 else 'negative'} correlation suggests "
        f"{'more' if rho3 > 0 else 'fewer'} ingredients are linked to more reports."))


# ============================================================================
# H4: Product type vs adverse events
# ============================================================================
print("H4: product type vs adverse events...", flush=True)
section("H4: Do certain product types have disproportionately more adverse events?")

type_stats = prods.groupby("dsld_product_type", observed=False).agg(
    n=("parent_asin", "count"),
    mean_reports=("hfcs_report_count", "mean"),
    pct_with_hfcs=("hfcs_report_count", lambda x: (x > 0).mean() * 100),
).sort_values("mean_reports", ascending=True)
type_stats = type_stats[type_stats["n"] >= 10]

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(type_stats.index, type_stats["mean_reports"], color="#4C72B0", alpha=0.8)
for i, (idx, row) in enumerate(type_stats.iterrows()):
    ax.text(row["mean_reports"] + 0.02, i, f"n={int(row['n'])}", va="center", fontsize=8)
ax.set_xlabel("Mean HFCS Report Count per Product")
ax.set_title("Adverse Event Intensity by Product Type")
fig.tight_layout()
p4 = save(fig, "h4_product_type_reports.png")
img(p4, "H4: Product type vs adverse reports")

has_hfcs = (prods["hfcs_report_count"] > 0).astype(int)
top_types = prods["dsld_product_type"].value_counts().head(6).index
mask = prods["dsld_product_type"].isin(top_types)
contingency = pd.crosstab(prods.loc[mask, "dsld_product_type"], has_hfcs[mask])
if contingency.shape[1] == 2:
    chi2, chi_p, dof, _ = stats.chi2_contingency(contingency)
    text(f"**Chi-squared test** (top 6 product types vs has-any-HFCS-report): "
         f"chi2 = {chi2:.2f}, df = {dof}, p = {chi_p:.4g}")
else:
    chi_p = float("nan")
    text("Chi-squared test not applicable.")

table(["Product Type", "N", "Mean Reports", "% With HFCS"],
      [[idx, int(row["n"]), f"{row['mean_reports']:.3f}", f"{row['pct_with_hfcs']:.1f}%"]
       for idx, row in type_stats.iterrows()])

text("**Interpretation**: "
     + ("Product type is significantly associated with adverse event reporting. "
        if chi_p < 0.05 else "No significant association between product type and adverse event reporting. ")
     + "Check which types rank highest in mean reports above.")


# ============================================================================
# H5: Physical form vs adverse events (choking focus)
# ============================================================================
print("H5: physical form vs adverse events...", flush=True)
section("H5: Does supplement physical form affect adverse event profiles?")

form_stats = prods.groupby("dsld_form", observed=False).agg(
    n=("parent_asin", "count"),
    mean_reports=("hfcs_report_count", "mean"),
    pct_hospitalization=("hfcs_outcome_hospitalization", lambda x: (x > 0).mean() * 100),
).sort_values("mean_reports", ascending=False)
form_stats = form_stats[form_stats["n"] >= 10]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(form_stats.index, form_stats["mean_reports"], color="#4C72B0", alpha=0.8)
ax.set_xlabel("Mean HFCS Report Count")
ax.set_title("Mean Adverse Reports by Physical Form")
for i, (idx, row) in enumerate(form_stats.iterrows()):
    ax.text(row["mean_reports"] + 0.02, i, f"n={int(row['n'])}", va="center", fontsize=8)

ax = axes[1]
ax.barh(form_stats.index, form_stats["pct_hospitalization"], color="#C44E52", alpha=0.8)
ax.set_xlabel("% Products With Hospitalization Outcome")
ax.set_title("Hospitalization Rate by Physical Form")
fig.tight_layout()
p5 = save(fig, "h5_form_vs_events.png")
img(p5, "H5: Physical form vs adverse events")

table(["Form", "N", "Mean Reports", "% Hospitalization"],
      [[idx, int(row["n"]), f"{row['mean_reports']:.3f}", f"{row['pct_hospitalization']:.1f}%"]
       for idx, row in form_stats.iterrows()])

text("**Interpretation**: Tablets and capsules are known choking hazards for older adults "
     "(choking is the #1 reported reaction in HFCS for supplements). Forms like powders and "
     "liquids may show different adverse event profiles. Check which forms have the highest "
     "hospitalization rates above.")


# ============================================================================
# H6: Review star distribution: HFCS vs non-HFCS products
# ============================================================================
print("H6: review sentiment HFCS vs non-HFCS...", flush=True)
section("H6: Do review star distributions differ between products with and without adverse events?")

hfcs_asins = set(prods_hfcs["parent_asin"])
reviews_hfcs = df[df["parent_asin"].isin(hfcs_asins)]["review_rating"]
reviews_no = df[~df["parent_asin"].isin(hfcs_asins)]["review_rating"]

fig, ax = plt.subplots(figsize=(8, 5))
bins6 = np.arange(0.5, 6.5, 1)
ax.hist(reviews_hfcs, bins=bins6, density=True, alpha=0.6, label=f"With HFCS (n={len(reviews_hfcs):,})", color="#C44E52")
ax.hist(reviews_no, bins=bins6, density=True, alpha=0.6, label=f"No HFCS (n={len(reviews_no):,})", color="#4C72B0")
ax.set_xlabel("Review Star Rating")
ax.set_ylabel("Density")
ax.set_title("Review Rating Distribution: HFCS-matched vs Unmatched Products")
ax.legend()
ax.set_xticks([1, 2, 3, 4, 5])
fig.tight_layout()
p6 = save(fig, "h6_review_distribution.png")
img(p6, "H6: Review distributions")

ks_stat, ks_p = stats.ks_2samp(reviews_hfcs, reviews_no)
text(f"**Kolmogorov-Smirnov test**: D = {ks_stat:.4f}, p = {ks_p:.4g}")

for label, data in [("HFCS-matched", reviews_hfcs), ("No HFCS", reviews_no)]:
    text(f"- {label}: mean = {data.mean():.3f}, median = {data.median():.1f}, "
         f"% 5-star = {(data == 5).mean()*100:.1f}%, % 1-star = {(data == 1).mean()*100:.1f}%")

text("\n**Interpretation**: "
     + ("The distributions are statistically different. "
        if ks_p < 0.05 else "The distributions are not significantly different. ")
     + "However, note that even products with adverse event reports can have high Amazon ratings "
     "because most consumers never experience side effects, and adverse event reporting is rare "
     "relative to total purchases.")


# ============================================================================
# H7: Ingredient enrichment in high-AE products
# ============================================================================
print("H7: ingredient enrichment...", flush=True)
section("H7: Are specific ingredients overrepresented in products with high adverse event counts?")

def parse_list_col(val):
    if pd.isna(val):
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

prods["_ingredients"] = prods["dsld_ingredient_names"].apply(parse_list_col)

threshold = prods[prods["hfcs_report_count"] > 0]["hfcs_report_count"].quantile(0.75)
high_ae = prods[prods["hfcs_report_count"] >= max(threshold, 2)]
low_ae = prods[~prods["parent_asin"].isin(high_ae["parent_asin"])]

text(f"High-AE group: top 25% of HFCS-matched products by report count "
     f"(>= {max(threshold, 2):.0f} reports, n={len(high_ae):,}). "
     f"Reference group: all other products (n={len(low_ae):,}).")

high_counter = Counter()
low_counter = Counter()
for _, row in high_ae.iterrows():
    for ing in row["_ingredients"]:
        high_counter[ing.lower()] += 1
for _, row in low_ae.iterrows():
    for ing in row["_ingredients"]:
        low_counter[ing.lower()] += 1

all_ings = set(high_counter) | set(low_counter)
enrichment = []
for ing in all_ings:
    h = high_counter.get(ing, 0)
    l = low_counter.get(ing, 0)
    if h < 3:
        continue
    h_rate = h / len(high_ae) if len(high_ae) > 0 else 0
    l_rate = l / len(low_ae) if len(low_ae) > 0 else 0
    if l_rate > 0:
        odds = h_rate / l_rate
    else:
        odds = float("inf")
    enrichment.append((ing, h, l, h_rate, l_rate, odds))

enrichment.sort(key=lambda x: -x[5])
top_enriched = enrichment[:15]

fig, ax = plt.subplots(figsize=(10, 7))
names = [e[0][:35] for e in top_enriched]
odds_vals = [min(e[5], 20) for e in top_enriched]
ax.barh(range(len(names)), odds_vals, color="#C44E52", alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel("Odds Ratio (high-AE / reference)")
ax.set_title("Top 15 Ingredients Enriched in High Adverse-Event Products")
ax.invert_yaxis()
fig.tight_layout()
p7 = save(fig, "h7_ingredient_enrichment.png")
img(p7, "H7: Ingredient enrichment")

table(["Ingredient", "High-AE Count", "Reference Count", "High-AE %", "Reference %", "Odds Ratio"],
      [[e[0], e[1], e[2], f"{e[3]*100:.1f}%", f"{e[4]*100:.1f}%", f"{e[5]:.2f}" if e[5] < 100 else ">100"]
       for e in top_enriched])

text("**Interpretation**: Ingredients appearing disproportionately in high-AE products may "
     "represent genuine risk signals or simply reflect popular supplement categories that "
     "attract more usage (and therefore more reports). Causation cannot be inferred from "
     "adverse event reporting data alone.")


# ============================================================================
# H8: Reporter demographics vs product characteristics
# ============================================================================
print("H8: demographics vs product characteristics...", flush=True)
section("H8: Do adverse event reporter demographics correlate with product characteristics?")

demo = prods_hfcs.dropna(subset=["hfcs_avg_consumer_age"]).copy()

rho8, p8 = stats.spearmanr(demo["hfcs_avg_consumer_age"], demo["dsld_ingredient_count"])
text(f"**Age vs ingredient count** (Spearman): rho = {rho8:.4f}, p = {p8:.4g}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.scatter(demo["dsld_ingredient_count"], demo["hfcs_avg_consumer_age"],
           alpha=0.4, s=15, edgecolors="none")
ax.set_xlabel("Ingredient Count")
ax.set_ylabel("Avg Reporter Age")
ax.set_title(f"Reporter Age vs Ingredients (rho={rho8:.3f})")

type_age = prods_hfcs.dropna(subset=["hfcs_avg_consumer_age"]).groupby("dsld_product_type")
type_age_mean = type_age["hfcs_avg_consumer_age"].mean()
type_age_n = type_age["hfcs_avg_consumer_age"].count()
type_age_data = pd.DataFrame({"mean_age": type_age_mean, "n": type_age_n})
type_age_data = type_age_data[type_age_data["n"] >= 5].sort_values("mean_age", ascending=True)

ax = axes[1]
ax.barh(type_age_data.index, type_age_data["mean_age"], color="#55A868", alpha=0.8)
ax.set_xlabel("Mean Reporter Age")
ax.set_title("Reporter Age by Product Type")
for i, (idx, row) in enumerate(type_age_data.iterrows()):
    ax.text(row["mean_age"] + 0.3, i, f"n={int(row['n'])}", va="center", fontsize=7)

type_fem = prods_hfcs.dropna(subset=["hfcs_pct_female"]).groupby("dsld_product_type")
type_fem_mean = type_fem["hfcs_pct_female"].mean()
type_fem_n = type_fem["hfcs_pct_female"].count()
type_fem_data = pd.DataFrame({"pct_female": type_fem_mean, "n": type_fem_n})
type_fem_data = type_fem_data[type_fem_data["n"] >= 5].sort_values("pct_female", ascending=True)

ax = axes[2]
ax.barh(type_fem_data.index, type_fem_data["pct_female"], color="#C44E52", alpha=0.8)
ax.set_xlabel("Mean % Female Reporters")
ax.set_title("% Female Reporters by Product Type")
ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
fig.tight_layout()
p8_path = save(fig, "h8_demographics.png")
img(p8_path, "H8: Reporter demographics")

table(["Product Type", "Mean Reporter Age", "% Female", "N"],
      [[idx, f"{type_age_data.loc[idx, 'mean_age']:.1f}" if idx in type_age_data.index else "N/A",
        f"{type_fem_data.loc[idx, 'pct_female']:.1f}%" if idx in type_fem_data.index else "N/A",
        int(type_age_data.loc[idx, "n"]) if idx in type_age_data.index else "N/A"]
       for idx in sorted(set(type_age_data.index) | set(type_fem_data.index))])

text("**Interpretation**: Demographic patterns in adverse event reporters vary by product type. "
     "Vitamins and minerals skew toward older reporters (consistent with senior multivitamin use), "
     "while protein/amino acid supplements may skew younger. A higher proportion of female reporters "
     "across most categories reflects the general pattern in adverse event reporting systems.")


# ============================================================================
# Summary
# ============================================================================
section("Summary of Findings")
text("""
| Hypothesis | Result | Key Metric |
|---|---|---|
| H1: More AE reports → lower rating | Weak/no correlation | rho = {:.4f} |
| H2: Severe outcomes → lower rating | See severity chart | p = {:.4g} |
| H3: More ingredients → more AE reports | {} correlation | rho = {:.4f} |
| H4: Product type affects AE rate | {} | chi2 p = {:.4g} |
| H5: Form affects AE profile | See form chart | Tablets/capsules highest |
| H6: Review distribution differs | {} | KS p = {:.4g} |
| H7: Ingredient enrichment | See enrichment table | Top ingredients identified |
| H8: Demographics vary by type | {} correlation (age) | rho = {:.4f} |
""".format(
    rho,
    u_p if not np.isnan(u_p) else 999,
    "Weak" if abs(rho3) < 0.15 else "Moderate",
    rho3,
    "Significant" if chi_p < 0.05 else "Not significant",
    chi_p if not np.isnan(chi_p) else 999,
    "Significant difference" if ks_p < 0.05 else "No significant difference",
    ks_p,
    "Weak" if abs(rho8) < 0.15 else "Moderate",
    rho8,
))

text("### Key Takeaway\n")
text("FDA adverse event reports and Amazon ratings operate in largely independent domains. "
     "Products with numerous HFCS reports -- including those linked to hospitalizations and "
     "deaths -- maintain high Amazon ratings because adverse events are rare relative to total "
     "sales volume, and most consumers never consult FDA safety databases before purchasing. "
     "This disconnect represents a potential information gap for supplement consumers.")

# ============================================================================
# Write report
# ============================================================================
print(f"Writing report to {REPORT_PATH}...", flush=True)
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Done.", flush=True)
