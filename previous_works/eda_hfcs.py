"""
EDA of the amazon_dsld_hfcs_merged_sample_10k dataset.

Focus areas:
  1. Overview: adverse-event coverage, report/reaction distributions
  2. Most common adverse reactions across all products
  3. Ingredient → reaction associations (chi-square / lift)
  4. Ingredient-pair co-occurrence with severe outcomes
  5. Product-type and supplement-form breakdowns
  6. Severity analysis by ingredient group
  7. Clustering products by reaction profiles

Saves plots and a summary CSV to results/eda_hfcs/.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "amazon_dsld_hfcs_merged_sample_10k.csv"
OUT_DIR = ROOT / "results" / "eda_hfcs"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def safe_parse_list(val) -> list[str]:
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return val
    s = str(val).strip()
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
    except (ValueError, SyntaxError):
        pass
    return [s] if s else []


def safe_parse_dict(val) -> dict:
    if pd.isna(val) or val == "":
        return {}
    if isinstance(val, dict):
        return val
    s = str(val).strip()
    try:
        v = json.loads(s)
        if isinstance(v, dict):
            return v
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        v = ast.literal_eval(s)
        if isinstance(v, dict):
            return v
    except (ValueError, SyntaxError):
        pass
    return {}


# ---------------------------------------------------------------------------
# Load & prepare product-level frame
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (full_df, product_df). product_df has one row per parent_asin."""
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows, {df['parent_asin'].nunique()} unique products")

    prod = df.drop_duplicates(subset=["parent_asin"], keep="first").reset_index(drop=True)

    prod["has_hfcs"] = prod["hfcs_report_count"].fillna(0).astype(int) > 0
    prod["hfcs_report_count"] = prod["hfcs_report_count"].fillna(0).astype(int)
    prod["hfcs_reaction_count"] = prod["hfcs_reaction_count"].fillna(0).astype(int)
    prod["hfcs_max_severity_score"] = prod["hfcs_max_severity_score"].fillna(0).astype(int)

    prod["_ingredients"] = prod["dsld_ingredient_names"].apply(safe_parse_list)
    prod["_groups"] = prod["dsld_ingredient_groups"].apply(safe_parse_list)
    prod["_reactions_dict"] = prod["hfcs_all_reactions"].apply(safe_parse_dict)
    prod["dsld_ingredient_count"] = pd.to_numeric(prod["dsld_ingredient_count"], errors="coerce").fillna(0).astype(int)

    print(f"Products with HFCS data: {prod['has_hfcs'].sum()} / {len(prod)}")
    return df, prod


# ---------------------------------------------------------------------------
# 1. Overview stats
# ---------------------------------------------------------------------------

def plot_overview(prod: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Coverage pie
    counts = prod["has_hfcs"].value_counts()
    axes[0].pie(
        [counts.get(True, 0), counts.get(False, 0)],
        labels=["Has adverse events", "No adverse events"],
        autopct="%1.1f%%",
        colors=["#ef4444", "#93c5fd"],
        startangle=90,
    )
    axes[0].set_title("HFCS Coverage Among Products")

    # Report count distribution (products with HFCS only)
    hfcs = prod[prod["has_hfcs"]]
    axes[1].hist(hfcs["hfcs_report_count"], bins=30, color="#6366f1", edgecolor="white")
    axes[1].set_xlabel("Report Count")
    axes[1].set_ylabel("Number of Products")
    axes[1].set_title("Distribution of Adverse Event Reports")

    # Severity breakdown
    sev = prod[prod["has_hfcs"]]["hfcs_max_severity"].value_counts()
    sev = sev[sev.index.notna() & (sev.index != "") & (sev.index != "None")]
    if not sev.empty:
        sev.plot.barh(ax=axes[2], color="#f59e0b", edgecolor="white")
        axes[2].set_xlabel("Number of Products")
        axes[2].set_title("Max Severity per Product")
        axes[2].invert_yaxis()
    else:
        axes[2].text(0.5, 0.5, "No severity data", ha="center", va="center")
        axes[2].set_title("Max Severity per Product")

    fig.suptitle("HFCS Adverse Event Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_overview.png")
    plt.close(fig)
    print("  Saved 01_overview.png")


# ---------------------------------------------------------------------------
# 2. Top adverse reactions
# ---------------------------------------------------------------------------

def plot_top_reactions(prod: pd.DataFrame) -> Counter:
    reaction_counter: Counter = Counter()
    for rd in prod["_reactions_dict"]:
        for reaction, count in rd.items():
            reaction_counter[reaction.lower()] += int(count)

    top30 = reaction_counter.most_common(30)
    if not top30:
        print("  No reactions found — skipping reaction plot")
        return reaction_counter

    names, counts = zip(*top30)
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(names))
    ax.barh(y, counts, color="#6366f1", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Total Occurrences Across All Products")
    ax.set_title("Top 30 Reported Adverse Reactions", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_top_reactions.png")
    plt.close(fig)
    print("  Saved 02_top_reactions.png")
    return reaction_counter


# ---------------------------------------------------------------------------
# 3. Ingredient group → reaction association (chi-square + lift)
# ---------------------------------------------------------------------------

def compute_ingredient_reaction_associations(prod: pd.DataFrame, reaction_counter: Counter):
    hfcs = prod[prod["has_hfcs"]].copy()
    if len(hfcs) < 10:
        print("  Too few products with HFCS data for association analysis")
        return pd.DataFrame()

    top_reactions = [r for r, _ in reaction_counter.most_common(15)]

    all_groups: Counter = Counter()
    for groups in hfcs["_groups"]:
        all_groups.update(g.strip() for g in groups if g.strip())
    top_groups = [g for g, c in all_groups.most_common(25) if c >= 3]

    if not top_groups or not top_reactions:
        print("  Not enough groups/reactions for association analysis")
        return pd.DataFrame()

    n = len(hfcs)
    rows = []
    for group in top_groups:
        has_group = hfcs["_groups"].apply(lambda gs: group in [g.strip() for g in gs])
        n_group = has_group.sum()
        if n_group < 2:
            continue

        for reaction in top_reactions:
            has_reaction = hfcs["_reactions_dict"].apply(lambda rd: reaction in rd)
            n_reaction = has_reaction.sum()
            if n_reaction < 2:
                continue

            both = (has_group & has_reaction).sum()

            expected = (n_group * n_reaction) / n if n > 0 else 0
            lift = both / expected if expected > 0 else 0

            table = np.array([
                [both, n_group - both],
                [n_reaction - both, n - n_group - n_reaction + both],
            ])
            table = np.maximum(table, 0)
            if table.sum() > 0 and table.min() >= 0:
                chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
            else:
                chi2, p = 0, 1.0

            rows.append({
                "ingredient_group": group,
                "reaction": reaction,
                "products_with_group": n_group,
                "products_with_reaction": n_reaction,
                "products_with_both": both,
                "lift": round(lift, 2),
                "chi2": round(chi2, 2),
                "p_value": p,
            })

    assoc = pd.DataFrame(rows)
    if assoc.empty:
        return assoc

    assoc = assoc.sort_values("lift", ascending=False)
    assoc.to_csv(OUT_DIR / "ingredient_reaction_associations.csv", index=False)
    print(f"  Saved ingredient_reaction_associations.csv ({len(assoc)} pairs)")

    # Heatmap of lift values
    sig = assoc[assoc["p_value"] < 0.1].copy()
    if len(sig) < 3:
        sig = assoc.head(50)

    pivot = sig.pivot_table(index="ingredient_group", columns="reaction", values="lift", aggfunc="max")
    pivot = pivot.fillna(0)
    # Keep groups/reactions with at least one notable lift
    col_mask = pivot.max() > 1.0
    row_mask = pivot.max(axis=1) > 1.0
    pivot_filt = pivot.loc[row_mask, col_mask] if row_mask.any() and col_mask.any() else pivot

    if pivot_filt.shape[0] >= 2 and pivot_filt.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(max(10, pivot_filt.shape[1] * 0.9), max(6, pivot_filt.shape[0] * 0.5)))
        im = ax.imshow(pivot_filt.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(pivot_filt.shape[1]))
        ax.set_xticklabels(pivot_filt.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(pivot_filt.shape[0]))
        ax.set_yticklabels(pivot_filt.index, fontsize=9)
        ax.set_title("Ingredient Group ↔ Adverse Reaction (Lift)", fontsize=13, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Lift (>1 = positive association)")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "03_ingredient_reaction_heatmap.png")
        plt.close(fig)
        print("  Saved 03_ingredient_reaction_heatmap.png")

    return assoc


# ---------------------------------------------------------------------------
# 4. Ingredient-pair co-occurrence with adverse events
# ---------------------------------------------------------------------------

def ingredient_pair_analysis(prod: pd.DataFrame):
    hfcs = prod[prod["has_hfcs"]].copy()
    if len(hfcs) < 10:
        return

    pair_counts: Counter = Counter()
    pair_severity: defaultdict = defaultdict(list)
    pair_reactions: defaultdict = defaultdict(list)

    for _, row in hfcs.iterrows():
        groups = sorted(set(g.strip() for g in row["_groups"] if g.strip()))
        sev = row["hfcs_max_severity_score"]
        rxns = list(row["_reactions_dict"].keys())
        for a, b in combinations(groups, 2):
            pair = (a, b)
            pair_counts[pair] += 1
            pair_severity[pair].append(sev)
            pair_reactions[pair].extend(rxns)

    if not pair_counts:
        print("  No ingredient pairs found")
        return

    rows = []
    for pair, count in pair_counts.most_common(50):
        sevs = pair_severity[pair]
        rxns = Counter(pair_reactions[pair]).most_common(3)
        rows.append({
            "ingredient_1": pair[0],
            "ingredient_2": pair[1],
            "products_with_adverse_events": count,
            "avg_severity_score": round(np.mean(sevs), 2),
            "max_severity_score": max(sevs),
            "top_reactions": ", ".join(f"{r} ({c})" for r, c in rxns),
        })

    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(OUT_DIR / "ingredient_pair_adverse_events.csv", index=False)
    print(f"  Saved ingredient_pair_adverse_events.csv ({len(pair_df)} pairs)")

    # Bar chart of top 20 pairs by frequency
    top20 = pair_df.head(20)
    fig, ax = plt.subplots(figsize=(10, 8))
    labels = [f"{r['ingredient_1']} + {r['ingredient_2']}" for _, r in top20.iterrows()]
    colors = plt.cm.RdYlGn_r(top20["avg_severity_score"] / max(top20["avg_severity_score"].max(), 1))
    ax.barh(range(len(labels)), top20["products_with_adverse_events"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Products with Adverse Events")
    ax.set_title("Top 20 Ingredient Pairs in Products with Adverse Events\n(color = avg severity)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_ingredient_pairs.png")
    plt.close(fig)
    print("  Saved 04_ingredient_pairs.png")


# ---------------------------------------------------------------------------
# 5. Product type & form vs adverse events
# ---------------------------------------------------------------------------

def product_type_form_analysis(prod: pd.DataFrame):
    hfcs = prod.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Product type
    type_stats = (
        hfcs.groupby("dsld_product_type")
        .agg(
            total=("has_hfcs", "size"),
            with_events=("has_hfcs", "sum"),
            avg_severity=("hfcs_max_severity_score", "mean"),
        )
        .assign(event_rate=lambda x: x["with_events"] / x["total"])
        .query("total >= 5")
        .sort_values("event_rate", ascending=False)
    )
    if not type_stats.empty:
        type_stats["event_rate"].head(15).plot.barh(ax=axes[0], color="#6366f1", edgecolor="white")
        axes[0].set_xlabel("Fraction with Adverse Events")
        axes[0].set_title("Adverse Event Rate by Product Type")
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.3)

    # Supplement form
    form_stats = (
        hfcs.groupby("dsld_form")
        .agg(
            total=("has_hfcs", "size"),
            with_events=("has_hfcs", "sum"),
            avg_severity=("hfcs_max_severity_score", "mean"),
        )
        .assign(event_rate=lambda x: x["with_events"] / x["total"])
        .query("total >= 5")
        .sort_values("event_rate", ascending=False)
    )
    if not form_stats.empty:
        form_stats["event_rate"].head(15).plot.barh(ax=axes[1], color="#f59e0b", edgecolor="white")
        axes[1].set_xlabel("Fraction with Adverse Events")
        axes[1].set_title("Adverse Event Rate by Supplement Form")
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Product Type & Form vs Adverse Events", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_product_type_form.png")
    plt.close(fig)
    print("  Saved 05_product_type_form.png")

    combined = pd.concat([
        type_stats.reset_index().rename(columns={"dsld_product_type": "category"}).assign(kind="product_type"),
        form_stats.reset_index().rename(columns={"dsld_form": "category"}).assign(kind="form"),
    ])
    combined.to_csv(OUT_DIR / "product_type_form_stats.csv", index=False)
    print("  Saved product_type_form_stats.csv")


# ---------------------------------------------------------------------------
# 6. Severity analysis by ingredient group
# ---------------------------------------------------------------------------

def severity_by_ingredient(prod: pd.DataFrame):
    hfcs = prod[prod["has_hfcs"]].copy()
    if len(hfcs) < 5:
        return

    group_severity: defaultdict = defaultdict(list)
    group_reactions: defaultdict = defaultdict(list)

    for _, row in hfcs.iterrows():
        sev = row["hfcs_max_severity_score"]
        rxns = list(row["_reactions_dict"].keys())
        for g in row["_groups"]:
            g = g.strip()
            if g:
                group_severity[g].append(sev)
                group_reactions[g].extend(rxns)

    rows = []
    for group, sevs in group_severity.items():
        if len(sevs) < 3:
            continue
        rxn_top = Counter(group_reactions[group]).most_common(5)
        rows.append({
            "ingredient_group": group,
            "products_with_events": len(sevs),
            "avg_severity": round(np.mean(sevs), 2),
            "max_severity": max(sevs),
            "pct_hospitalization": round(sum(1 for s in sevs if s >= 5) / len(sevs) * 100, 1),
            "top_reactions": ", ".join(f"{r} ({c})" for r, c in rxn_top),
        })

    sev_df = pd.DataFrame(rows).sort_values("avg_severity", ascending=False)
    sev_df.to_csv(OUT_DIR / "severity_by_ingredient_group.csv", index=False)
    print(f"  Saved severity_by_ingredient_group.csv ({len(sev_df)} groups)")

    top25 = sev_df.head(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Reds(top25["avg_severity"] / max(top25["avg_severity"].max(), 1))
    ax.barh(range(len(top25)), top25["avg_severity"], color=colors, edgecolor="white")
    ax.set_yticks(range(len(top25)))
    ax.set_yticklabels(top25["ingredient_group"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Average Max Severity Score (0–7)")
    ax.set_title("Ingredient Groups by Average Severity of Adverse Events", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for i, (_, row) in enumerate(top25.iterrows()):
        ax.text(row["avg_severity"] + 0.05, i, f'n={row["products_with_events"]}', va="center", fontsize=8, color="#666")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_severity_by_ingredient.png")
    plt.close(fig)
    print("  Saved 06_severity_by_ingredient.png")


# ---------------------------------------------------------------------------
# 7. Ingredient count vs adverse event rate & severity
# ---------------------------------------------------------------------------

def ingredient_count_analysis(prod: pd.DataFrame):
    df = prod.copy()
    df["count_bin"] = pd.cut(
        df["dsld_ingredient_count"],
        bins=[0, 1, 3, 5, 10, 20, 100],
        labels=["1", "2–3", "4–5", "6–10", "11–20", "21+"],
    )

    stats_df = (
        df.groupby("count_bin", observed=True)
        .agg(
            total=("has_hfcs", "size"),
            with_events=("has_hfcs", "sum"),
            avg_severity=("hfcs_max_severity_score", "mean"),
            avg_report_count=("hfcs_report_count", "mean"),
        )
        .assign(event_rate=lambda x: x["with_events"] / x["total"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stats_df["event_rate"].plot.bar(ax=axes[0], color="#6366f1", edgecolor="white")
    axes[0].set_ylabel("Fraction with Adverse Events")
    axes[0].set_xlabel("Number of Ingredients")
    axes[0].set_title("Adverse Event Rate by Ingredient Count")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].grid(axis="y", alpha=0.3)

    stats_df["avg_severity"].plot.bar(ax=axes[1], color="#ef4444", edgecolor="white")
    axes[1].set_ylabel("Average Max Severity Score")
    axes[1].set_xlabel("Number of Ingredients")
    axes[1].set_title("Severity by Ingredient Count")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].grid(axis="y", alpha=0.3)

    fig.suptitle("Does Ingredient Complexity Correlate with Adverse Events?", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_ingredient_count_vs_events.png")
    plt.close(fig)
    print("  Saved 07_ingredient_count_vs_events.png")


# ---------------------------------------------------------------------------
# 8. Adverse events vs Amazon ratings
# ---------------------------------------------------------------------------

def events_vs_ratings(prod: pd.DataFrame):
    df = prod.copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    with_events = df[df["has_hfcs"]]["average_rating"].dropna()
    without_events = df[~df["has_hfcs"]]["average_rating"].dropna()

    bins = np.arange(0.75, 5.5, 0.25)
    axes[0].hist(without_events, bins=bins, alpha=0.6, color="#93c5fd", edgecolor="white", label="No adverse events", density=True)
    axes[0].hist(with_events, bins=bins, alpha=0.6, color="#ef4444", edgecolor="white", label="Has adverse events", density=True)
    axes[0].set_xlabel("Average Amazon Rating")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Rating Distribution: With vs Without Adverse Events")
    axes[0].legend()

    if len(with_events) >= 5 and len(without_events) >= 5:
        t_stat, p_val = stats.mannwhitneyu(with_events, without_events, alternative="two-sided")
        axes[0].text(0.03, 0.95, f"Mann-Whitney p={p_val:.4f}", transform=axes[0].transAxes,
                     fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Severity vs rating scatter
    hfcs = df[df["has_hfcs"]].copy()
    if len(hfcs) >= 5:
        axes[1].scatter(hfcs["hfcs_max_severity_score"], hfcs["average_rating"],
                        alpha=0.5, s=hfcs["hfcs_report_count"].clip(upper=50) * 3,
                        color="#6366f1", edgecolors="white", linewidths=0.3)
        axes[1].set_xlabel("Max Severity Score (0–7)")
        axes[1].set_ylabel("Average Amazon Rating")
        axes[1].set_title("Severity vs Rating (size = report count)")
        axes[1].grid(alpha=0.3)

    fig.suptitle("Do Adverse Events Relate to Amazon Ratings?", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_events_vs_ratings.png")
    plt.close(fig)
    print("  Saved 08_events_vs_ratings.png")


# ---------------------------------------------------------------------------
# 9. Specific interesting findings — deep-dive tables
# ---------------------------------------------------------------------------

def deep_dive_findings(prod: pd.DataFrame):
    hfcs = prod[prod["has_hfcs"]].copy()
    if len(hfcs) < 5:
        return

    # Products with highest severity
    severe = hfcs.nlargest(20, "hfcs_max_severity_score")[
        ["dsld_brand_name", "dsld_full_name", "dsld_product_type",
         "dsld_ingredient_count", "hfcs_report_count", "hfcs_max_severity",
         "hfcs_max_severity_score", "hfcs_top_reactions", "average_rating"]
    ].copy()
    severe.to_csv(OUT_DIR / "high_severity_products.csv", index=False)
    print(f"  Saved high_severity_products.csv ({len(severe)} products)")

    # Products with most reports
    most_reported = hfcs.nlargest(20, "hfcs_report_count")[
        ["dsld_brand_name", "dsld_full_name", "dsld_product_type",
         "dsld_ingredient_count", "hfcs_report_count", "hfcs_reaction_count",
         "hfcs_max_severity", "hfcs_top_reactions", "average_rating"]
    ].copy()
    most_reported.to_csv(OUT_DIR / "most_reported_products.csv", index=False)
    print(f"  Saved most_reported_products.csv ({len(most_reported)} products)")

    # Ingredients that appear ONLY in adverse-event products vs never
    all_groups_with = Counter()
    all_groups_without = Counter()
    for _, row in prod.iterrows():
        groups = [g.strip() for g in row["_groups"] if g.strip()]
        if row["has_hfcs"]:
            all_groups_with.update(groups)
        else:
            all_groups_without.update(groups)

    exclusive_rows = []
    for group, cnt in all_groups_with.most_common():
        if cnt < 3:
            continue
        in_safe = all_groups_without.get(group, 0)
        total = cnt + in_safe
        rate = cnt / total if total > 0 else 0
        exclusive_rows.append({
            "ingredient_group": group,
            "in_adverse_products": cnt,
            "in_safe_products": in_safe,
            "adverse_rate": round(rate, 3),
        })

    exc_df = pd.DataFrame(exclusive_rows).sort_values("adverse_rate", ascending=False)
    exc_df.to_csv(OUT_DIR / "ingredient_adverse_rate.csv", index=False)
    print(f"  Saved ingredient_adverse_rate.csv ({len(exc_df)} groups)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df, prod = load_data()

    print("\n--- Generating plots and tables ---")
    plot_overview(prod)
    reaction_counter = plot_top_reactions(prod)
    assoc = compute_ingredient_reaction_associations(prod, reaction_counter)
    ingredient_pair_analysis(prod)
    product_type_form_analysis(prod)
    severity_by_ingredient(prod)
    ingredient_count_analysis(prod)
    events_vs_ratings(prod)
    deep_dive_findings(prod)

    # Print summary
    hfcs = prod[prod["has_hfcs"]]
    print(f"\n{'='*60}")
    print("EDA SUMMARY")
    print(f"{'='*60}")
    print(f"Total unique products:            {len(prod)}")
    print(f"Products with adverse events:     {len(hfcs)} ({len(hfcs)/len(prod)*100:.1f}%)")
    print(f"Total reports across products:    {hfcs['hfcs_report_count'].sum()}")
    print(f"Total reactions across products:  {hfcs['hfcs_reaction_count'].sum()}")
    print(f"Unique reaction types:            {len(reaction_counter)}")
    if not assoc.empty:
        sig_assoc = assoc[assoc["p_value"] < 0.05]
        print(f"Significant associations (p<.05): {len(sig_assoc)}")
        if not sig_assoc.empty:
            print(f"\nTop 5 strongest ingredient→reaction associations (by lift):")
            for _, r in sig_assoc.head(5).iterrows():
                print(f"  {r['ingredient_group']:25s} → {r['reaction']:25s}  lift={r['lift']:.1f}  p={r['p_value']:.4f}")

    print(f"\nAll outputs saved to: {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
