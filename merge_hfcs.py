"""
Pull FDA HFCS adverse event reports for dietary supplements, aggregate to
product-level summaries, match to the existing amazon_dsld_merged dataset,
and output an enriched CSV.

Data source: OpenFDA Food Adverse Event API (industry_code 54 = supplements)
"""
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import requests

# ============================================================================
# Configuration
# ============================================================================
OPENFDA_URL = "https://api.fda.gov/food/event.json"
INDUSTRY_CODE = "54"  # Vit/Min/Prot/Unconv Diet(Human/Animal)
PAGE_SIZE = 1000
MERGED_CSV = "data/amazon_dsld_merged.csv"
OUTPUT_CSV = "data/amazon_dsld_hfcs_merged.csv"

OUTCOME_SEVERITY = {
    "Death": 7,
    "Life Threatening": 6,
    "Hospitalization": 5,
    "Disability": 4,
    "Required Intervention": 3,
    "Visited Emergency Room": 2,
    "Visited a Health Care Provider": 1,
    "Other Serious or Important Medical Event": 1,
    "Other Serious Outcome": 1,
    "Allergic Reaction": 1,
    "Injury": 1,
    "Congenital Anomaly": 1,
    "Other Outcome": 0,
}


# ============================================================================
# Helpers (reuse normalize/tokenize from merge_amazon_dsld.py)
# ============================================================================
def normalize(s: str) -> str:
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_brand(brand: str) -> str:
    if not brand:
        return ""
    b = normalize(brand)
    b = re.sub(r"\b(inc|llc|ltd|co|corp|corporation|company|laboratories|laboratory|labs|lab|international|intl|usa|us)\b", "", b)
    b = re.sub(r"\s+", " ", b).strip()
    return b


def tokenize(s: str) -> set:
    if not s:
        return set()
    stop = {"the", "a", "an", "and", "or", "of", "for", "with", "in", "by", "to",
            "from", "is", "at", "on", "as", "per", "mg", "mcg", "iu", "oz", "ml",
            "count", "ct", "pack", "size", "serving", "servings", "supply", "day",
            "capsules", "capsule", "tablets", "tablet", "softgels", "softgel",
            "gummies", "gummy", "powder", "liquid", "drops", "each", "ea"}
    return {w for w in s.split() if w not in stop and len(w) > 1}


# ============================================================================
# Step 1: Pull all HFCS supplement records from OpenFDA
# ============================================================================
def pull_hfcs() -> list[dict]:
    """Pull all HFCS supplement records using date-range windowing.

    OpenFDA caps skip at 25,000, so we split the full date range into
    windows small enough that each fits under 25K results, then paginate
    within each window.
    """
    print("[1/4] Pulling HFCS supplement adverse events from OpenFDA...", flush=True)

    first_resp = requests.get(OPENFDA_URL, params={
        "search": f"products.industry_code:{INDUSTRY_CODE}",
        "limit": 1,
    })
    total = first_resp.json()["meta"]["results"]["total"]
    print(f"       Total available: {total:,} records", flush=True)

    date_windows = [
        ("20040101", "20141231"),
        ("20150101", "20181231"),
        ("20190101", "20201231"),
        ("20210101", "20221231"),
        ("20230101", "20241231"),
        ("20250101", "20261231"),
    ]

    all_records = []
    seen_report_nums = set()

    for win_start, win_end in date_windows:
        search_q = f"products.industry_code:{INDUSTRY_CODE} AND date_created:[{win_start} TO {win_end}]"
        skip = 0
        window_count = 0
        retries = 0

        while True:
            params = {
                "search": search_q,
                "limit": PAGE_SIZE,
                "skip": skip,
            }
            try:
                resp = requests.get(OPENFDA_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                retries = 0
            except Exception as e:
                retries += 1
                if retries > 3:
                    print(f"       Giving up on window {win_start}-{win_end} at skip={skip}", flush=True)
                    break
                print(f"       Error at skip={skip}: {e}. Retry {retries}/3...", flush=True)
                time.sleep(5)
                continue

            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                rn = rec.get("report_number", "")
                if rn not in seen_report_nums:
                    seen_report_nums.add(rn)
                    all_records.append(rec)
                    window_count += 1

            skip += PAGE_SIZE
            if skip >= 25000:
                print(f"       WARNING: window {win_start}-{win_end} hit 25K skip limit", flush=True)
                break

            time.sleep(0.2)

        print(f"       Window {win_start}-{win_end}: {window_count:,} new records (total so far: {len(all_records):,})", flush=True)

    print(f"       Done: {len(all_records):,} unique records pulled.", flush=True)
    return all_records


# ============================================================================
# Step 2: Flatten and aggregate to product-level summaries
# ============================================================================
def aggregate_hfcs(records: list[dict]) -> pd.DataFrame:
    print("[2/4] Aggregating HFCS to product-level summaries...", flush=True)

    product_reports = defaultdict(lambda: {
        "report_numbers": [],
        "reactions": [],
        "outcomes": [],
        "ages": [],
        "genders": [],
        "dates": [],
    })

    for rec in records:
        report_num = rec.get("report_number", "")
        reactions = rec.get("reactions") or []
        outcomes = rec.get("outcomes") or []
        consumer = rec.get("consumer") or {}
        date_created = rec.get("date_created", "")

        age = None
        age_str = consumer.get("age")
        age_unit = consumer.get("age_unit", "")
        if age_str:
            try:
                age_val = float(age_str)
                if "month" in age_unit.lower():
                    age_val /= 12.0
                age = age_val
            except (ValueError, TypeError):
                pass
        gender = consumer.get("gender", "")

        for prod in rec.get("products", []):
            if prod.get("industry_code") != INDUSTRY_CODE:
                continue
            if prod.get("role", "").upper() != "SUSPECT":
                continue

            name = prod.get("name_brand", "")
            if not name:
                continue

            name_key = normalize(name)
            if not name_key or len(name_key) < 3:
                continue

            bucket = product_reports[name_key]
            bucket["report_numbers"].append(report_num)
            bucket["reactions"].extend(reactions)
            bucket["outcomes"].extend(outcomes)
            if age is not None:
                bucket["ages"].append(age)
            if gender:
                bucket["genders"].append(gender)
            if date_created:
                bucket["dates"].append(date_created)

    rows = []
    for name_key, data in product_reports.items():
        reaction_counts = Counter(r.lower() for r in data["reactions"])
        outcome_counts = Counter(data["outcomes"])
        top_5 = [r for r, _ in reaction_counts.most_common(5)]

        max_sev_score = max((OUTCOME_SEVERITY.get(o, 0) for o in outcome_counts), default=0)
        max_sev_label = "None"
        for o, s in OUTCOME_SEVERITY.items():
            if s == max_sev_score and o in outcome_counts:
                max_sev_label = o
                break

        dates_sorted = sorted(d for d in data["dates"] if d)
        gender_counts = Counter(data["genders"])

        rows.append({
            "hfcs_product_norm": name_key,
            "hfcs_report_count": len(set(data["report_numbers"])),
            "hfcs_reaction_count": len(data["reactions"]),
            "hfcs_top_reactions": json.dumps(top_5),
            "hfcs_all_reactions": json.dumps(dict(reaction_counts.most_common(50))),
            "hfcs_outcome_hospitalization": outcome_counts.get("Hospitalization", 0),
            "hfcs_outcome_er": outcome_counts.get("Visited Emergency Room", 0),
            "hfcs_outcome_death": outcome_counts.get("Death", 0),
            "hfcs_outcome_life_threatening": outcome_counts.get("Life Threatening", 0),
            "hfcs_outcome_disability": outcome_counts.get("Disability", 0),
            "hfcs_max_severity": max_sev_label,
            "hfcs_max_severity_score": max_sev_score,
            "hfcs_avg_consumer_age": round(sum(data["ages"]) / len(data["ages"]), 1) if data["ages"] else None,
            "hfcs_pct_female": round(gender_counts.get("Female", 0) / len(data["genders"]) * 100, 1) if data["genders"] else None,
            "hfcs_date_earliest": dates_sorted[0] if dates_sorted else "",
            "hfcs_date_latest": dates_sorted[-1] if dates_sorted else "",
        })

    df = pd.DataFrame(rows)
    print(f"       {len(df):,} unique HFCS product names aggregated.", flush=True)
    return df


# ============================================================================
# Step 3: Match HFCS product names to existing merged dataset
# ============================================================================
def match_hfcs_to_merged(hfcs_agg: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    print("[3/4] Matching HFCS products to Amazon-DSLD merged dataset...", flush=True)

    merged_products = merged.drop_duplicates(subset=["parent_asin"])[
        ["parent_asin", "amazon_title", "amazon_store", "dsld_id", "dsld_full_name", "dsld_brand_name"]
    ].copy()

    merged_products["title_norm"] = merged_products["amazon_title"].apply(normalize)
    merged_products["store_norm"] = merged_products["amazon_store"].apply(normalize_brand)
    merged_products["dsld_name_norm"] = (
        merged_products["dsld_brand_name"].fillna("") + " " + merged_products["dsld_full_name"].fillna("")
    ).apply(normalize)
    merged_products["dsld_brand_norm"] = merged_products["dsld_brand_name"].apply(normalize_brand)

    token_to_asin = defaultdict(set)
    asin_info = {}

    for _, row in merged_products.iterrows():
        asin = row["parent_asin"]
        title_norm = row["title_norm"]
        store_norm = row["store_norm"]
        dsld_norm = row["dsld_name_norm"]
        dsld_brand = row["dsld_brand_norm"]
        combined_norm = f"{title_norm} {dsld_norm}"
        tokens = tokenize(combined_norm)
        asin_info[asin] = (title_norm, store_norm, dsld_norm, dsld_brand, tokens)
        for tok in tokens:
            token_to_asin[tok].add(asin)

    print(f"       Index: {len(token_to_asin):,} tokens from {len(asin_info):,} merged products", flush=True)

    matches = {}
    match_reasons = Counter()

    for _, hrow in hfcs_agg.iterrows():
        hfcs_norm = hrow["hfcs_product_norm"]
        hfcs_tokens = tokenize(hfcs_norm)

        if not hfcs_tokens or len(hfcs_norm) < 3:
            continue

        candidate_asins = set()
        for tok in hfcs_tokens:
            if tok in token_to_asin:
                candidate_asins.update(token_to_asin[tok])

        if not candidate_asins:
            continue

        best_asin = None
        best_score = 0.0
        best_reason = None

        for asin in candidate_asins:
            title_norm, store_norm, dsld_norm, dsld_brand, m_tokens = asin_info[asin]

            if not m_tokens:
                continue

            overlap = hfcs_tokens & m_tokens
            overlap_fwd = len(overlap) / len(hfcs_tokens) if hfcs_tokens else 0
            overlap_rev = len(overlap) / len(m_tokens) if m_tokens else 0

            substring_in_title = len(hfcs_norm) >= 10 and hfcs_norm in title_norm
            substring_in_dsld = len(hfcs_norm) >= 10 and hfcs_norm in dsld_norm

            hfcs_brand_tokens = tokenize(hfcs_norm)
            brand_match = False
            if store_norm and len(store_norm) >= 3:
                brand_match = store_norm in hfcs_norm or hfcs_norm.startswith(store_norm)
            if not brand_match and dsld_brand and len(dsld_brand) >= 3:
                brand_match = dsld_brand in hfcs_norm or hfcs_norm.startswith(dsld_brand)

            score = 0.0
            reason = None

            if (substring_in_title or substring_in_dsld) and brand_match:
                score = 1.0
                reason = "substring+brand"
            elif substring_in_title or substring_in_dsld:
                score = 0.85
                reason = "substring"
            elif brand_match and overlap_fwd >= 0.6 and len(overlap) >= 2:
                score = 0.8
                reason = "brand+token_overlap"
            elif overlap_fwd >= 0.8 and len(overlap) >= 2 and overlap_rev >= 0.3:
                score = 0.7
                reason = "high_token_overlap"

            if score > best_score and score >= 0.7:
                best_score = score
                best_asin = asin
                best_reason = reason

        if best_asin:
            if best_asin not in matches or best_score > matches[best_asin][1]:
                matches[best_asin] = (hfcs_norm, best_score, best_reason)
                match_reasons[best_reason] += 1

    print(f"       {len(matches):,} merged products matched to HFCS reports.", flush=True)
    for reason, count in match_reasons.most_common():
        print(f"         {reason}: {count:,}", flush=True)

    match_rows = []
    for asin, (hfcs_norm, score, reason) in matches.items():
        hfcs_row = hfcs_agg[hfcs_agg["hfcs_product_norm"] == hfcs_norm].iloc[0]
        row_dict = hfcs_row.to_dict()
        row_dict["parent_asin"] = asin
        row_dict["hfcs_match_score"] = score
        row_dict["hfcs_match_reason"] = reason
        match_rows.append(row_dict)

    return pd.DataFrame(match_rows)


# ============================================================================
# Step 4: Left-join onto merged CSV
# ============================================================================
def join_and_save(hfcs_matched: pd.DataFrame, merged: pd.DataFrame):
    print("[4/4] Joining HFCS data onto merged CSV...", flush=True)

    hfcs_cols = [c for c in hfcs_matched.columns if c != "parent_asin"]
    hfcs_join = hfcs_matched[["parent_asin"] + hfcs_cols].copy()

    result = merged.merge(hfcs_join, on="parent_asin", how="left")

    result["hfcs_report_count"] = result["hfcs_report_count"].fillna(0).astype(int)
    for col in ["hfcs_outcome_hospitalization", "hfcs_outcome_er", "hfcs_outcome_death",
                "hfcs_outcome_life_threatening", "hfcs_outcome_disability",
                "hfcs_max_severity_score", "hfcs_reaction_count"]:
        if col in result.columns:
            result[col] = result[col].fillna(0).astype(int)

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}", flush=True)

    matched_asins = set(hfcs_matched["parent_asin"])
    rows_with_hfcs = result[result["parent_asin"].isin(matched_asins)]

    print(f"\n{'='*60}")
    print("HFCS MERGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows in output:           {len(result):,}")
    print(f"Total columns:                  {len(result.columns)}")
    print(f"Products with HFCS data:        {len(matched_asins):,}")
    print(f"Review rows with HFCS data:     {len(rows_with_hfcs):,}")
    print(f"Products without HFCS data:     {result['parent_asin'].nunique() - len(matched_asins):,}")
    print(f"\nNew HFCS columns:")
    for c in hfcs_cols:
        print(f"  - {c}")

    print(f"\nSample matched products:")
    sample = hfcs_matched.sort_values("hfcs_report_count", ascending=False).head(10)
    pd.set_option("display.max_colwidth", 50)
    pd.set_option("display.width", 200)
    print(sample[["hfcs_product_norm", "hfcs_report_count", "hfcs_top_reactions", "hfcs_max_severity", "hfcs_match_score"]].to_string(index=False))

    return result


# ============================================================================
# Main
# ============================================================================
def main():
    records = pull_hfcs()

    hfcs_agg = aggregate_hfcs(records)

    print(f"\n       HFCS stats: {hfcs_agg['hfcs_report_count'].sum():,} total reports across {len(hfcs_agg):,} products", flush=True)
    print(f"       Top 10 HFCS products by report count:", flush=True)
    top = hfcs_agg.nlargest(10, "hfcs_report_count")
    for _, r in top.iterrows():
        print(f"         {r['hfcs_report_count']:>5,}  {r['hfcs_product_norm'][:60]}", flush=True)

    print(f"\n       Loading merged dataset from {MERGED_CSV}...", flush=True)
    merged = pd.read_csv(MERGED_CSV)
    print(f"       {len(merged):,} rows, {merged['parent_asin'].nunique():,} unique products.", flush=True)

    hfcs_matched = match_hfcs_to_merged(hfcs_agg, merged)

    if hfcs_matched.empty:
        print("\nNo HFCS matches found. Saving original merged CSV unchanged.")
        merged.to_csv(OUTPUT_CSV, index=False)
        return

    join_and_save(hfcs_matched, merged)


if __name__ == "__main__":
    main()
