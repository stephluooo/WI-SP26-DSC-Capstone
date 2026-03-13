"""
Merge Amazon Health & Personal Care reviews + metadata with NIH DSLD.
Keep only products where Amazon product name overlaps with a DSLD product.
Output: one combined DataFrame saved as parquet and CSV.

Data sources:
  - Health_and_Personal_Care.jsonl.gz       (Amazon reviews)
  - meta_Health_and_Personal_Care.jsonl.gz  (Amazon product metadata)
  - DSLD-full-database-JSON.zip             (NIH DSLD bulk download)
"""
import gzip
import json
import re
import zipfile
import sys
from pathlib import Path

import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
REVIEWS_FILE = "Health_and_Personal_Care.jsonl.gz"
META_FILE = "meta_Health_and_Personal_Care.jsonl.gz"
DSLD_ZIP = "DSLD-full-database-JSON.zip"

# ============================================================================
# Helpers
# ============================================================================
def find_file(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    alt = Path(name.replace("_and_", "_And_"))
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Neither {name} nor {alt} found")


def normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def tokenize(s: str) -> set:
    """Return set of meaningful tokens (words) from a normalized string."""
    if not s:
        return set()
    stop = {"the", "a", "an", "and", "or", "of", "for", "with", "in", "by", "to",
            "from", "is", "at", "on", "as", "per", "mg", "mcg", "iu", "oz", "ml",
            "count", "ct", "pack", "size", "serving", "servings", "supply", "day",
            "capsules", "capsule", "tablets", "tablet", "softgels", "softgel",
            "gummies", "gummy", "powder", "liquid", "drops", "each", "ea"}
    return {w for w in s.split() if w not in stop and len(w) > 1}


# ============================================================================
# Step 1: Load Amazon metadata → {parent_asin: record}
# ============================================================================
def load_meta(filepath: Path) -> pd.DataFrame:
    print(f"[1/5] Loading Amazon metadata from {filepath}...", flush=True)
    rows = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "parent_asin": rec.get("parent_asin"),
                "amazon_title": rec.get("title") or "",
                "amazon_store": rec.get("store") or "",
                "main_category": rec.get("main_category"),
                "average_rating": rec.get("average_rating"),
                "rating_number": rec.get("rating_number"),
            })
    df = pd.DataFrame(rows)
    df["amazon_name_norm"] = df["amazon_title"].apply(normalize)
    df["amazon_store_norm"] = df["amazon_store"].apply(normalize)
    print(f"       {len(df):,} products loaded.", flush=True)
    return df


# ============================================================================
# Step 2: Load Amazon reviews
# ============================================================================
def load_reviews(filepath: Path) -> pd.DataFrame:
    print(f"[2/5] Loading Amazon reviews from {filepath}...", flush=True)
    rows = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "parent_asin": rec.get("parent_asin") or rec.get("asin"),
                "asin": rec.get("asin"),
                "review_rating": rec.get("rating"),
                "review_title": rec.get("title") or "",
                "review_text": rec.get("text") or "",
                "helpful_vote": rec.get("helpful_vote", 0),
                "verified_purchase": rec.get("verified_purchase", False),
                "timestamp": rec.get("timestamp"),
                "user_id": rec.get("user_id"),
            })
    df = pd.DataFrame(rows)
    print(f"       {len(df):,} reviews loaded.", flush=True)
    return df


# ============================================================================
# Step 3: Load DSLD from local bulk zip
# ============================================================================
def load_dsld_local(zip_path: str) -> pd.DataFrame:
    print(f"[3/5] Loading DSLD from {zip_path}...", flush=True)
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        json_files = [n for n in z.namelist() if n.endswith(".json")]
        total = len(json_files)
        for i, name in enumerate(json_files):
            if (i + 1) % 25000 == 0:
                print(f"       Processed {i + 1:,} / {total:,} DSLD labels...", flush=True)
            try:
                with z.open(name) as f:
                    rec = json.load(f)
            except Exception:
                continue

            # ---- Ingredient rows: full dosage info ----
            ingredient_rows = rec.get("ingredientRows") or []
            ingredients_detailed = []
            ingredient_names = []
            ingredient_groups_set = set()
            for ir in ingredient_rows:
                if not ir:
                    continue
                ing_name = ir.get("name", "")
                ing_category = ir.get("category", "")
                ing_group = ir.get("ingredientGroup", "")
                ing_unii = ir.get("uniiCode", "")
                ing_notes = ir.get("notes", "")
                ing_forms = [f.get("name", "") for f in (ir.get("forms") or []) if f]
                ing_alt_names = ir.get("alternateNames") or []

                # Dosage from quantity array
                qty_list = ir.get("quantity") or []
                dose_str = ""
                dose_amount = None
                dose_unit = ""
                dv_percent = None
                if qty_list:
                    q = qty_list[0]
                    dose_amount = q.get("quantity")
                    dose_unit = q.get("unit", "")
                    dv_groups = q.get("dailyValueTargetGroup") or []
                    if dv_groups:
                        dv_percent = dv_groups[0].get("percent")
                    dv_part = f" ({dv_percent}% DV)" if dv_percent is not None else ""
                    dose_str = f"{ing_name}: {dose_amount} {dose_unit}{dv_part}"

                ingredient_names.append(ing_name)
                if ing_group:
                    ingredient_groups_set.add(ing_group)

                ingredients_detailed.append({
                    "name": ing_name,
                    "group": ing_group,
                    "category": ing_category,
                    "dose_amount": dose_amount,
                    "dose_unit": dose_unit,
                    "dv_percent": dv_percent,
                    "dose_display": dose_str,
                    "forms": ing_forms,
                    "alt_names": ing_alt_names,
                    "unii_code": ing_unii,
                    "notes": ing_notes,
                })

            dose_displays = [d["dose_display"] for d in ingredients_detailed if d["dose_display"]]

            # ---- Other ingredients (inactive/excipients) ----
            other_ing_raw = rec.get("otheringredients") or {}
            other_ing_text = other_ing_raw.get("text", "")
            other_ing_list = [oi.get("name", "") for oi in (other_ing_raw.get("ingredients") or []) if oi]

            # ---- Statements (all types) ----
            statements_raw = rec.get("statements") or []
            statements_dict = {}
            all_statements = []
            for s in statements_raw:
                stype = s.get("type", "")
                snotes = s.get("notes", "")
                statements_dict[stype] = snotes
                all_statements.append(f"{stype}: {snotes}")

            # ---- Claims ----
            claims = [c.get("langualCodeDescription", "").strip() for c in (rec.get("claims") or [])]

            # ---- Events (dates) ----
            events = rec.get("events") or []
            off_market_date = ""
            entry_date = rec.get("entryDate", "")
            for ev in events:
                if ev.get("type") == "Off Market":
                    off_market_date = ev.get("date", "")

            # ---- Contacts ----
            contacts_raw = rec.get("contacts") or []
            contact_names = []
            contact_info = []
            for c in contacts_raw:
                cd = c.get("contactDetails") or {}
                cname = cd.get("name", "")
                if cname:
                    contact_names.append(cname)
                parts = [cd.get("city", ""), cd.get("state", ""), cd.get("zipCode", ""),
                         cd.get("phoneNumber", ""), cd.get("webAddress", "")]
                info = ", ".join(p for p in parts if p)
                if info:
                    contact_info.append(info)

            # ---- Serving sizes ----
            serving_sizes = rec.get("servingSizes") or []
            serving_size_str = ""
            serving_min_qty = None
            serving_max_qty = None
            serving_unit = ""
            serving_daily_min = None
            serving_daily_max = None
            if serving_sizes:
                ss = serving_sizes[0]
                serving_min_qty = ss.get("minQuantity")
                serving_max_qty = ss.get("maxQuantity")
                serving_unit = ss.get("unit", "")
                serving_daily_min = ss.get("minDailyServings")
                serving_daily_max = ss.get("maxDailyServings")
                if serving_min_qty == serving_max_qty:
                    serving_size_str = f"{serving_min_qty} {serving_unit}"
                else:
                    serving_size_str = f"{serving_min_qty}-{serving_max_qty} {serving_unit}"

            # ---- Net contents ----
            net_contents = rec.get("netContents") or []
            net_contents_str = net_contents[0].get("display", "") if net_contents else ""

            # ---- User groups (DV target) ----
            user_groups = [ug.get("dailyValueTargetGroupName", "") for ug in (rec.get("userGroups") or [])]

            # ---- Label relationships ----
            label_rels = rec.get("labelRelationships") or []
            related_ids = [lr.get("labelId") for lr in label_rels if lr.get("labelId")]

            rows.append({
                "dsld_id": rec.get("id"),
                "dsld_full_name": rec.get("fullName", ""),
                "dsld_brand_name": rec.get("brandName", ""),
                "dsld_brand_ip_symbol": rec.get("brandIpSymbol", ""),
                "dsld_upc": rec.get("upcSku", ""),
                "dsld_product_version_code": rec.get("productVersionCode", ""),
                "dsld_nhanes_id": rec.get("nhanesId", ""),
                "dsld_bundle_name": rec.get("bundleName", ""),
                # Physical form
                "dsld_form": (rec.get("physicalState") or {}).get("langualCodeDescription", ""),
                "dsld_form_code": (rec.get("physicalState") or {}).get("langualCode", ""),
                # Product type
                "dsld_product_type": (rec.get("productType") or {}).get("langualCodeDescription", ""),
                "dsld_product_type_code": (rec.get("productType") or {}).get("langualCode", ""),
                # Serving info
                "dsld_servings_per_container": rec.get("servingsPerContainer", ""),
                "dsld_serving_size": serving_size_str,
                "dsld_serving_min_qty": serving_min_qty,
                "dsld_serving_max_qty": serving_max_qty,
                "dsld_serving_unit": serving_unit,
                "dsld_serving_daily_min": serving_daily_min,
                "dsld_serving_daily_max": serving_daily_max,
                # Net contents
                "dsld_net_contents": net_contents_str,
                # Market status & dates
                "dsld_off_market": rec.get("offMarket", 0),
                "dsld_entry_date": entry_date,
                "dsld_off_market_date": off_market_date,
                # Ingredients: structured
                "dsld_ingredient_count": len(ingredient_rows),
                "dsld_ingredient_names": ingredient_names,
                "dsld_ingredient_groups": list(ingredient_groups_set),
                "dsld_ingredient_doses": dose_displays,
                "dsld_ingredients_detailed": json.dumps(ingredients_detailed),
                # Other / inactive ingredients
                "dsld_other_ingredients": other_ing_list,
                "dsld_other_ingredients_text": other_ing_text or "",
                # Claims
                "dsld_claims": claims,
                # Target groups
                "dsld_target_groups": rec.get("targetGroups") or [],
                # DV user groups
                "dsld_dv_user_groups": user_groups,
                # Statements (all)
                "dsld_directions": statements_dict.get("Suggested/Recommended/Usage/Directions", ""),
                "dsld_fda_identity": statements_dict.get("FDA Statement of Identity", ""),
                "dsld_formula_contains": statements_dict.get("Formula re: Contains", ""),
                "dsld_precautions_children": statements_dict.get("Precautions re: Children", ""),
                "dsld_precautions_pregnant": statements_dict.get("Precautions re: Pregnant or Nursing or Prescription Medications", ""),
                "dsld_precautions_other": statements_dict.get("Precautions re: All Other", ""),
                "dsld_storage": statements_dict.get("Storage", ""),
                "dsld_all_statements": all_statements,
                # Contacts
                "dsld_contact_names": contact_names,
                "dsld_contact_info": contact_info,
                # Related labels
                "dsld_related_label_ids": related_ids,
                # DV footnote
                "dsld_percent_dv_footnote": rec.get("percentDvFootnote", ""),
                "dsld_has_outer_carton": rec.get("hasOuterCarton", False),
            })

    df = pd.DataFrame(rows)
    df["dsld_name_norm"] = (df["dsld_brand_name"] + " " + df["dsld_full_name"]).apply(normalize)
    df["dsld_product_norm"] = df["dsld_full_name"].apply(normalize)
    df["dsld_brand_norm"] = df["dsld_brand_name"].apply(normalize)
    print(f"       {len(df):,} DSLD labels loaded.", flush=True)
    return df


# ============================================================================
# Step 4: Match Amazon products to DSLD by normalized name
# ============================================================================
def match_products(meta: pd.DataFrame, dsld: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-strategy matching:
      A) Token-set overlap (order-independent): match if product-name tokens overlap
         enough between Amazon title and DSLD product name.
      B) Substring match: DSLD product name found inside Amazon title (or vice versa).
      C) Brand + product: brand names match AND product-name tokens overlap.

    Uses an inverted token index for speed (single-word lookups included).
    """
    from collections import defaultdict

    print("[4/5] Matching Amazon products to DSLD by name...", flush=True)

    # ---- Build DSLD indexes ----
    print("       Building DSLD indexes...", flush=True)

    # Inverted index: token -> set of dsld_ids
    token_to_dsld = defaultdict(set)
    dsld_info = {}  # dsld_id -> (product_norm, brand_norm, product_tokens)

    for _, row in dsld.iterrows():
        did = row["dsld_id"]
        pn = row["dsld_product_norm"]
        bn = row["dsld_brand_norm"]
        if not pn or len(pn) < 3:
            continue
        ptokens = tokenize(pn)
        dsld_info[did] = (pn, bn, ptokens)
        for tok in ptokens:
            token_to_dsld[tok].add(did)

    print(f"       Index: {len(token_to_dsld):,} unique tokens from {len(dsld_info):,} DSLD products", flush=True)

    # ---- Match each Amazon product ----
    MIN_TOKEN_OVERLAP = 0.6  # 60% of DSLD product tokens must appear in Amazon title
    matches = []
    match_reasons = defaultdict(int)
    checked = 0

    for _, row in meta.iterrows():
        checked += 1
        if checked % 10000 == 0:
            print(f"       Checked {checked:,} / {len(meta):,}, {len(matches):,} matches...", flush=True)

        a_title = row["amazon_name_norm"]
        a_store = row["amazon_store_norm"]
        a_tokens = tokenize(a_title)
        asin = row["parent_asin"]

        if not a_title or len(a_title) < 5:
            continue

        # Gather candidate DSLD IDs: any DSLD product sharing at least one token
        candidate_ids = set()
        for tok in a_tokens:
            if tok in token_to_dsld:
                candidate_ids.update(token_to_dsld[tok])

        if not candidate_ids:
            continue

        best_match = None
        best_score = 0.0

        for did in candidate_ids:
            pn, bn, ptokens = dsld_info[did]

            if not ptokens:
                continue

            # Strategy A: Token-set overlap (order-independent)
            overlap = a_tokens & ptokens
            overlap_ratio = len(overlap) / len(ptokens)

            # Strategy B: Substring match (DSLD product name in Amazon title)
            substring_match = (len(pn) >= 8 and pn in a_title) or (len(a_title) >= 15 and a_title in pn)

            # Strategy C: Brand match boosts confidence
            brand_match = False
            if bn and a_store:
                brand_match = bn in a_store or a_store in bn
            if not brand_match and bn:
                brand_match = bn in a_title

            # Score the match
            score = 0.0
            reason = None

            if substring_match and brand_match:
                score = 1.0
                reason = "substring+brand"
            elif substring_match and len(pn) >= 12:
                score = 0.9
                reason = "substring"
            elif brand_match and overlap_ratio >= MIN_TOKEN_OVERLAP and len(ptokens) >= 2:
                score = 0.85
                reason = "brand+token_overlap"
            elif overlap_ratio >= 0.8 and len(ptokens) >= 2:
                score = 0.7
                reason = "high_token_overlap"
            elif brand_match and overlap_ratio >= 0.5 and len(ptokens) == 1 and len(list(ptokens)[0]) >= 6:
                score = 0.65
                reason = "brand+single_token"
            elif substring_match and len(pn) >= 8:
                score = 0.6
                reason = "short_substring"

            if score > best_score:
                best_score = score
                best_match = (did, reason)

        if best_match and best_score >= 0.6:
            did, reason = best_match
            matches.append({"parent_asin": asin, "dsld_id": did, "match_score": best_score, "match_reason": reason})
            match_reasons[reason] += 1

    match_df = pd.DataFrame(matches)
    if match_df.empty:
        print("       WARNING: No matches found!", flush=True)
        return match_df

    unique_asins = match_df["parent_asin"].nunique()
    unique_dsld = match_df["dsld_id"].nunique()
    print(f"       {unique_asins:,} Amazon products matched to {unique_dsld:,} DSLD products.", flush=True)
    print(f"       Match breakdown:", flush=True)
    for reason, count in sorted(match_reasons.items(), key=lambda x: -x[1]):
        print(f"         {reason}: {count:,}", flush=True)
    return match_df


# ============================================================================
# Step 5: Merge everything into one DataFrame
# ============================================================================
def build_merged(reviews, meta, dsld, matches):
    print("[5/5] Building merged DataFrame...", flush=True)

    matched_asins = set(matches["parent_asin"].unique())
    reviews_m = reviews[reviews["parent_asin"].isin(matched_asins)].copy()
    meta_drop = ["amazon_name_norm", "amazon_store_norm"]
    meta_m = meta[meta["parent_asin"].isin(matched_asins)].drop(columns=meta_drop).copy()

    merged = reviews_m.merge(meta_m, on="parent_asin", how="left")
    merged = merged.merge(matches, on="parent_asin", how="inner")
    dsld_drop = ["dsld_name_norm", "dsld_product_norm", "dsld_brand_norm"]
    merged = merged.merge(
        dsld.drop(columns=dsld_drop),
        on="dsld_id",
        how="left",
    )

    print(f"       Final: {len(merged):,} rows x {len(merged.columns)} columns.", flush=True)
    return merged


# ============================================================================
# Main
# ============================================================================
def main():
    meta = load_meta(find_file(META_FILE))
    reviews = load_reviews(find_file(REVIEWS_FILE))
    dsld = load_dsld_local(DSLD_ZIP)

    matches = match_products(meta, dsld)
    if matches.empty:
        print("\nNo overlapping products found. Exiting.")
        return

    merged = build_merged(reviews, meta, dsld, matches)

    # Save as CSV (parquet requires pyarrow; install with `pip install pyarrow` if wanted)
    merged.to_csv("amazon_dsld_merged.csv", index=False)
    print(f"\nSaved: amazon_dsld_merged.csv")

    try:
        merged.to_parquet("amazon_dsld_merged.parquet", index=False)
        print(f"Saved: amazon_dsld_merged.parquet")
    except ImportError:
        print("(Skipped parquet -- install pyarrow for parquet support)")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Amazon products matched:  {matches['parent_asin'].nunique():,}")
    print(f"DSLD products matched:    {matches['dsld_id'].nunique():,}")
    print(f"Total review rows:        {len(merged):,}")
    print(f"Columns:                  {len(merged.columns)}")
    print(f"\nColumn list:")
    for c in merged.columns:
        print(f"  - {c}")
    print(f"\nSample (first 3 rows):")
    print(merged[["parent_asin", "amazon_title", "review_rating", "dsld_full_name", "dsld_brand_name", "dsld_form", "dsld_ingredient_count"]].head(3).to_string())


if __name__ == "__main__":
    main()
