"""Analyze what our matching approach misses."""
import gzip
import json
import re
import pandas as pd

def normalize(s):
    if not s: return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# Load Amazon meta
print("Loading Amazon metadata...")
meta = []
with gzip.open("meta_Health_and_Personal_Care.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        title = rec.get("title") or ""
        store = rec.get("store") or ""
        meta.append({
            "asin": rec.get("parent_asin"),
            "title_raw": title,
            "store_raw": store,
            "title_norm": normalize(title),
            "combined_norm": normalize(store + " " + title),
        })

titles = [m["title_norm"] for m in meta if m["title_norm"]]

print(f"\nTotal Amazon products: {len(meta):,}")
print(f"With titles: {len(titles):,}")

# Issue 1: 15-char minimum cuts off short names
print(f"\n{'='*60}")
print("ISSUE 1: 15-char minimum on normalized combined name")
print(f"{'='*60}")
combined = [m["combined_norm"] for m in meta if m["combined_norm"]]
under_15 = [c for c in combined if len(c) < 15]
print(f"  Combined names < 15 chars: {len(under_15):,} (all skipped)")
for c in under_15[:10]:
    print(f'    "{c}"')

# Issue 2: Single-word product names produce zero 2-grams → no candidates found
print(f"\n{'='*60}")
print("ISSUE 2: Single-word DSLD names produce no 2-grams")
print(f"{'='*60}")
print("  (These are invisible to the n-gram index)")
print("  Common supplement names that are 1 word:")
single_words = ["melatonin", "biotin", "ashwagandha", "turmeric", "elderberry",
                 "magnesium", "zinc", "iron", "calcium", "collagen",
                 "probiotics", "multivitamin", "creatine", "glutathione"]
for w in single_words:
    count = sum(1 for t in titles if w in t)
    print(f'    "{w}" appears in {count:,} Amazon titles')

# Issue 3: Substring matching misses reordered words
print(f"\n{'='*60}")
print("ISSUE 3: Substring matching requires exact word order")
print(f"{'='*60}")
print('  e.g., DSLD: "calcium magnesium zinc" won\'t match')
print('        Amazon: "zinc magnesium calcium supplement"')

# Issue 4: Brand name discrepancies
print(f"\n{'='*60}")
print("ISSUE 4: Brand name mismatches")
print(f"{'='*60}")
print("  Amazon 'store' vs DSLD 'brandName' may differ:")
print('    Amazon: "NOW Foods"  vs  DSLD: "NOW"')
print('    Amazon: "Nature Made" vs  DSLD: "NatureMade"')
print('    Amazon: "Doctor\'s Best" vs DSLD: "Doctors Best"')

# Issue 5: Abbreviations and variations
print(f"\n{'='*60}")
print("ISSUE 5: Naming variations")
print(f"{'='*60}")
print('  "Vitamin D3" vs "Vitamin D-3" vs "D3"')
print('  "CoQ10" vs "Coenzyme Q10" vs "Co Q-10"')
print('  "Fish Oil" vs "Omega-3 Fish Oil"')

# What we actually matched vs could match
print(f"\n{'='*60}")
print("CURRENT MATCH STATS")
print(f"{'='*60}")
matched = pd.read_csv("amazon_dsld_merged.csv", usecols=["parent_asin"])
unique_matched = matched["parent_asin"].nunique()
print(f"  Matched:   {unique_matched:,} / {len(meta):,} Amazon products ({100*unique_matched/len(meta):.1f}%)")
print(f"  Unmatched: {len(meta) - unique_matched:,}")

# Estimate: how many Amazon titles contain common supplement keywords
supp_keywords = re.compile(
    r"\b(vitamin|supplement|capsule|tablet|softgel|probiotic|mineral|"
    r"magnesium|zinc|calcium|iron|biotin|collagen|melatonin|turmeric|"
    r"ashwagandha|elderberry|fish oil|omega|coq10|multivitamin|"
    r"creatine|protein powder|bcaa|glutamine|fiber|folate|folic)\b", re.IGNORECASE
)
likely_supplements = sum(1 for m in meta if supp_keywords.search(m["title_raw"]))
print(f"\n  Amazon titles with supplement keywords: {likely_supplements:,}")
print(f"  Of those, we matched: {unique_matched:,}")
print(f"  Potential gap: ~{likely_supplements - unique_matched:,} supplement products possibly missed")
