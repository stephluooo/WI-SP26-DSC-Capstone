"""
Load Amazon Health & Personal Care reviews from JSONL.gz
and print 10 entries for Magnesium-related products.
"""
import gzip
import json
import re
from datetime import datetime

FILE = "Health_and_Personal_Care.jsonl.gz"
KEYWORD = re.compile(r"\bmagnesium\b", re.IGNORECASE)
LIMIT = 10

# Supplement-related terms that should appear in the review to qualify
SUPPLEMENT_SIGNALS = re.compile(
    r"\b("
    r"supplement|capsule|tablet|softgel|vitamin|mineral|dose|dosage|"
    r"daily value|serving|mg\b|milligram|dietary|nutrient|bioavail|"
    r"absorb|deficien|glycinate|citrate|oxide|threonate|taurate|"
    r"chelat|powder|pill|gummies|gummy"
    r")\b",
    re.IGNORECASE,
)

# Non-supplement products to exclude
EXCLUDE = re.compile(
    r"\b("
    r"bath bomb|lotion|cream|shampoo|conditioner|soap|candle|"
    r"deodorant|perfume|fragrance|mascara|lipstick|moisturizer|"
    r"body wash|body butter|scrub|exfoli|makeup|cosmetic|"
    r"detergent|cleaning|toothpaste|mouthwash"
    r")\b",
    re.IGNORECASE,
)

print(f"Scanning {FILE} for Magnesium supplement reviews...")
print(f"Filtering: must mention supplement terms, excluding personal care products\n")

matches = []

with gzip.open(FILE, "rt", encoding="utf-8") as f:
    scanned = 0
    for line in f:
        scanned += 1
        record = json.loads(line)

        title = record.get("title", "")
        text = record.get("text", "")
        combined = f"{title} {text}"

        # Must mention magnesium
        if not KEYWORD.search(combined):
            continue

        # Must have supplement-related language
        if not SUPPLEMENT_SIGNALS.search(combined):
            continue

        # Exclude personal care / non-supplement products
        if EXCLUDE.search(combined):
            continue

        matches.append(record)
        if len(matches) >= LIMIT:
            break

        # Progress update every 50k lines
        if scanned % 50000 == 0:
            print(f"  ...scanned {scanned:,} reviews, found {len(matches)} matches so far")

print(f"\nScanned {scanned:,} reviews total. Found {len(matches)} Magnesium matches.\n")
print("=" * 80)

for i, r in enumerate(matches, 1):
    # Convert timestamp to readable date
    ts = r.get("timestamp", 0)
    date_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d") if ts else "N/A"

    rating = r.get("rating", "N/A")
    stars = "*" * int(rating) if isinstance(rating, (int, float)) else "N/A"
    title = r.get("title", "N/A")
    text = r.get("text", "")
    asin = r.get("asin", "N/A")
    verified = "Yes" if r.get("verified_purchase") else "No"
    helpful = r.get("helpful_vote", 0)

    # Truncate long review text
    text_preview = text[:250] + "..." if len(text) > 250 else text

    print(f"\n--- Review {i} ---")
    print(f"  ASIN:       {asin}  (https://amazon.com/dp/{asin})")
    print(f"  Rating:     {rating}/5  {stars}")
    print(f"  Title:      {title}")
    print(f"  Date:       {date_str}")
    print(f"  Verified:   {verified}")
    print(f"  Helpful:    {helpful} vote(s)")
    print(f"  Review:     {text_preview}")

print("\n" + "=" * 80)
print("Done.")
