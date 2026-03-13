import pandas as pd

df = pd.read_csv("amazon_dsld_merged.csv", nrows=10)
print(f"Shape: {df.shape}\n")

cols = [
    "parent_asin", "amazon_title", "review_rating", "review_title",
    "dsld_full_name", "dsld_brand_name", "dsld_form", "dsld_product_type",
    "dsld_ingredient_count", "dsld_ingredient_groups", "verified_purchase",
]

for i, row in df.iterrows():
    print(f"--- Row {i+1} ---")
    for c in cols:
        val = str(row.get(c, "N/A"))
        if len(val) > 120:
            val = val[:120] + "..."
        print(f"  {c:25s}: {val}")
    txt = str(row.get("review_text", ""))[:150]
    print(f"  {'review_text':25s}: {txt}...")
    print()
