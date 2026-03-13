"""
Sample 200 random rows from the merged CSV, keeping only merge-relevant columns.
Inject exactly 10 false rows with random noise to simulate incorrect matches.
"""
import pandas as pd
import numpy as np
import random
import string

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

df = pd.read_csv("amazon_dsld_merged.csv")

merge_cols = [
    "parent_asin",
    "amazon_title",
    "amazon_store",
    "dsld_id",
    "dsld_full_name",
    "dsld_brand_name",
    "match_score",
    "match_reason",
]

df_slim = df[merge_cols].copy()

# Deduplicate to unique Amazon-DSLD pairs (one row per match, not per review)
df_slim = df_slim.drop_duplicates(subset=["parent_asin", "dsld_id"])

sample = df_slim.sample(n=200, random_state=SEED).reset_index(drop=True)

# ---------- Generate 10 false rows ----------

def random_asin():
    return "B0" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

def garble(text):
    """Shuffle words and inject a random word to break any real match."""
    if not isinstance(text, str) or not text:
        return "NOISE_PRODUCT"
    words = text.split()
    random.shuffle(words)
    noise_word = "".join(random.choices(string.ascii_lowercase, k=6))
    words.insert(random.randint(0, len(words)), noise_word)
    return " ".join(words[:8])

real_amazon = sample.sample(n=10, random_state=SEED + 1)
real_dsld = sample.sample(n=10, random_state=SEED + 2).reset_index(drop=True)

false_rows = []
for i in range(10):
    a_row = real_amazon.iloc[i]
    d_row = real_dsld.iloc[i]
    false_rows.append({
        "parent_asin": random_asin(),
        "amazon_title": garble(a_row["amazon_title"]),
        "amazon_store": garble(a_row["amazon_store"]),
        "dsld_id": int(d_row["dsld_id"]) + random.randint(900000, 999999),
        "dsld_full_name": garble(d_row["dsld_full_name"]),
        "dsld_brand_name": garble(d_row["dsld_brand_name"]),
        "match_score": round(random.uniform(0.6, 1.0), 2),
        "match_reason": random.choice(["substring", "high_token_overlap", "brand+token_overlap", "short_substring"]),
    })

false_df = pd.DataFrame(false_rows)
false_df["is_false"] = True
sample["is_false"] = False

combined = pd.concat([sample, false_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=SEED).reset_index(drop=True)

combined.to_csv("amazon_dsld_sample_210.csv", index=False)

print(f"Real rows:  {(~combined['is_false']).sum()}")
print(f"False rows: {combined['is_false'].sum()}")
print(f"Total:      {len(combined)}")
print(f"Columns:    {list(combined.columns)}")
print(f"\nSaved: amazon_dsld_sample_210.csv")
print(f"\nSample false row:")
print(combined[combined["is_false"]].iloc[0].to_string())
