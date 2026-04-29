"""
Convert FAERS CSV (or zipped CSV) to Parquet with snappy compression.

Usage:
    python convert_to_parquet.py                          # auto-detect CSV or ZIP
    python convert_to_parquet.py --input data/faers_full.csv
    python convert_to_parquet.py --input data/faers_full.csv.zip

Run this once after download_faers.py finishes. ddi_study.py will
automatically use the Parquet file when available.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "faers_full.csv"
ZIP_PATH = DATA_DIR / "faers_full.csv.zip"
PARQUET_PATH = DATA_DIR / "faers_full.parquet"


def main():
    parser = argparse.ArgumentParser(description="Convert FAERS CSV to Parquet")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to CSV or zipped CSV (auto-detects if omitted)")
    args = parser.parse_args()

    if args.input:
        src = Path(args.input)
    elif CSV_PATH.exists():
        src = CSV_PATH
    elif ZIP_PATH.exists():
        src = ZIP_PATH
    else:
        print(f"ERROR: No FAERS CSV found. Expected {CSV_PATH} or {ZIP_PATH}")
        sys.exit(1)

    print(f"Reading {src} ...")
    if str(src).endswith(".zip"):
        df = pd.read_csv(src, dtype=str, low_memory=False)
    else:
        import csv as csvmod
        csvmod.field_size_limit(sys.maxsize)
        df = pd.read_csv(src, dtype=str, low_memory=False)

    print(f"  {len(df):,} rows, {df.shape[1]} columns")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    print(f"Writing {PARQUET_PATH} ...")
    df.to_parquet(PARQUET_PATH, engine="pyarrow", compression="snappy", index=False)

    src_mb = src.stat().st_size / 1e6
    pq_mb = PARQUET_PATH.stat().st_size / 1e6
    ratio = src_mb / pq_mb if pq_mb > 0 else 0
    print(f"  Source:  {src_mb:,.1f} MB")
    print(f"  Parquet: {pq_mb:,.1f} MB  ({ratio:.1f}x smaller)")
    print("Done.")


if __name__ == "__main__":
    main()
