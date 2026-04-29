"""
Convert FAERS CSV (or zipped CSV) to Parquet with snappy compression.
Processes in chunks to avoid OOM on large files.

Usage:
    python convert_to_parquet.py                          # auto-detect CSV or ZIP
    python convert_to_parquet.py --input data/faers_full.csv
    python convert_to_parquet.py --chunk-size 500000      # rows per chunk

Run this once after download_faers.py finishes. ddi_study.py will
automatically use the Parquet file when available.
"""

import argparse
import csv as csvmod
import io
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = Path("data")
CSV_PATH = DATA_DIR / "faers_full.csv"
ZIP_PATH = DATA_DIR / "faers_full.csv.zip"
PARQUET_PATH = DATA_DIR / "faers_full.parquet"


def open_csv_reader(src: Path, chunk_size: int):
    """Yield DataFrames in chunks from CSV or zipped CSV."""
    import pandas as pd
    csvmod.field_size_limit(sys.maxsize)
    kwargs = dict(dtype=str, low_memory=False, chunksize=chunk_size)
    if str(src).endswith(".zip"):
        zf = zipfile.ZipFile(src)
        name = zf.namelist()[0]
        buf = io.TextIOWrapper(zf.open(name), encoding="utf-8", errors="replace")
        yield from pd.read_csv(buf, **kwargs)
    else:
        yield from pd.read_csv(src, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Convert FAERS CSV to Parquet")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to CSV or zipped CSV (auto-detects if omitted)")
    parser.add_argument("--chunk-size", type=int, default=500_000,
                        help="Rows per chunk (default: 500000)")
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

    print(f"Converting {src} -> {PARQUET_PATH}")
    print(f"  Chunk size: {args.chunk_size:,} rows")

    writer = None
    total_rows = 0

    for i, chunk in enumerate(open_csv_reader(src, args.chunk_size)):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                PARQUET_PATH, table.schema, compression="snappy"
            )
        writer.write_table(table)
        total_rows += len(chunk)
        print(f"  Chunk {i+1}: {len(chunk):,} rows  (total: {total_rows:,})", flush=True)

    if writer:
        writer.close()

    src_mb = src.stat().st_size / 1e6
    pq_mb = PARQUET_PATH.stat().st_size / 1e6
    ratio = src_mb / pq_mb if pq_mb > 0 else 0
    print(f"\n  Source:  {src_mb:,.1f} MB")
    print(f"  Parquet: {pq_mb:,.1f} MB  ({ratio:.1f}x smaller)")
    print(f"  Rows:    {total_rows:,}")
    print("Done.")


if __name__ == "__main__":
    main()
