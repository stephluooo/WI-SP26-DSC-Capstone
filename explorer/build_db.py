"""Build an indexed SQLite database from the DDI study result files.

Run once before launching the Streamlit explorer:

    python explorer/build_db.py

It streams the large signal table (~9M rows / 650 MB) in chunks so memory
stays low, then creates indexes for fast pair / drug / reaction lookups.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "ddi_study"
DATA_DIR = ROOT / "data"
DB_PATH = RESULTS_DIR / "ddi.db"

SIGNALS_CSV = RESULTS_DIR / "phase1_signals.csv"
LABELED_CSV = RESULTS_DIR / "phase1_labeled_pairs.csv"
MATCH_CSV = RESULTS_DIR / "phase2_match_details.csv"
METRICS_CSV = RESULTS_DIR / "phase3_metrics.csv"
NOVEL_CSV = RESULTS_DIR / "phase4_novel_predictions.csv"
VALIDATION_CSV = RESULTS_DIR / "phase4_validation.csv"
DBID_NAME_JSON = RESULTS_DIR / "dbid_to_name.json"
KNOWN_DDI_CSV = DATA_DIR / "drugbank_all_drug_drug_interactions.csv"

BATCH = 50_000

csv.field_size_limit(10 ** 7)


def log(msg: str) -> None:
    print(f"[build_db] {msg}", flush=True)


def fast_pragmas(con: sqlite3.Connection) -> None:
    con.execute("PRAGMA journal_mode = OFF")
    con.execute("PRAGMA synchronous = OFF")
    con.execute("PRAGMA temp_store = MEMORY")
    con.execute("PRAGMA cache_size = -200000")  # ~200 MB page cache


def stream_csv(path: Path):
    """Yield (header, row_iterator) for a CSV file."""
    f = path.open("r", encoding="utf-8", errors="replace", newline="")
    reader = csv.reader(f)
    header = next(reader)
    return f, header, reader


def load_signals(con: sqlite3.Connection) -> int:
    if not SIGNALS_CSV.exists():
        log(f"SKIP signals: {SIGNALS_CSV} not found")
        return 0
    con.execute("DROP TABLE IF EXISTS signals")
    con.execute(
        """CREATE TABLE signals (
            drug_a TEXT, drug_b TEXT, reaction TEXT,
            a INTEGER, b INTEGER, c INTEGER, d INTEGER,
            ror REAL, ci_low REAL, ci_high REAL
        )"""
    )
    f, header, reader = stream_csv(SIGNALS_CSV)
    log(f"signals header: {header}")
    ins = "INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?)"
    buf, total, t0 = [], 0, time.time()
    try:
        for row in reader:
            buf.append((
                row[0], row[1], row[2],
                int(row[3]), int(row[4]), int(row[5]), int(row[6]),
                float(row[7]), float(row[8]), float(row[9]),
            ))
            if len(buf) >= BATCH:
                con.executemany(ins, buf)
                total += len(buf)
                buf.clear()
                if total % 1_000_000 == 0:
                    log(f"  signals loaded: {total:,} ({time.time() - t0:.0f}s)")
        if buf:
            con.executemany(ins, buf)
            total += len(buf)
    finally:
        f.close()
    con.commit()
    log(f"signals: {total:,} rows loaded in {time.time() - t0:.0f}s; indexing...")
    con.execute("CREATE INDEX idx_sig_a ON signals(drug_a)")
    con.execute("CREATE INDEX idx_sig_b ON signals(drug_b)")
    con.execute("CREATE INDEX idx_sig_rxn ON signals(reaction)")
    con.execute("CREATE INDEX idx_sig_ror ON signals(ror)")
    con.execute("CREATE INDEX idx_sig_pair ON signals(drug_a, drug_b)")
    con.commit()
    log("signals indexed.")
    return total


def load_simple_csv(con: sqlite3.Connection, path: Path, table: str,
                    coltypes: dict, indexes: list[str]) -> int:
    if not path.exists():
        log(f"SKIP {table}: {path} not found")
        return 0
    con.execute(f"DROP TABLE IF EXISTS {table}")
    cols_sql = ", ".join(f"{c} {t}" for c, t in coltypes.items())
    con.execute(f"CREATE TABLE {table} ({cols_sql})")
    f, header, reader = stream_csv(path)
    cols = list(coltypes.keys())
    placeholders = ",".join("?" * len(cols))
    ins = f"INSERT INTO {table} VALUES ({placeholders})"

    # Map CSV header positions to our column order (case-insensitive).
    hmap = {h.strip().lower(): i for i, h in enumerate(header)}
    idx = []
    for c in cols:
        if c.lower() in hmap:
            idx.append(hmap[c.lower()])
        else:
            idx.append(None)

    def cast(val, t):
        if val is None or val == "":
            return None
        if t.startswith("INT"):
            try:
                return int(float(val))
            except ValueError:
                return None
        if t.startswith("REAL"):
            try:
                return float(val)
            except ValueError:
                return None
        return val

    buf, total = [], 0
    try:
        for row in reader:
            vals = []
            for pos, (c, t) in zip(idx, coltypes.items()):
                raw = row[pos] if pos is not None and pos < len(row) else None
                vals.append(cast(raw, t))
            buf.append(tuple(vals))
            if len(buf) >= BATCH:
                con.executemany(ins, buf)
                total += len(buf)
                buf.clear()
        if buf:
            con.executemany(ins, buf)
            total += len(buf)
    finally:
        f.close()
    con.commit()
    for col in indexes:
        con.execute(f"CREATE INDEX idx_{table}_{col} ON {table}({col})")
    con.commit()
    log(f"{table}: {total:,} rows")
    return total


def load_known_ddi(con: sqlite3.Connection) -> int:
    """Store DrugBank known DDIs with a canonical (sorted) pair key for O(1) flagging."""
    if not KNOWN_DDI_CSV.exists():
        log(f"SKIP known_ddi: {KNOWN_DDI_CSV} not found (per-row flagging disabled)")
        return 0
    con.execute("DROP TABLE IF EXISTS known_ddi")
    con.execute(
        """CREATE TABLE known_ddi (
            pair_key TEXT, drug_a_id TEXT, drug_b_id TEXT,
            drug_a_name TEXT, drug_b_name TEXT, description TEXT
        )"""
    )
    f, header, reader = stream_csv(KNOWN_DDI_CSV)
    hmap = {h.strip().lower(): i for i, h in enumerate(header)}
    ia = hmap.get("drug_a_id")
    ib = hmap.get("drug_b_id")
    ian = hmap.get("drug_a_name")
    ibn = hmap.get("drug_b_name")
    idesc = hmap.get("description")
    ins = "INSERT INTO known_ddi VALUES (?,?,?,?,?,?)"
    buf, total = [], 0
    try:
        for row in reader:
            try:
                a, b = row[ia], row[ib]
            except (IndexError, TypeError):
                continue
            key = "|".join(sorted([a, b]))
            buf.append((
                key, a, b,
                row[ian] if ian is not None and ian < len(row) else None,
                row[ibn] if ibn is not None and ibn < len(row) else None,
                row[idesc] if idesc is not None and idesc < len(row) else None,
            ))
            if len(buf) >= BATCH:
                con.executemany(ins, buf)
                total += len(buf)
                buf.clear()
        if buf:
            con.executemany(ins, buf)
            total += len(buf)
    finally:
        f.close()
    con.commit()
    con.execute("CREATE INDEX idx_known_key ON known_ddi(pair_key)")
    con.commit()
    log(f"known_ddi: {total:,} rows")
    return total


def load_drugs(con: sqlite3.Connection) -> int:
    if not DBID_NAME_JSON.exists():
        log(f"SKIP drugs: {DBID_NAME_JSON} not found")
        return 0
    con.execute("DROP TABLE IF EXISTS drugs")
    con.execute("CREATE TABLE drugs (dbid TEXT PRIMARY KEY, name TEXT)")
    with DBID_NAME_JSON.open(encoding="utf-8") as fh:
        mapping = json.load(fh)
    con.executemany("INSERT OR REPLACE INTO drugs VALUES (?,?)", list(mapping.items()))
    con.commit()
    con.execute("CREATE INDEX idx_drugs_name ON drugs(name)")
    con.commit()
    log(f"drugs: {len(mapping):,} rows")
    return len(mapping)


def build_drug_universe(con: sqlite3.Connection) -> int:
    """Distinct drugs that actually appear in signals, with resolved names, for fast search."""
    con.execute("DROP TABLE IF EXISTS signal_drugs")
    con.execute("CREATE TABLE signal_drugs (dbid TEXT PRIMARY KEY, name TEXT)")
    con.execute(
        """INSERT OR IGNORE INTO signal_drugs (dbid)
           SELECT drug_a FROM signals
           UNION
           SELECT drug_b FROM signals"""
    )
    con.execute(
        """UPDATE signal_drugs
           SET name = COALESCE((SELECT name FROM drugs WHERE drugs.dbid = signal_drugs.dbid), dbid)"""
    )
    con.commit()
    con.execute("CREATE INDEX idx_sigdrugs_name ON signal_drugs(name)")
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM signal_drugs").fetchone()[0]
    log(f"signal_drugs: {n:,} distinct drugs in signals")
    return n


def write_meta(con: sqlite3.Connection) -> None:
    con.execute("DROP TABLE IF EXISTS meta")
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    stats = {}
    for label, q in [
        ("n_signals", "SELECT COUNT(*) FROM signals"),
        ("n_pairs", "SELECT COUNT(DISTINCT drug_a || '|' || drug_b) FROM signals"),
        ("n_reactions", "SELECT COUNT(DISTINCT reaction) FROM signals"),
        ("n_drugs", "SELECT COUNT(*) FROM signal_drugs"),
    ]:
        try:
            stats[label] = str(con.execute(q).fetchone()[0])
        except sqlite3.OperationalError:
            stats[label] = "0"
    stats["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", list(stats.items()))
    con.commit()
    log(f"meta: {stats}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the DDI explorer SQLite DB.")
    ap.add_argument("--db", default=str(DB_PATH), help="Output SQLite path")
    ap.add_argument("--skip-signals", action="store_true",
                    help="Skip the large signals table (for quick rebuilds of small tables)")
    args = ap.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"building {db_path}")
    con = sqlite3.connect(str(db_path))
    fast_pragmas(con)

    if not args.skip_signals:
        load_signals(con)

    load_drugs(con)
    load_simple_csv(
        con, LABELED_CSV, "labeled_pairs",
        {"drug_a": "TEXT", "drug_b": "TEXT", "label": "INTEGER"},
        indexes=["drug_a", "drug_b"],
    )
    load_simple_csv(
        con, MATCH_CSV, "match_details",
        {"faers_name": "TEXT", "drugbank_id": "TEXT", "matched": "TEXT"},
        indexes=["faers_name", "drugbank_id"],
    )
    load_simple_csv(
        con, METRICS_CSV, "metrics",
        {"auc": "REAL", "accuracy": "REAL", "precision": "REAL",
         "recall": "REAL", "f1": "REAL", "fold": "INTEGER"},
        indexes=[],
    )
    load_simple_csv(
        con, NOVEL_CSV, "novel_predictions",
        {"drug_a": "TEXT", "drug_b": "TEXT", "predicted_probability": "REAL",
         "drug_a_name": "TEXT", "drug_b_name": "TEXT"},
        indexes=["drug_a", "drug_b"],
    )
    load_simple_csv(
        con, VALIDATION_CSV, "validation",
        {"k": "INTEGER", "hits": "INTEGER", "precision": "REAL"},
        indexes=[],
    )
    load_known_ddi(con)

    if not args.skip_signals or con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
    ).fetchone():
        build_drug_universe(con)

    write_meta(con)
    con.execute("PRAGMA optimize")
    con.close()
    log(f"DONE -> {db_path}")


if __name__ == "__main__":
    sys.exit(main())
