"""
Download FDA FAERS (drug adverse event) reports from OpenFDA bulk download
files.  Downloads partition zips, extracts JSON, flattens records to CSV
rows (one per report x drug), then zips.

Set YEAR_FILTER to a list of substrings (e.g. ["2024", "2025q1"]) to
download only matching partitions, or None / [] to download everything.

Supports --resume to continue from the last partition if interrupted.
"""
import argparse
import csv
import io
import json
import os
import time
import zipfile
from pathlib import Path

import requests

OUT_DIR = Path("data")
CSV_PATH = OUT_DIR / "faers_full.csv"
ZIP_PATH = OUT_DIR / "faers_full.csv.zip"
PROGRESS_PATH = OUT_DIR / ".faers_progress"
DOWNLOAD_INDEX = "https://api.fda.gov/download.json"
YEAR_FILTER = None  # None = download ALL partitions (full FAERS database)

COLUMNS = [
    "safetyreportid",
    "receivedate",
    "receiptdate",
    "serious",
    "seriousnessdeath",
    "seriousnesshospitalization",
    "seriousnessdisabling",
    "seriousnesslifethreatening",
    "seriousnessother",
    "occurcountry",
    "reportercountry",
    "reporter_qualification",
    "patient_age",
    "patient_age_unit",
    "patient_sex",
    "patient_death_date",
    "drug_characterization",
    "drug_name",
    "drug_dosage_form",
    "drug_route",
    "drug_indication",
    "drug_active_substance",
    "reactions",
]


def get_partition_urls(year_filter: list = None) -> list[str]:
    print("Fetching partition list...", flush=True)
    r = requests.get(DOWNLOAD_INDEX, timeout=60)
    r.raise_for_status()
    partitions = r.json()["results"]["drug"]["event"]["partitions"]
    if year_filter:
        partitions = [p for p in partitions
                      if any(y in p["file"] for y in year_filter)]
    urls = [p["file"] for p in partitions]
    total_mb = sum(float(p["size_mb"]) for p in partitions)
    print(f"  {len(urls)} files, {total_mb:.0f} MB compressed", flush=True)
    return urls


def download_partition(url: str) -> list[dict]:
    for attempt in range(5):
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()
            chunks = []
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                chunks.append(chunk)
            content = b"".join(chunks)
            break
        except Exception as e:
            if attempt == 4:
                print(f"    FAILED: {url} ({e})", flush=True)
                return []
            time.sleep(5 * (attempt + 1))

    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
        name = zf.namelist()[0]
        data = json.loads(zf.read(name))
        return data.get("results", [])
    except Exception as e:
        print(f"    PARSE ERROR: {url} ({e})", flush=True)
        return []


def flatten_record(rec: dict) -> list[dict]:
    patient = rec.get("patient", {}) or {}
    drugs = patient.get("drug", []) or []
    reactions = "|".join(
        r.get("reactionmeddrapt", "") for r in (patient.get("reaction", []) or [])
    )
    death = patient.get("patientdeath", {}) or {}
    primary = rec.get("primarysource", {}) or {}

    base = {
        "safetyreportid": rec.get("safetyreportid", ""),
        "receivedate": rec.get("receivedate", ""),
        "receiptdate": rec.get("receiptdate", ""),
        "serious": rec.get("serious", ""),
        "seriousnessdeath": rec.get("seriousnessdeath", ""),
        "seriousnesshospitalization": rec.get("seriousnesshospitalization", ""),
        "seriousnessdisabling": rec.get("seriousnessdisabling", ""),
        "seriousnesslifethreatening": rec.get("seriousnesslifethreatening", ""),
        "seriousnessother": rec.get("seriousnessother", ""),
        "occurcountry": rec.get("occurcountry", ""),
        "reportercountry": primary.get("reportercountry", ""),
        "reporter_qualification": primary.get("qualification", ""),
        "patient_age": patient.get("patientonsetage", ""),
        "patient_age_unit": patient.get("patientonsetageunit", ""),
        "patient_sex": patient.get("patientsex", ""),
        "patient_death_date": death.get("patientdeathdate", ""),
        "reactions": reactions,
    }

    if not drugs:
        return [{**base, "drug_characterization": "", "drug_name": "",
                 "drug_dosage_form": "", "drug_route": "",
                 "drug_indication": "", "drug_active_substance": ""}]

    rows = []
    for drug in drugs:
        active = drug.get("activesubstance", {}) or {}
        rows.append({
            **base,
            "drug_characterization": drug.get("drugcharacterization", ""),
            "drug_name": drug.get("medicinalproduct", ""),
            "drug_dosage_form": drug.get("drugdosageform", ""),
            "drug_route": drug.get("drugadministrationroute", ""),
            "drug_indication": drug.get("drugindication", ""),
            "drug_active_substance": active.get("activesubstancename", ""),
        })
    return rows


def save_progress(index: int):
    PROGRESS_PATH.write_text(str(index))


def load_progress() -> int:
    if PROGRESS_PATH.exists():
        try:
            return int(PROGRESS_PATH.read_text().strip())
        except ValueError:
            pass
    return -1


def main():
    parser = argparse.ArgumentParser(description="Download FAERS bulk data")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last completed partition")
    parser.add_argument("--years", nargs="*", default=None,
                        help="Year/quarter filter, e.g. --years 2024 2025q1. "
                             "Omit to download everything.")
    parser.add_argument("--no-zip", action="store_true",
                        help="Keep uncompressed CSV (skip zipping)")
    args = parser.parse_args()

    year_filter = args.years if args.years else YEAR_FILTER
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    urls = get_partition_urls(year_filter)

    start_index = 0
    if args.resume:
        last = load_progress()
        if last >= 0:
            start_index = last + 1
            print(f"Resuming from partition {start_index}/{len(urls)}")

    file_mode = "a" if (args.resume and start_index > 0 and CSV_PATH.exists()) else "w"
    write_header = file_mode == "w"

    total_rows = 0
    total_reports = 0
    t0 = time.time()

    with open(CSV_PATH, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        for i in range(start_index, len(urls)):
            url = urls[i]
            fname = url.split("/")[-1]
            records = download_partition(url)
            file_rows = 0
            for rec in records:
                for row in flatten_record(rec):
                    writer.writerow(row)
                    file_rows += 1
            total_reports += len(records)
            total_rows += file_rows

            elapsed = time.time() - t0
            done = i - start_index + 1
            pct = (i + 1) / len(urls) * 100
            eta = (elapsed / done) * (len(urls) - i - 1) if done else 0
            print(
                f"  [{i+1}/{len(urls)}, {pct:.0f}%] {fname}: "
                f"{len(records):,} reports, {file_rows:,} rows | "
                f"Total: {total_reports:,} reports, {total_rows:,} rows | "
                f"{elapsed:.0f}s, ~{eta:.0f}s ETA",
                flush=True,
            )
            f.flush()
            save_progress(i)

    elapsed = time.time() - t0
    size_mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"\nCSV: {total_rows:,} rows, {size_mb:.1f} MB, {elapsed:.0f}s", flush=True)

    if not args.no_zip:
        print("Zipping...", flush=True)
        with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(CSV_PATH, CSV_PATH.name)
        zip_mb = ZIP_PATH.stat().st_size / 1024 / 1024
        print(f"ZIP: {zip_mb:.1f} MB", flush=True)
        CSV_PATH.unlink()
        print(f"Done. Output: {ZIP_PATH}", flush=True)
    else:
        print(f"Done. Output: {CSV_PATH}", flush=True)

    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()


if __name__ == "__main__":
    main()
