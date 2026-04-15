"""
Download the complete FDA HFCS (CAERS) adverse event database for dietary
supplements (industry_code 54).  Flattens each report × product × reaction
into one row and saves as a CSV, then zips it.
"""
import csv
import io
import time
import zipfile
from pathlib import Path

import requests

OPENFDA_URL = "https://api.fda.gov/food/event.json"
INDUSTRY_CODE = "54"
PAGE_SIZE = 1000

OUT_DIR = Path("data")
CSV_PATH = OUT_DIR / "hfcs_full.csv"
ZIP_PATH = OUT_DIR / "hfcs_full.csv.zip"

DATE_WINDOWS = [
    ("20040101", "20101231"),
    ("20110101", "20131231"),
    ("20140101", "20151231"),
    ("20160101", "20171231"),
    ("20180101", "20191231"),
    ("20200101", "20211231"),
    ("20220101", "20231231"),
    ("20240101", "20261231"),
]

COLUMNS = [
    "report_number",
    "date_created",
    "date_started",
    "outcomes",
    "consumer_age",
    "consumer_age_unit",
    "consumer_gender",
    "product_role",
    "product_name_brand",
    "product_industry_code",
    "product_industry_name",
    "reactions",
]


def pull_all() -> list[dict]:
    """Pull every supplement adverse-event record via date-range windowing."""
    first = requests.get(OPENFDA_URL, params={
        "search": f"products.industry_code:{INDUSTRY_CODE}",
        "limit": 1,
    }, timeout=30)
    total = first.json()["meta"]["results"]["total"]
    print(f"Total records available: {total:,}", flush=True)

    all_records: list[dict] = []
    seen = set()

    for win_start, win_end in DATE_WINDOWS:
        search_q = (
            f"products.industry_code:{INDUSTRY_CODE} "
            f"AND date_created:[{win_start} TO {win_end}]"
        )
        skip = 0
        win_new = 0
        retries = 0

        while True:
            params = {"search": search_q, "limit": PAGE_SIZE, "skip": skip}
            try:
                resp = requests.get(OPENFDA_URL, params=params, timeout=30)
                if resp.status_code == 404:
                    break
                resp.raise_for_status()
                data = resp.json()
                retries = 0
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    break
                retries += 1
                if retries > 5:
                    print(f"  Giving up on {win_start}-{win_end} at skip={skip}", flush=True)
                    break
                wait = 5 * retries
                print(f"  Error at skip={skip}: {e}. Retry {retries}/5 in {wait}s...", flush=True)
                time.sleep(wait)
                continue
            except Exception as e:
                retries += 1
                if retries > 5:
                    print(f"  Giving up on {win_start}-{win_end} at skip={skip}", flush=True)
                    break
                wait = 5 * retries
                print(f"  Error at skip={skip}: {e}. Retry {retries}/5 in {wait}s...", flush=True)
                time.sleep(wait)
                continue

            results = data.get("results", [])
            if not results:
                break

            for rec in results:
                rn = rec.get("report_number", "")
                if rn and rn not in seen:
                    seen.add(rn)
                    all_records.append(rec)
                    win_new += 1

            skip += PAGE_SIZE
            if skip >= 25000:
                print(f"  WARNING: {win_start}-{win_end} hit 25K skip limit", flush=True)
                break
            time.sleep(0.15)

        print(
            f"  Window {win_start}-{win_end}: +{win_new:,} "
            f"(total {len(all_records):,})",
            flush=True,
        )

    print(f"Pull complete: {len(all_records):,} unique reports.", flush=True)
    return all_records


def flatten(records: list[dict]) -> list[dict]:
    """Flatten each report into one row per (report, product).

    Reactions and outcomes are joined with '|' so every field from the API
    is preserved in a single flat row.
    """
    rows = []
    for rec in records:
        report_number = rec.get("report_number", "")
        date_created = rec.get("date_created", "")
        date_started = rec.get("date_started", "")
        outcomes = "|".join(rec.get("outcomes", []))
        reactions = "|".join(rec.get("reactions", []))

        consumer = rec.get("consumer", {})
        consumer_age = consumer.get("age", "")
        consumer_age_unit = consumer.get("age_unit", "")
        consumer_gender = consumer.get("gender", "")

        products = rec.get("products", [])
        if not products:
            rows.append({
                "report_number": report_number,
                "date_created": date_created,
                "date_started": date_started,
                "outcomes": outcomes,
                "consumer_age": consumer_age,
                "consumer_age_unit": consumer_age_unit,
                "consumer_gender": consumer_gender,
                "product_role": "",
                "product_name_brand": "",
                "product_industry_code": "",
                "product_industry_name": "",
                "reactions": reactions,
            })
            continue

        for prod in products:
            rows.append({
                "report_number": report_number,
                "date_created": date_created,
                "date_started": date_started,
                "outcomes": outcomes,
                "consumer_age": consumer_age,
                "consumer_age_unit": consumer_age_unit,
                "consumer_gender": consumer_gender,
                "product_role": prod.get("role", ""),
                "product_name_brand": prod.get("name_brand", ""),
                "product_industry_code": prod.get("industry_code", ""),
                "product_industry_name": prod.get("industry_name", ""),
                "reactions": reactions,
            })
    return rows


def save_csv(rows: list[dict]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(rows):,} rows to {CSV_PATH}...", flush=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    size_mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"  CSV size: {size_mb:.1f} MB", flush=True)


def zip_csv():
    print(f"Zipping to {ZIP_PATH}...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV_PATH, CSV_PATH.name)
    size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"  ZIP size: {size_mb:.1f} MB", flush=True)


def main():
    records = pull_all()
    rows = flatten(records)
    save_csv(rows)
    zip_csv()
    CSV_PATH.unlink()
    print(f"Done. CSV deleted, zip at {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
