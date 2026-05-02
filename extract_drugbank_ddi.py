"""
Extract drug-drug interactions from the DrugBank full database XML.

Produces data/drugbank_all_drug_drug_interactions.csv with columns:
    drug_a_id, drug_a_name, drug_b_id, drug_b_name, description

Usage:
    python extract_drugbank_ddi.py
    python extract_drugbank_ddi.py --input data/drugbank_all_full_database.xml.zip
"""

import argparse
import csv
import io
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

INPUT_PATH = Path("data/drugbank_all_full_database.xml.zip")
OUTPUT_PATH = Path("data/drugbank_all_drug_drug_interactions.csv")
NS = "{http://www.drugbank.ca}"


def extract(input_path: Path, output_path: Path):
    print(f"Parsing {input_path} ...")

    if str(input_path).endswith(".zip"):
        zf = zipfile.ZipFile(input_path)
        source = zf.open(zf.namelist()[0])
    else:
        source = open(input_path, "rb")

    seen = set()
    total_drugs = 0
    total_ddis = 0

    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["drug_a_id", "drug_a_name", "drug_b_id", "drug_b_name", "description"])

        for event, elem in ET.iterparse(source, events=("end",)):
            if elem.tag != f"{NS}drug":
                continue
            if elem.attrib.get("type") is None:
                continue

            dbid_el = elem.find(f"{NS}drugbank-id[@primary='true']")
            name_el = elem.find(f"{NS}name")
            if dbid_el is None or name_el is None:
                elem.clear()
                continue

            drug_a_id = dbid_el.text.strip()
            drug_a_name = name_el.text.strip() if name_el.text else ""
            total_drugs += 1

            interactions = elem.find(f"{NS}drug-interactions")
            if interactions is not None:
                for ddi in interactions.findall(f"{NS}drug-interaction"):
                    b_id_el = ddi.find(f"{NS}drugbank-id")
                    b_name_el = ddi.find(f"{NS}name")
                    desc_el = ddi.find(f"{NS}description")

                    if b_id_el is None:
                        continue
                    drug_b_id = b_id_el.text.strip()
                    drug_b_name = b_name_el.text.strip() if b_name_el is not None and b_name_el.text else ""
                    desc = desc_el.text.strip() if desc_el is not None and desc_el.text else ""

                    pair = tuple(sorted([drug_a_id, drug_b_id]))
                    if pair not in seen:
                        seen.add(pair)
                        writer.writerow([pair[0], "", pair[1], "", desc])
                        total_ddis += 1

            elem.clear()

            if total_drugs % 2000 == 0:
                print(f"  {total_drugs:,} drugs, {total_ddis:,} unique DDI pairs ...", flush=True)

    # Second pass: fill in names
    print(f"\n  Total: {total_drugs:,} drugs, {total_ddis:,} unique DDI pairs.")
    print(f"  Saved to {output_path}")

    # Re-read to add names
    print("  Adding drug names ...")
    if str(input_path).endswith(".zip"):
        zf2 = zipfile.ZipFile(input_path)
        source2 = zf2.open(zf2.namelist()[0])
    else:
        source2 = open(input_path, "rb")

    id_to_name = {}
    for event, elem in ET.iterparse(source2, events=("end",)):
        if elem.tag != f"{NS}drug":
            continue
        if elem.attrib.get("type") is None:
            elem.clear()
            continue
        dbid_el = elem.find(f"{NS}drugbank-id[@primary='true']")
        name_el = elem.find(f"{NS}name")
        if dbid_el is not None and name_el is not None and name_el.text:
            id_to_name[dbid_el.text.strip()] = name_el.text.strip()
        elem.clear()

    rows = []
    with open(output_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            row[1] = id_to_name.get(row[0], "")
            row[3] = id_to_name.get(row[2], "")
            rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Done. {len(rows):,} DDI pairs with names.")


def main():
    parser = argparse.ArgumentParser(description="Extract DDIs from DrugBank XML")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()
    extract(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
