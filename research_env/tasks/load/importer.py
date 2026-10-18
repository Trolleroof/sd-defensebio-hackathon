"""Trusted importer: EVOBREATH 2015 XLS -> person-condition records.

Implements the shared ``prepare(source_root, output_root, split_manifest)``
interface. Every numeric value keeps its source cell, original cell type and
original value. Text cells are parsed explicitly; a cell that is neither a
number nor a parseable numeric string is recorded as missing with a flag,
never guessed. Reference values are not reordered, smoothed or forced to be
monotonic in load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schema import (FIXED_CONTEXT, GRADE_FRACTION, LOAD_COLUMNS, SPEED_KM_H, SPEED_M_S,
                     TASK_ID, kcal_per_day_to_watts)

XLS_NAME = "Mateos2020_Dataset_WW2015.xls"
PDF_NAME = "Mateos2020_Dataset_WW2015.pdf"
EXPECTED_SHA256 = {
    XLS_NAME: "51b42232fd63e15f2566f1d1053c92fe2de88477c3c4f14ce95fdc75784c58da",
    PDF_NAME: "5410a21e0ab3bdbd96b7dc564dd8e4f1a7e948d158a70f9f5c673b3e7cac57f5",
}
SOURCE = {
    "dataset": "EVOBREATH 2015 walking and load-carriage",
    "doi": "10.17632/rhmdzmx6kk.1",
    "url": "https://data.mendeley.com/datasets/rhmdzmx6kk/1",
    "version": "1",
    "license": "CC BY 4.0",
    "author": "Ana Mateos (CENIEH)",
}
EXPECTED_HEADER = ["ID", "Sex", "Age", "Height", "Weight", "FL", "BIL", "FM", "FFM", "RMR",
                   "EE_WW00", "EE_WW05", "EE_WW10", "EE_WW15"]
# Source column -> (record field, unit as stated in the Variables sheet).
PERSON_FIELDS = {
    "Age": ("age_years", "years"),
    "Height": ("stature_cm", "cm"),
    "Weight": ("body_mass_kg", "kg"),
    "FL": ("femur_length_cm", "cm"),
    "BIL": ("biiliac_breadth_cm", "cm"),
    "FM": ("fat_mass_kg", "kg"),
    "FFM": ("fat_free_mass_kg", "kg"),
    "RMR": ("rmr_kcal_day", "kcal/day"),
}
QUALITY_RULES = {
    "text_cell_parsed": "Source stored the number as text; parsed with float() and original kept.",
    "missing_source_value": "Source cell is not numeric (e.g. '-'); target is ineligible, not imputed.",
    "body_composition_mismatch": "|FM + FFM - body mass| > 2.0 kg; lab-only fields, flag only.",
    "repeated_ffm_value": ("FFM recorded with more precision than the 0.1 kg used elsewhere and identical "
                           "to another person's FFM; possible copy error, flag only."),
    "low_walk_to_rest_ratio": "Unloaded walking EE / RMR < 2.0; plausible measurement issue, flag only, retained.",
    "nonmonotone_in_load": "Person's measured cost is not non-decreasing in load; retained as measured.",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cell_ref(row: int, col: int) -> str:
    letters = ""
    col += 1
    while col:
        col, rem = divmod(col - 1, 26)
        letters = chr(65 + rem) + letters
    return f"DATA!{letters}{row + 1}"


def parse_numeric(ctype: int, value):
    """Return (float | None, cell_type_name, flag | None). xlrd ctypes: 1 text, 2 number."""
    if ctype == 2:
        return float(value), "number", None
    if ctype == 1:
        text = str(value).strip()
        try:
            return float(text), "text", "text_cell_parsed"
        except ValueError:
            return None, "text", "missing_source_value"
    return None, f"ctype{ctype}", "missing_source_value"


def read_workbook(xls_path: Path) -> tuple[list[dict], dict]:
    import xlrd  # pinned in requirements.lock; only the trusted importer needs it

    book = xlrd.open_workbook(str(xls_path))
    sheet = book.sheet_by_name("DATA")
    header = [str(c.value) for c in sheet.row(0)]
    if header != EXPECTED_HEADER:
        raise ValueError(f"unexpected DATA header: {header}")
    variables = book.sheet_by_name("Variables")
    units = {str(variables.cell_value(r, 0)): str(variables.cell_value(r, 2))
             for r in range(1, variables.nrows)}
    for col in LOAD_COLUMNS.values():
        if units.get(col) != "kcal/day":
            raise ValueError(f"{col} unit is {units.get(col)!r}, expected kcal/day")
    rows = []
    for r in range(1, sheet.nrows):
        cells = {}
        for c, name in enumerate(header):
            cell = sheet.cell(r, c)
            cells[name] = {"cell": _cell_ref(r, c), "ctype": cell.ctype, "value": cell.value}
        rows.append(cells)
    return rows, units


def build_records(rows: list[dict], source_hash: str) -> tuple[list[dict], list[dict]]:
    people, records = [], []
    for cells in rows:
        pid = str(cells["ID"]["value"]).strip()
        sex = str(cells["Sex"]["value"]).strip()
        if sex not in ("M", "F"):
            raise ValueError(f"{pid}: unexpected sex code {sex!r}")
        person = {"source_person_id": pid, "sex": sex, "lineage": {}, "flags": []}
        for column, (field, unit) in PERSON_FIELDS.items():
            raw = cells[column]
            value, ctype, flag = parse_numeric(raw["ctype"], raw["value"])
            person[field] = value
            person["lineage"][field] = {"cell": raw["cell"], "cell_type": ctype,
                                        "original": raw["value"], "unit": unit}
            if flag:
                person["flags"].append(f"{flag}:{field}")
        person["rmr_w"] = (kcal_per_day_to_watts(person["rmr_kcal_day"])
                           if person["rmr_kcal_day"] is not None else None)
        people.append(person)
        for load, column in LOAD_COLUMNS.items():
            raw = cells[column]
            kcal, ctype, flag = parse_numeric(raw["ctype"], raw["value"])
            flags = [flag] if flag else []
            records.append({
                "record_id": f"{pid}:{int(load):02d}kg",
                "source_person_id": pid,
                "source_file_sha256": source_hash,
                "sex": sex,
                "body_mass_kg": person["body_mass_kg"],
                "stature_cm": person["stature_cm"],
                "age_years": person["age_years"],
                "load_kg": load,
                "speed_m_s": SPEED_M_S,
                "grade_fraction": GRADE_FRACTION,
                "reference_kcal_day": kcal,
                "reference_w": kcal_per_day_to_watts(kcal) if kcal is not None else None,
                "reference_convention": "gross",
                "eligible": kcal is not None,
                "lineage": {"reference": {"cell": raw["cell"], "cell_type": ctype,
                                          "original": raw["value"], "unit": "kcal/day"}},
                "flags": flags,
            })
    _flag_people(people, records)
    return people, records


def _flag_people(people: list[dict], records: list[dict]) -> None:
    ffm_counts: dict[float, int] = {}
    for p in people:
        if p["fat_free_mass_kg"] is not None:
            key = round(p["fat_free_mass_kg"], 2)
            ffm_counts[key] = ffm_counts.get(key, 0) + 1
    by_person: dict[str, list[dict]] = {}
    for r in records:
        by_person.setdefault(r["source_person_id"], []).append(r)
    for p in people:
        fm, ffm, mass = p["fat_mass_kg"], p["fat_free_mass_kg"], p["body_mass_kg"]
        if None not in (fm, ffm, mass) and abs(fm + ffm - mass) > 2.0:
            p["flags"].append("body_composition_mismatch")
        extra_precision = ffm is not None and abs(ffm * 10 - round(ffm * 10)) > 1e-6
        if extra_precision and ffm_counts[round(ffm, 2)] > 1:
            p["flags"].append("repeated_ffm_value")
        rows = sorted(by_person[p["source_person_id"]], key=lambda r: r["load_kg"])
        unloaded = rows[0]["reference_kcal_day"]
        if unloaded is not None and p["rmr_kcal_day"] and unloaded / p["rmr_kcal_day"] < 2.0:
            p["flags"].append("low_walk_to_rest_ratio")
        measured = [r["reference_kcal_day"] for r in rows if r["reference_kcal_day"] is not None]
        if any(b < a for a, b in zip(measured, measured[1:])):
            p["flags"].append("nonmonotone_in_load")
        person_flags = [f for f in p["flags"] if not f.startswith("text_cell_parsed")]
        for r in rows:
            r["flags"].extend(f"person:{f}" for f in person_flags)


def _write_jsonl(path: Path, items) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def prepare(source_root, output_root, split_manifest) -> dict:
    """Verify source hashes, parse, flag, assign frozen splits, and write a dataset manifest."""
    source_root, output_root = Path(source_root), Path(output_root)
    hashes = {}
    for name, expected in EXPECTED_SHA256.items():
        actual = sha256_file(source_root / name)
        if actual != expected:
            raise ValueError(f"{name}: sha256 {actual} != expected {expected}")
        hashes[name] = actual
    split = json.loads(Path(split_manifest).read_text()) if not isinstance(split_manifest, dict) else split_manifest
    rows, units = read_workbook(source_root / XLS_NAME)
    people, records = build_records(rows, hashes[XLS_NAME])
    assignment = {pid: name for name, ids in split["splits"].items() for pid in ids}
    source_ids = {p["source_person_id"] for p in people}
    if set(assignment) != source_ids:
        missing, extra = source_ids - set(assignment), set(assignment) - source_ids
        raise ValueError(f"split does not cover source people exactly: missing={sorted(missing)} extra={sorted(extra)}")
    for item in people + records:
        item["split"] = assignment[item["source_person_id"]]

    output_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_root / "records.jsonl", records)
    _write_jsonl(output_root / "people.jsonl", people)
    exclusions = [{"record_id": r["record_id"], "split": r["split"], "reason": "missing_source_value",
                   "source_cell": r["lineage"]["reference"]["cell"],
                   "original": r["lineage"]["reference"]["original"]}
                  for r in records if not r["eligible"]]
    flag_counts: dict[str, int] = {}
    for p in people:
        for f in p["flags"]:
            key = f.split(":")[0]
            flag_counts[key] = flag_counts.get(key, 0) + 1
    manifest = {
        "task_id": TASK_ID,
        "source": SOURCE,
        "source_files": hashes,
        "source_units": units,
        "conversion": {
            "reference_w": "kcal_per_day * 4184 / 86400 (rate equivalent; not energy of the 11-min bout)",
            "speed_m_s": f"{SPEED_KM_H} km/h / 3.6",
            "grade_fraction": "0 (level treadmill per protocol PDF)",
            "reference_convention": "gross (RMR not subtracted)",
        },
        "fixed_context": list(FIXED_CONTEXT),
        "split_id": split["split_id"],
        "counts": {
            "source_people": len(people),
            "condition_records": len(records),
            "eligible_records": sum(r["eligible"] for r in records),
            "excluded_records": len(exclusions),
            "people_by_split": {s: len(ids) for s, ids in split["splits"].items()},
            "eligible_records_by_split": {
                s: sum(r["eligible"] for r in records if r["split"] == s) for s in split["splits"]},
            "people_by_sex": {s: sum(p["sex"] == s for p in people) for s in ("F", "M")},
        },
        "exclusions": exclusions,
        "person_flag_counts": flag_counts,
        "quality_rules": QUALITY_RULES,
        "note": "Up to 192 condition observations from 48 people; condition records are not independent people.",
        "files": {name: sha256_file(output_root / name) for name in ("records.jsonl", "people.jsonl")},
    }
    (output_root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
