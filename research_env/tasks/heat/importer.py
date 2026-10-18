"""prepare(): convert the CORE workbook into audited, cell-lineage records.

Layout of the source workbook (verified by ``audit_layout``):

* ``Core Temp``, ``Skin Temp``: row 1 block headers, row 2 subject numbers,
  rows 3..62 minutes 1..60 (column B label equals row - 2). Columns C..Z are
  the first block (Rectal / YSI TSK), AB..AY the second (CORE / CORE TSK).
* ``Exercise Data``: same rows, but column B labels intervals such as ``0-1``
  or only their end minute (``26``). The end minute always equals row - 2, so a
  row is stamped at the end of its interval. Blocks: VO2 C.., HR AB.., RPE BA..,
  Wattage BZ.. .
* ``Environmental Conditions``: rows 3..10 at sparse intervals, ambient
  temperature C.., relative humidity AB.. .
* ``Demographics``: row per subject.

Subject numbers are 1..10 and 12..25 (24 people; 11 absent in every sheet).
Each record keeps the sheet and A1 address of the cell it came from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import quality
from .source import SOURCE, ensure_source, verify
from .xlsx import Cell, col_index, read_workbook

SUBJECTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25)
MINUTES = range(1, 61)
FIRST_DATA_ROW = 3

# (sheet, block header in row 1, first column letter, field name, role)
CHANNELS = (
    ("Core Temp", "Rectal", "C", "rectal_c", "reference"),
    ("Core Temp", "CORE", "AB", "vendor_core_c", "excluded"),
    ("Skin Temp", "YSI TSK", "C", "ysi_skin_c", "excluded"),
    ("Skin Temp", "CORE TSK", "AB", "device_skin_c", "input"),
    ("Exercise Data", "VO2", "C", "vo2_l_min", "excluded"),
    ("Exercise Data", "HR", "AB", "hr_bpm", "input"),
    ("Exercise Data", "RPE", "BA", "rpe", "excluded"),
    ("Exercise Data", "Wattage", "BZ", "work_rate_w", "input"),
    ("Environmental Conditions", "Ambient Temp. (C) ", "C", "ambient_c", "input"),
    ("Environmental Conditions", "Ambient RH (%)", "AB", "ambient_rh_pct", "input"),
)
DEMOGRAPHIC_COLUMNS = (
    ("B", "Sex (f,m)", "sex", "input"),
    ("C", "Age (yrs)", "age_yr", "input"),
    ("D", "Height (cm)", "height_cm", "input"),
    ("E", "Body Mass (kg)", "mass_kg", "input"),
    ("F", "BMI (kg/m^2)", "bmi_kg_m2", "input"),
    ("G", "Vo2peak (ml/kg/min)", "vo2peak_ml_kg_min", "excluded"),
)
EXCLUSION_REASONS = {
    "vendor_core_c": "vendor core-temperature estimate; never an input or a target",
    "ysi_skin_c": "laboratory reference thermistor skin temperature; not wearable-available in the primary tier",
    "vo2_l_min": "indirect calorimetry; excluded by brief",
    "rpe": "subjective rating collected by investigators; not a wearable channel",
    "vo2peak_ml_kg_min": "laboratory VO2 test; excluded by brief",
    "subcutaneous_fat": "ultrasound-derived; excluded by brief (sheet 'Subcu. Fat' not imported)",
}
INPUT_FIELDS = ("hr_bpm", "device_skin_c", "work_rate_w", "ambient_c", "ambient_rh_pct")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_json(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _end_minute(label: str) -> int:
    return int(str(label).strip().split("-")[-1])


def audit_layout(wb) -> dict:
    """Assert the structural assumptions above; return what was checked."""
    checks = {}
    for sheet, header, start, field, _ in CHANNELS:
        cells = wb[sheet]
        c0 = col_index(start)
        head = cells.get((1, c0))
        if head is None or head.raw != header:
            raise ValueError(f"{sheet}!{start}1 header {head.raw if head else None!r} != {header!r}")
        subjects = tuple(int(float(cells[(2, c0 + i)].raw)) for i in range(len(SUBJECTS)))
        if subjects != SUBJECTS:
            raise ValueError(f"{sheet} {header} subject order {subjects}")
        checks[f"{sheet}/{header}"] = {"header_cell": f"{start}1", "subjects_row": 2}
    for sheet in ("Core Temp", "Skin Temp", "Exercise Data"):
        cells = wb[sheet]
        for row in range(FIRST_DATA_ROW, FIRST_DATA_ROW + 60):
            if _end_minute(cells[(row, 2)].raw) != row - 2:
                raise ValueError(f"{sheet}!B{row} label {cells[(row, 2)].raw!r} != minute {row - 2}")
        checks[f"{sheet}/time"] = "column B end minute == row - 2 for rows 3..62"
    demo = wb["Demographics"]
    ids = tuple(int(float(demo[(r, 1)].raw)) for r in range(2, 26))
    if ids != SUBJECTS:
        raise ValueError(f"Demographics subject order {ids}")
    for letter, header, _, _ in DEMOGRAPHIC_COLUMNS:
        if demo[(1, col_index(letter))].raw != header:
            raise ValueError(f"Demographics!{letter}1 header mismatch")
    return checks


def _lineage(cell: Cell) -> dict:
    return {"sheet": cell.sheet, "cell": cell.address, "raw": cell.raw, "type": cell.kind}


def extract(wb) -> tuple[list, list, dict, list]:
    """Return (input_rows, reference_rows, context, quality_log)."""
    qlog: list = []
    series: dict[tuple[int, str], dict[int, dict]] = {}
    for sheet, _, start, field, role in CHANNELS:
        if role == "excluded":
            continue
        cells = wb[sheet]
        c0 = col_index(start)
        last_row = max(r for r, _ in cells)
        for i, subject in enumerate(SUBJECTS):
            col = c0 + i
            for row in range(FIRST_DATA_ROW, last_row + 1):
                minute = _end_minute(cells[(row, 2)].raw)
                cell = cells.get((row, col))
                if cell is None:
                    continue
                series.setdefault((subject, field), {})[minute] = {"cell": cell}
    # Quality rules operate on whole recorded series but only decide masks for
    # the cells they examine; masks are frozen before any model comparison.
    for (subject, field), points in series.items():
        for minute, point in points.items():
            point["value"], point["flags"] = quality.check(field, point["cell"], minute, points)
            for flag in point["flags"]:
                qlog.append({"subject": subject, "field": field, "minute": minute,
                             **_lineage(point["cell"]), "rule": flag,
                             "action": "masked (no replacement)"})
    inputs, references = [], []
    for subject in SUBJECTS:
        for field in INPUT_FIELDS:
            for minute, point in sorted(series.get((subject, field), {}).items()):
                inputs.append({"subject": subject, "minute": minute, "field": field,
                               "value": point["value"], "flags": point["flags"],
                               "source": _lineage(point["cell"])})
        for minute, point in sorted(series.get((subject, "rectal_c"), {}).items()):
            references.append({"subject": subject, "minute": minute, "rectal_c": point["value"],
                               "eligible": point["value"] is not None, "flags": point["flags"],
                               "source": _lineage(point["cell"])})
    demo = wb["Demographics"]
    context = {}
    for r, subject in zip(range(2, 26), SUBJECTS):
        entry = {}
        for letter, _, field, role in DEMOGRAPHIC_COLUMNS:
            if role == "excluded":
                continue
            cell = demo[(r, col_index(letter))]
            value = cell.raw.strip().upper() if field == "sex" else cell.number
            entry[field] = {"value": value, "source": _lineage(cell)}
        context[str(subject)] = entry
    return inputs, references, context, qlog


def recording_summary(references) -> dict:
    out = {}
    for subject in SUBJECTS:
        rows = [r for r in references if r["subject"] == subject]
        out[str(subject)] = {"first_minute": min(r["minute"] for r in rows),
                             "last_minute": max(r["minute"] for r in rows),
                             "reference_minutes": len(rows),
                             "eligible_reference_minutes": sum(r["eligible"] for r in rows)}
    return out


def _write_jsonl(path: Path, rows) -> str:
    text = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(source_root, output_root, split_manifest, allow_network: bool = True) -> dict:
    """Common v1 interface: build the audited dataset and return its manifest."""
    source_path, acquisition = ensure_source(Path(source_root), allow_network=allow_network)
    check = verify(source_path)
    if not check["verified"]:
        raise RuntimeError("source verification failed")
    split = json.loads(Path(split_manifest).read_text())
    split_people = sorted(p for group in split["groups"].values() for p in group)
    if split_people != sorted(SUBJECTS):
        raise ValueError("split manifest must assign every source subject exactly once")
    wb = read_workbook(source_path)
    layout = audit_layout(wb)
    inputs, references, context, qlog = extract(wb)
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    files = {
        "inputs.jsonl": _write_jsonl(out / "inputs.jsonl", inputs),
        "references.jsonl": _write_jsonl(out / "references.jsonl", references),
        "quality_log.jsonl": _write_jsonl(out / "quality_log.jsonl", qlog),
    }
    (out / "context.json").write_text(json.dumps(context, indent=1, sort_keys=True))
    files["context.json"] = hashlib.sha256((out / "context.json").read_bytes()).hexdigest()
    manifest = {
        "task": "heat-core-forecast-v1",
        "source": {**SOURCE, "local_path": str(source_path), "verification": check,
                   "acquisition": acquisition},
        "layout_checks": layout,
        "quality_rules": quality.RULES,
        "quality_rules_version": quality.VERSION,
        "split_manifest": {"path": str(split_manifest),
                           "sha256": hashlib.sha256(Path(split_manifest).read_bytes()).hexdigest(),
                           "groups": split["groups"]},
        "channels": [{"sheet": s, "header": h, "first_column": c, "field": f, "role": r,
                      **({"exclusion_reason": EXCLUSION_REASONS[f]} if r == "excluded" else {})}
                     for s, h, c, f, r in CHANNELS],
        "demographics": [{"column": c, "header": h, "field": f, "role": r,
                          **({"exclusion_reason": EXCLUSION_REASONS[f]} if r == "excluded" else {})}
                         for c, h, f, r in DEMOGRAPHIC_COLUMNS],
        "not_imported": {"Subcu. Fat": EXCLUSION_REASONS["subcutaneous_fat"]},
        "time_semantics": {
            "minute": "integer minutes since exercise start, 1..60",
            "Core Temp/Skin Temp": "sample labelled minute t is stamped t",
            "Exercise Data/Environmental Conditions": "interval (t-1, t] is stamped at its end minute t",
            "availability": "a value stamped t is available at cutoff c iff t <= c (current minute included)",
        },
        "counts": {"subjects": len(SUBJECTS), "input_values": len(inputs),
                   "masked_input_values": sum(r["value"] is None for r in inputs),
                   "reference_values": len(references),
                   "ineligible_reference_values": sum(not r["eligible"] for r in references),
                   "quality_events": len(qlog)},
        "recordings": recording_summary(references),
        "files": files,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest
