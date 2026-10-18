"""Read raw cell values from an .xlsx workbook with the standard library.

Values are returned exactly as stored (shared strings resolved, numbers kept as
their original text) together with sheet name and A1 address, so every derived
record can cite its source cell. Formulas are not evaluated; the cached value
stored by the authoring application is returned and the formula is retained.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_ADDR = re.compile(r"^([A-Z]+)([0-9]+)$")


@dataclass(frozen=True)
class Cell:
    sheet: str
    address: str
    row: int
    col: int
    raw: str
    kind: str          # "n" number, "s" shared string, "str" formula string, "b", "inlineStr", "e"
    formula: str | None

    @property
    def number(self) -> float | None:
        if self.kind != "n":
            return None
        try:
            return float(self.raw)
        except ValueError:
            return None


def col_index(letters: str) -> int:
    value = 0
    for ch in letters:
        value = value * 26 + (ord(ch) - 64)
    return value


def col_letters(index: int) -> str:
    out = ""
    while index:
        index, rem = divmod(index - 1, 26)
        out = chr(65 + rem) + out
    return out


def read_workbook(path) -> dict[str, dict[tuple[int, int], Cell]]:
    """Return {sheet_name: {(row, col): Cell}} for every worksheet."""
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", _NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels.iter(f"{_REL_NS}Relationship")}
        book = ET.fromstring(zf.read("xl/workbook.xml"))
        sheets: dict[str, dict[tuple[int, int], Cell]] = {}
        for sheet in book.find("m:sheets", _NS):
            name = sheet.get("name")
            target = targets[sheet.get(f"{{{_NS['r']}}}id")].lstrip("/")
            target = target if target.startswith("xl/") else "xl/" + target
            cells: dict[tuple[int, int], Cell] = {}
            for c in ET.fromstring(zf.read(target)).iter(f"{{{_NS['m']}}}c"):
                addr = c.get("r")
                m = _ADDR.match(addr)
                if not m:
                    raise ValueError(f"unexpected cell address {addr!r} in {name}")
                kind = c.get("t", "n")
                v = c.find("m:v", _NS)
                f = c.find("m:f", _NS)
                if kind == "inlineStr":
                    raw = "".join(t.text or "" for t in c.iter(f"{{{_NS['m']}}}t"))
                elif v is None:
                    continue
                elif kind == "s":
                    raw = shared[int(v.text)]
                else:
                    raw = v.text or ""
                row, col = int(m.group(2)), col_index(m.group(1))
                cells[(row, col)] = Cell(name, addr, row, col, raw, kind,
                                         f.text if f is not None else None)
            sheets[name] = cells
        return sheets
