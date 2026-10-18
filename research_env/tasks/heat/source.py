"""Source registry for the CORE cycling heat workbook: identity, fetch, verify."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

SOURCE = {
    "source_id": "figshare-29120024-v1",
    "title": "Validity of the CORE Wearable Sensor During Constant-load Cycling Exercise in the Heat",
    "article_url": "https://figshare.com/articles/dataset/Validity_of_the_CORE_Wearable_Sensor_During_Constant-load_Cycling_Exercise_in_the_Heat_xlsx/29120024",
    "metadata_api": "https://api.figshare.com/v2/articles/29120024",
    "study_url": "https://pubmed.ncbi.nlm.nih.gov/40876129/",
    "doi": "10.6084/m9.figshare.29120024.v1",
    "version": 1,
    "license": "CC BY 4.0",
    "license_url": "https://creativecommons.org/licenses/by/4.0/",
    "attribution": "Andrew D'Lugos (Figshare, published 2025-05-21)",
    "download_url": "https://ndownloader.figshare.com/files/54711737",
    "file_name": "Validity of the CORE Wearable Sensor During Constant-load Cycling Exercise in the Heat.xlsx",
    "size_bytes": 84215,
    # md5 is the value published by the Figshare metadata API; sha256 was
    # computed locally from a file whose md5 matched it.
    "md5": "01214bfc835215fe1a5d3f5da1e9ad52",
    "sha256": "499f546648822d99465c476702183cd3e37bdbaa6274de383f542708b052d531",
}

LOCAL_CANDIDATES = (
    Path("/Users/nikhi/Documents/Army_Physiology_Research_20261017/evidence/heat/core_cycling_2025.xlsx"),
)
CACHE_NAME = "core_cycling_figshare_29120024_v1.xlsx"


def digests(path: Path) -> dict:
    data = Path(path).read_bytes()
    return {"size_bytes": len(data), "md5": hashlib.md5(data).hexdigest(),
            "sha256": hashlib.sha256(data).hexdigest()}


def verify(path: Path) -> dict:
    found = digests(path)
    ok = (found["size_bytes"] == SOURCE["size_bytes"] and found["md5"] == SOURCE["md5"]
          and found["sha256"] == SOURCE["sha256"])
    return {**found, "verified": ok}


def ensure_source(source_root: Path, allow_network: bool = True) -> tuple[Path, dict]:
    """Return a verified immutable copy of the workbook inside ``source_root``.

    Order: existing cache, known local evidence copies, then the original
    Figshare download URL. Every attempt and failure is recorded.
    """
    source_root = Path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    cached = source_root / CACHE_NAME
    attempts = []
    if cached.is_file():
        check = verify(cached)
        attempts.append({"origin": "cache", "path": str(cached), **check})
        if check["verified"]:
            return cached, {"acquired_from": "cache", "attempts": attempts}
        raise RuntimeError(f"cached source failed verification: {check}")
    for candidate in LOCAL_CANDIDATES:
        if candidate.is_file():
            check = verify(candidate)
            attempts.append({"origin": "local-evidence", "path": str(candidate), **check})
            if check["verified"]:
                shutil.copyfile(candidate, cached)
                cached.chmod(0o444)
                return cached, {"acquired_from": "local-evidence", "original_path": str(candidate),
                                "attempts": attempts}
        else:
            attempts.append({"origin": "local-evidence", "path": str(candidate), "error": "absent"})
    if not allow_network:
        raise RuntimeError(f"source unavailable offline; attempts={attempts}")
    tmp = cached.with_suffix(".part")
    try:
        with urllib.request.urlopen(SOURCE["download_url"], timeout=60) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except OSError as exc:
        attempts.append({"origin": "figshare", "url": SOURCE["download_url"],
                         "error": f"{type(exc).__name__}: {exc}"})
        raise RuntimeError(f"source download failed; attempts={attempts}") from exc
    check = verify(tmp)
    attempts.append({"origin": "figshare", "url": SOURCE["download_url"], **check})
    if not check["verified"]:
        tmp.unlink()
        raise RuntimeError(f"downloaded source failed verification; attempts={attempts}")
    tmp.rename(cached)
    cached.chmod(0o444)
    return cached, {"acquired_from": "figshare", "attempts": attempts}
