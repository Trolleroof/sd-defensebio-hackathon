"""Strict adapter for supplied wearable CSV recordings.

The adapter deliberately does not fetch, infer, or fabricate measurements. A caller
must provide the canonical CSV and identify its origin and heart-rate target source.
"""

import csv
import math
from collections import Counter
from pathlib import Path


CANONICAL_COLUMNS = ("participant_id", "session_id", "time_s", "hr_bpm", "role")
ROLES = {"support", "acquisition", "development", "test"}


def load_csv(path, *, origin, dataset, target_provenance, source_url=None):
    """Load canonical CSV into an episode-ready recording collection.

    Required columns are participant_id,session_id,time_s,hr_bpm,role. Time is
    session-relative seconds; each session may therefore start at zero. A session
    ID must identify one recording globally and have exactly one role. The caller
    supplies provenance instead of letting the adapter guess whether HR is raw,
    device-derived, or reference ECG.
    """
    if origin not in {"public", "local"}:
        raise ValueError("origin must be 'public' or 'local'")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset name is required")
    if not isinstance(target_provenance, str) or not target_provenance.strip():
        raise ValueError("explicit target_provenance is required")
    if origin == "public" and (not isinstance(source_url, str) or not source_url.strip()):
        raise ValueError("source_url is required for public data")

    sessions = {}
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(CANONICAL_COLUMNS):
            raise ValueError("CSV must contain exactly: " + ",".join(CANONICAL_COLUMNS))
        for line, row in enumerate(reader, start=2):
            participant = row["participant_id"].strip()
            session_id = row["session_id"].strip()
            role = row["role"].strip()
            if not participant or not session_id:
                raise ValueError(f"line {line}: participant_id and session_id are required")
            if role not in ROLES:
                raise ValueError(f"line {line}: unknown role {role!r}")
            try:
                time_s, hr_bpm = float(row["time_s"]), float(row["hr_bpm"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"line {line}: time_s and hr_bpm must be numeric") from exc
            if not math.isfinite(time_s) or not math.isfinite(hr_bpm) or hr_bpm <= 0:
                raise ValueError(f"line {line}: finite time and positive HR required")
            if session_id not in sessions:
                sessions[session_id] = {"id": session_id, "participant_id": participant,
                                        "role": role, "rows": [], "_times": set()}
            session = sessions[session_id]
            if session["participant_id"] != participant or session["role"] != role:
                raise ValueError(f"line {line}: session_id reused across participants or roles")
            if time_s in session["_times"]:
                raise ValueError(f"line {line}: duplicate timestamp in session {session_id!r}")
            session["_times"].add(time_s)
            session["rows"].append({"time_s": time_s, "hr_bpm": hr_bpm})

    if not sessions:
        raise ValueError("CSV contains no recordings")
    clean_sessions, by_role = [], Counter()
    for session_id in sorted(sessions):
        session = sessions[session_id]
        session["rows"].sort(key=lambda x: x["time_s"])
        times = [row["time_s"] for row in session["rows"]]
        session["coverage"] = {"rows": len(times), "start_s": times[0], "end_s": times[-1],
                                "duration_s": times[-1] - times[0]}
        session.pop("_times")
        by_role[session["role"]] += 1
        clean_sessions.append(session)
    return {
        "origin": origin,
        "dataset": dataset.strip(),
        "source_url": source_url,
        "target_provenance": target_provenance.strip(),
        "provenance": {"dataset": dataset.strip(), "source_url": source_url,
                       "target_provenance": target_provenance.strip()},
        "coverage": {"participants": len({s["participant_id"] for s in clean_sessions}),
                     "sessions": len(clean_sessions),
                     "rows": sum(len(s["rows"]) for s in clean_sessions),
                     "sessions_by_role": dict(sorted(by_role.items()))},
        "sessions": clean_sessions,
    }


def make_episode(collection, *, participant_id, episode_id, budgets=None):
    """Build and validate a single-participant episode from loaded recordings.

    CSV roles are preserved as-is. The caller must supply complete, non-overlapping
    source recordings for support/acquisition/development/test; this helper will
    not carve arbitrary windows or invent a split.
    """
    from .workspace import validate_episode

    participant_id = str(participant_id).strip()
    if not participant_id:
        raise ValueError("participant_id is required")
    chosen = [s for s in collection.get("sessions", [])
              if s.get("participant_id") == participant_id]
    if not chosen:
        raise ValueError(f"no recordings for participant {participant_id!r}")
    acquisition_count = sum(s["role"] == "acquisition" for s in chosen)
    episode = {
        "schema_version": 1,
        "id": str(episode_id),
        "origin": collection["origin"],
        "provenance": dict(collection.get("provenance", {})),
        "budgets": dict(budgets) if budgets is not None else {
            "acquisition": acquisition_count, "evaluations": 2},
        "sessions": [],
    }
    for source in chosen:
        session = {"id": source["id"], "participant_id": participant_id,
                   "role": source["role"], "description": source.get(
                       "description", f"Supplied {collection['dataset']} recording {source['id']}"),
                   "rows": [dict(row) for row in source["rows"]]}
        if session["role"] == "acquisition":
            session["cost"] = source.get("cost", 1)
        episode["sessions"].append(session)
    validate_episode(episode)
    return episode


def select_baseline(development_results, *, metric="mae_bpm"):
    """Choose the lowest-scoring valid development baseline, with stable ties.

    The input should contain development results only. No held-out score is
    consumed. Returns the selected model name or raises if none is usable.
    """
    candidates = []
    for name, result in development_results.items():
        value = result.get(metric) if result.get("valid") is True else None
        if type(value) in (int, float) and math.isfinite(value):
            candidates.append((float(value), str(name)))
    if not candidates:
        raise ValueError("no valid finite development baseline scores")
    return min(candidates)[1]
