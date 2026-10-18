"""Evidence-linked, chronology-bounded packets for an external trace judge.

This module does not call a model. It prepares blind per-action packets, checks
structured verdicts returned by an external judge, and keeps those judgments
separate from deterministic controller audits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .workspace import read, write


RUBRIC_VERSION = "recovery-adaptation-v1"
RUBRIC = {
    "version": RUBRIC_VERSION,
    "scale": {"min": 1, "max": 5, "meaning": "1=unsupported/poor, 3=mixed, 5=strong and well supported"},
    "judge_guidance": "Judge each action only from its isolated packet and evidence available before it. Score only behavior observable for that action; when a panel is not observable, use 3 and explain the limitation rather than infer a result.",
    "panels": {
        "evidence_use": "Uses only evidence visible before the action and accurately distinguishes observation from inference.",
        "experiment_choice": "Chooses a permitted, informative, budget-aware experiment when one is available; otherwise gives a defensible reason to proceed.",
        "feedback_integration": "Uses prior evaluation feedback accurately and changes or retains the approach for a stated, evidence-linked reason.",
        "action_consistency": "The action follows from the recorded evidence, hypothesis, and expected result without unsupported claims.",
        "final_selection": "Selects a final candidate using only pre-submission evidence, while respecting uncertainty and held-out boundaries.",
    },
}
PANEL_NAMES = tuple(RUBRIC["panels"])


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _evidence(index: int, kind: str, description: str, payload: Any) -> dict[str, Any]:
    return {"index": index, "kind": kind, "description": description, "payload": payload}


def _blind_record(value: Any) -> Any:
    """Copy visible data while removing direct participant identifiers."""
    if isinstance(value, dict):
        return {key: _blind_record(item) for key, item in value.items() if key != "participant_id"}
    if isinstance(value, list):
        return [_blind_record(item) for item in value]
    return value


def _candidate_payload(control_root: Path | None, digest: Any) -> dict[str, Any]:
    """Load a frozen source artifact by verified content hash when available."""
    payload: dict[str, Any] = {"sha256": digest, "code_available": False}
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return payload
    code_path = control_root / "candidates" / f"{digest}.py" if control_root else None
    if code_path and code_path.is_file():
        code = code_path.read_text()
        if hashlib.sha256(code.encode()).hexdigest() == digest:
            payload.update({"code": code, "code_available": True})
    return payload


def build_blind_judge_packet(state: Mapping[str, Any] | str | Path,
                             workspace_dir: str | Path) -> dict[str, Any]:
    """Build one isolated packet per action, bounded to evidence then visible.

    ``state`` is the controller state object or its JSON path. The function
    reads only public workspace evidence and prior event results. It never reads
    the private episode or final-score files. A target action's result and all
    later actions are omitted from its packet, preventing hindsight leakage.
    """
    control_root = Path(state).resolve().parent if isinstance(state, (str, Path)) else None
    state_obj = _load_json(Path(state)) if isinstance(state, (str, Path)) else dict(state)
    events = state_obj.get("events")
    if not isinstance(events, list):
        raise ValueError("state must contain an events list")
    visible = Path(workspace_dir)
    support_path = visible / "support.json"
    brief_path = visible / "brief.json"
    requests_path = visible / "requests.json"
    support = _load_json(support_path) if support_path.exists() else []
    if not isinstance(support, list):
        raise ValueError("support.json must contain a list")
    brief = _load_json(brief_path) if brief_path.exists() else {}
    requests = _load_json(requests_path) if requests_path.exists() else []
    if not isinstance(brief, dict) or not isinstance(requests, list):
        raise ValueError("brief.json and requests.json have invalid shapes")
    # Public request metadata contains cutoff histories and forecast times but
    # no future labels. Drop role tags so held-out/development targets are not
    # called out in the judge context.
    public_requests = [{"id": r.get("id"), "history": r.get("history", []),
                        "forecast_times_s": r.get("forecast_times_s", [])}
                       for r in requests if isinstance(r, dict)]
    public_context = {"brief": _blind_record(brief), "requests": _blind_record(public_requests)}

    packets = []
    for target_pos, event in enumerate(events):
        if not isinstance(event, dict) or event.get("index") != target_pos:
            raise ValueError("events must have contiguous chronological indices")
        if event.get("status") not in ("started", "completed", "interrupted"):
            raise ValueError("unknown event status")
        decision = event.get("decision")
        if not isinstance(decision, dict) or set(decision) != {"evidence", "hypothesis", "action", "expected_result"}:
            raise ValueError("event is missing its pre-action decision")

        snapshot = event.get('public_snapshot')
        action_context = public_context
        action_support = support
        if isinstance(snapshot, dict):
            action_context = {'brief': _blind_record(snapshot['brief']),
                              'requests': _blind_record(snapshot['requests'])}
            action_support = snapshot['recordings']
        evidence: list[dict[str, Any]] = []
        evidence.append(_evidence(len(evidence), "public_task_context", "Public brief, budget, and forecast request metadata", action_context))
        evidence.append(_evidence(len(evidence), "support_recordings", "Recordings available from the decision", _blind_record(action_support)))
        # Completed prior operations become observations only after their own
        # action. Keep each observation tied to its event index and outcome.
        for prior_pos, prior in enumerate(events[:target_pos]):
            if prior.get("status") != "completed":
                continue
            operation = prior.get("operation")
            result = prior.get("result")
            if operation in ("edit", "revert", "evaluate_code"):
                digest = prior.get("artifact_sha256")
                payload = _candidate_payload(control_root, digest)
                evidence.append(_evidence(len(evidence), "candidate_code",
                                          f"Candidate artifact made available after action {prior['index']}", payload))
            if operation == "acquire":
                filename = None
                recording_id = result.get("recording") if isinstance(result, dict) else None
                if recording_id:
                    # Match the acquisition by order in the completed trace,
                    # never use the identifier as a path.
                    acquisition_no = sum(1 for e in events[:prior_pos + 1]
                                         if e.get("operation") == "acquire" and e.get("status") == "completed")
                    filename = visible / f"acquisition-{acquisition_no}.json"
                acquired_payload = _blind_record(_load_json(filename)) if filename and filename.exists() else {"unavailable": True}
                evidence.append(_evidence(len(evidence), "acquired_recording",
                                          f"Recording acquired after action {prior['index']}", acquired_payload))
                evidence.append(_evidence(len(evidence), "prior_action_result",
                                          f"Result of action {prior['index']}", {"operation": operation, "result": result}))
            elif operation in ("evaluate", "evaluate_code"):
                evidence.append(_evidence(len(evidence), "development_feedback",
                                          f"Development evaluation returned after action {prior['index']}", result))
            else:
                evidence.append(_evidence(len(evidence), "prior_action_result",
                                          f"Result of action {prior['index']}", {"operation": operation, "result": result}))

        action = {
            "event_index": target_pos,
            "operation": event.get("operation"),
            "decision": decision,
            "status": event.get("status"),
        }
        if event.get("operation") in ("edit", "revert", "evaluate_code", "submit_code", "checkpoint"):
            digest = event.get("artifact_sha256")
            artifact = _candidate_payload(control_root, digest)
            action["artifact_sha256"] = digest
            action["candidate_code"] = artifact.get("code")
            action["candidate_code_available"] = artifact["code_available"]
        packet = {
            "schema_version": 1,
            "rubric_version": RUBRIC_VERSION,
            "episode_id": "blind-episode",
            "action": action,
            "visible_evidence": evidence,
        }
        packet["packet_sha256"] = hashlib.sha256(_canonical(packet)).hexdigest()
        packets.append(packet)

    return {"schema_version": 1, "rubric": RUBRIC, "event_packets": packets}


def validate_verdict(packet: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a judge result against one action packet and the frozen rubric."""
    if set(verdict) != {"schema_version", "rubric_version", "packet_sha256", "event_index", "panels"}:
        raise ValueError("unexpected verdict fields")
    action = packet.get("action", {})
    if verdict["schema_version"] != 1 or verdict["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("unsupported verdict or rubric version")
    if verdict["packet_sha256"] != packet.get("packet_sha256"):
        raise ValueError("verdict packet hash mismatch")
    if type(verdict["event_index"]) is not int or verdict["event_index"] != action.get("event_index"):
        raise ValueError("verdict event index mismatch")
    panels = verdict["panels"]
    if not isinstance(panels, dict) or set(panels) != set(PANEL_NAMES):
        raise ValueError("all frozen rubric panels are required")
    allowed_refs = {item["index"] for item in packet.get("visible_evidence", [])}
    clean_panels = {}
    for name in PANEL_NAMES:
        panel = panels[name]
        if not isinstance(panel, dict) or set(panel) != {"score", "rationale", "evidence_indices"}:
            raise ValueError(f"invalid {name} panel")
        score = panel["score"]
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"{name} score must be an integer from 1 to 5")
        rationale, refs = panel["rationale"], panel["evidence_indices"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{name} rationale is required")
        if not isinstance(refs, list) or any(type(ref) is not int for ref in refs):
            raise ValueError(f"{name} evidence_indices must be integer indices")
        if len(refs) != len(set(refs)) or not set(refs) <= allowed_refs:
            raise ValueError(f"{name} cites evidence unavailable before this action")
        clean_panels[name] = {"score": score, "rationale": rationale.strip(), "evidence_indices": refs.copy()}
    return {"schema_version": 1, "rubric_version": RUBRIC_VERSION,
            "packet_sha256": verdict["packet_sha256"], "event_index": verdict["event_index"],
            "panels": clean_panels}


def deterministic_audit(state: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical trace checks, reported separately from semantic judgment."""
    events = state.get("events", [])
    problems = []
    if len(events) != len(packet.get("event_packets", [])):
        problems.append("packet count does not match event count")
    interrupted = []
    for i, event in enumerate(events):
        if event.get("index") != i:
            problems.append(f"event {i} has a non-contiguous index")
        if event.get("status") != "completed":
            problems.append(f"event {i} is not completed")
        if event.get("interrupted") or event.get("status") == "interrupted":
            interrupted.append(i)
            problems.append(f"event {i} is interrupted")
        if not isinstance(event.get("decision"), dict) or set(event["decision"]) != {"evidence", "hypothesis", "action", "expected_result"}:
            problems.append(f"event {i} lacks a complete decision record")
    return {"status": "pass" if not problems else "findings", "checks": [
        "event order and action packet count", "pre-action decision record completeness", "event completion state"],
        "findings": problems,
        "interrupted_event_indices": interrupted,
        "scope": "Mechanical trace integrity only; this does not score reasoning quality."}


def save_judging_report(path: str | Path, packet: Mapping[str, Any],
                        verdicts: list[Mapping[str, Any]] | None = None,
                        state: Mapping[str, Any] | str | Path | None = None,
                        judge_model: str | None = None) -> dict[str, Any]:
    """Write a report with independent deterministic and LLM-judge panels."""
    state_path = Path(state).resolve() if isinstance(state, (str, Path)) else None
    state_obj = _load_json(state_path) if state_path else dict(state or {})
    event_packets = packet.get("event_packets", [])
    validated: list[dict[str, Any]] = []
    if verdicts is not None:
        if len(verdicts) != len(event_packets):
            raise ValueError("one verdict is required for every action packet")
        validated = [validate_verdict(p, v) for p, v in zip(event_packets, verdicts)]
    execution_status = {"execution_valid": None, "final_code_verified": None,
                        "source": "private final execution receipt, status fields only"}
    if state_path:
        score_path = state_path.parent / "final-score.json"
        if score_path.is_file():
            final_score = _load_json(score_path)
            execution = final_score.get("execution")
            if isinstance(execution, dict) and type(execution.get("valid")) is bool:
                execution_status["execution_valid"] = execution["valid"]
            if type(final_score.get("code_execution_verified")) is bool:
                execution_status["final_code_verified"] = final_score["code_execution_verified"]
    report = {
        "schema_version": 1,
        "rubric_version": RUBRIC_VERSION,
        "packet_sha256": hashlib.sha256(_canonical(packet)).hexdigest(),
        "panels": {
            "deterministic_audit": deterministic_audit(state_obj, packet),
            "execution_status": execution_status,
            "external_llm_judge": {"status": "complete" if verdicts is not None else "pending",
                                   "verdicts": validated,
                                   "note": "External semantic judgment is separate from deterministic audit."},
        },
    }
    if judge_model is not None:
        report["panels"]["external_llm_judge"]["model"] = judge_model
    write(path, report)
    return report
