# Trace judging

`benchmark.judge` prepares evidence-linked packets for an external language-model judge. It does not contact a provider or make a charged API call. The caller may submit each packet to an approved judge separately, then validate and save the returned structured verdict.

## Chronology and blindness

Call `build_blind_judge_packet(state, agent_workspace)` with the controller's `state.json` path (or its decoded object) and the agent-visible workspace directory. It returns one packet per recorded action. A packet contains that action's pre-action decision and only evidence that was visible before the action: support recordings, completed prior acquisitions, candidate code artifacts from prior edit/revert/evaluate-code actions, and completed development feedback. The target action's result and every later action are omitted. Candidate artifacts are included only when the controller state path identifies their private artifact directory. The private episode and `final-score.json` are never read for packets. Participant identifiers in session records are removed, and the episode is represented as `blind-episode`.

Packets are intentionally isolated. A judge should receive one packet at a time; sending the entire event trace alongside it would reintroduce hindsight. Each packet has integer evidence indices and a SHA-256 digest. A judge must cite indices from that packet, not invent identifiers or cite material from a later action.

## Frozen rubric and verdict schema

The rubric version is `recovery-adaptation-v1`; its text and 1–5 scale are embedded in the packet. It scores five panels: evidence use, experiment choice, feedback integration, action consistency, and final selection. Verdicts must include every panel, a nonempty rationale, and evidence indices. Scores must be integer values from 1 through 5. The verdict must echo the packet digest, rubric version, and action index. `validate_verdict(packet, verdict)` rejects schema drift, out-of-range scores, fabricated or duplicate evidence references, and references that were not available before that action.

The five panels are retained for every action packet so the rubric stays fixed across traces. When a criterion does not apply to a particular action, the judge should say so in its rationale and score only the evidence relevant to that action; scores still use the frozen 1–5 scale.

## Separate deterministic audit

`deterministic_audit` checks event ordering, decision-record completeness, packet count, and completion state. These checks establish trace integrity only. They do not judge whether an experiment was wise, whether feedback was interpreted well, or whether the selected model is scientifically sound. The semantic panel remains pending until validated external verdicts are supplied.

`save_judging_report(path, packet, verdicts=None, state=None)` writes a JSON report with separate `deterministic_audit`, `execution_status`, and `external_llm_judge` panels. With no verdicts, the LLM panel is explicitly `pending`; deterministic checks never stand in for it. With verdicts, it requires exactly one valid verdict for every action packet. Pass the controller `state.json` path to include `execution_valid` and `final_code_verified`; only these booleans are read from the private final-score receipt, never the score values. The report contains evidence-linked rationales and scores, not an overall clinical or deployment claim.

## Minimal use

```python
from benchmark.judge import build_blind_judge_packet, save_judging_report
from benchmark.workspace import read

state_path = "/path/to/private/state.json"
state = read(state_path)
packet = build_blind_judge_packet(state_path, "/path/to/agent-workspace")
# Send packet["event_packets"][i] to an explicitly configured external judge.
# Validate each response, then save all action verdicts:
# save_judging_report("judging-report.json", packet, verdicts, state=state_path)
save_judging_report("judging-report.json", packet, state=state_path)
```

No provider credentials are inspected or used by this module. Add an explicit integration only when provider, model, budget, and authorization are established by the caller.
