import copy
import tempfile
import unittest
from pathlib import Path

from benchmark.examples import decision, episode, predict
from benchmark.judge import (PANEL_NAMES, RUBRIC_VERSION, build_blind_judge_packet,
                             save_judging_report, validate_verdict)
from benchmark.workspace import Workspace, read


class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.agent = self.root / "agent"
        self.private = self.root / "private"
        self.w = Workspace.create(episode(), self.private, self.agent)
        self.requests = read(self.agent / "requests.json")
        self.dev = next(r for r in self.requests if r["role"] == "development")
        self.test = next(r for r in self.requests if r["role"] == "test")

    def _packet(self):
        return build_blind_judge_packet(self.private / "state.json", self.agent)

    def _verdict(self, action_packet):
        refs = [e["index"] for e in action_packet["visible_evidence"]]
        return {"schema_version": 1, "rubric_version": RUBRIC_VERSION,
                "packet_sha256": action_packet["packet_sha256"],
                "event_index": action_packet["action"]["event_index"],
                "panels": {name: {"score": 4, "rationale": "Supported by the visible trace.",
                                  "evidence_indices": refs[:1]} for name in PANEL_NAMES}}

    def test_each_action_packet_excludes_its_own_and_future_results(self):
        self.w.evaluate({self.dev["id"]: predict(self.dev, 0)}, decision("Try a flat forecast"))
        self.w.acquire("long-bout", decision("Acquire the longer bout"))
        self.w.evaluate({self.dev["id"]: predict(self.dev, 10)}, decision("Use the observed decline"))
        self.w.submit("candidate", {self.test["id"]: predict(self.test, 10)}, decision("Select revised candidate"))
        packet = self._packet()
        actions = packet["event_packets"]
        self.assertEqual(len(actions), 4)
        self.assertEqual([p["action"]["event_index"] for p in actions], [0, 1, 2, 3])
        self.assertNotIn("development_feedback", [e["kind"] for e in actions[0]["visible_evidence"]])
        self.assertNotIn("acquired_recording", [e["kind"] for e in actions[1]["visible_evidence"]])
        self.assertIn("development_feedback", [e["kind"] for e in actions[2]["visible_evidence"]])
        self.assertIn("acquired_recording", [e["kind"] for e in actions[2]["visible_evidence"]])
        for action in actions:
            context = next(e['payload'] for e in action['visible_evidence'] if e['kind'] == 'public_task_context')
            for request in context['requests']:
                self.assertTrue(all(r['time_s'] <= 0 for r in request['history']))
                self.assertNotIn('rows', request)
        self.assertNotIn("final-score", str(packet))
        # The final selection packet still cannot see the test score produced
        # by submit; it sees only earlier support, acquisition, and dev feedback.
        final_packet = actions[3]
        self.assertNotIn("Result of action 3", str(final_packet["visible_evidence"]))

    def test_verdict_rejects_fabricated_and_hindsight_references(self):
        self.w.evaluate({self.dev["id"]: predict(self.dev, 0)}, decision("Evaluate baseline"))
        self.w.acquire("long-bout", decision("Acquire"))
        self.w.evaluate({self.dev["id"]: predict(self.dev, 10)}, decision("Evaluate after acquisition"))
        packets = self._packet()["event_packets"]
        valid = self._verdict(packets[1])
        self.assertEqual(validate_verdict(packets[1], valid)["event_index"], 1)
        bad = copy.deepcopy(valid)
        bad["panels"]["evidence_use"]["evidence_indices"] = [999]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_verdict(packets[1], bad)
        # A later action's feedback evidence index is not valid for an earlier
        # action even if the index exists elsewhere in the episode.
        later_only_index = max(e["index"] for e in packets[2]["visible_evidence"])
        self.assertGreater(later_only_index, max(e["index"] for e in packets[1]["visible_evidence"]))
        bad = copy.deepcopy(valid)
        bad["panels"]["feedback_integration"]["evidence_indices"] = [later_only_index]
        with self.assertRaisesRegex(ValueError, "unavailable"):
            validate_verdict(packets[1], bad)

    def test_verdict_requires_frozen_panels_and_integer_scores_in_range(self):
        self.w.evaluate({self.dev["id"]: predict(self.dev, 0)}, decision("Evaluate"))
        packet = self._packet()["event_packets"][0]
        valid = self._verdict(packet)
        bad = copy.deepcopy(valid)
        bad["panels"].pop("final_selection")
        with self.assertRaisesRegex(ValueError, "frozen rubric"):
            validate_verdict(packet, bad)
        bad = copy.deepcopy(valid)
        bad["panels"]["evidence_use"]["score"] = 5.5
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_verdict(packet, bad)

    def test_report_keeps_audit_separate_and_judge_pending_without_verdict(self):
        packet = self._packet()
        report = save_judging_report(self.root / "report.json", packet, state=self.private / "state.json")
        self.assertEqual(report["panels"]["external_llm_judge"]["status"], "pending")
        self.assertIn("deterministic_audit", report["panels"])
        self.assertEqual(read(self.root / "report.json"), report)

    def test_report_marks_external_judge_complete_only_after_validation(self):
        self.w.submit("candidate", {self.test["id"]: predict(self.test, 10)}, decision("Select"))
        packet = self._packet()
        verdicts = [self._verdict(p) for p in packet["event_packets"]]
        report = save_judging_report(self.root / "judged.json", packet, verdicts,
                                     state=self.private / "state.json")
        self.assertEqual(report["panels"]["external_llm_judge"]["status"], "complete")
        self.assertEqual(len(report["panels"]["external_llm_judge"]["verdicts"]), 1)
        self.assertIsNone(report["panels"]["execution_status"]["execution_valid"])
        self.assertFalse(report["panels"]["execution_status"]["final_code_verified"])

    def test_code_events_make_candidate_available_only_after_edit(self):
        from benchmark.agent import AgentSession

        agent = AgentSession(self.w)
        agent.act({"operation": "edit", "code": "def predict(history, forecast_times_s): return []",
                   "decision": decision("Write first candidate")})
        agent.act({"operation": "edit", "code": "def predict(history, forecast_times_s): return [] # revised",
                   "decision": decision("Revise candidate")})
        packets = self._packet()["event_packets"]
        self.assertNotIn("candidate_code", [e["kind"] for e in packets[0]["visible_evidence"]])
        codes = [e["payload"]["code"] for e in packets[1]["visible_evidence"] if e["kind"] == "candidate_code"]
        self.assertEqual(codes, ["def predict(history, forecast_times_s): return []"])


if __name__ == "__main__":
    unittest.main()
