import csv
import tempfile
import unittest
from pathlib import Path

from benchmark.baselines import baseline_codes
from benchmark.data import (CANONICAL_COLUMNS, load_csv, make_episode,
                            select_baseline)


class DataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "recordings.csv"

    def write_rows(self, rows):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, participant="p1", session="s1", time=0, hr=100, role="support"):
        return dict(participant_id=participant, session_id=session, time_s=time,
                    hr_bpm=hr, role=role)

    def load(self):
        return load_csv(self.path, origin="public", dataset="test corpus",
                        source_url="https://example.org/data", target_provenance="reference ECG")

    def test_loads_canonical_data_and_reports_session_coverage(self):
        self.write_rows([self.row(time=5, hr=96), self.row(time=0, hr=101),
                         self.row(session="s2", time=0, role="test", hr=98)])
        data = self.load()
        self.assertEqual(data["coverage"], {"participants": 1, "sessions": 2, "rows": 3,
                                            "sessions_by_role": {"support": 1, "test": 1}})
        first = data["sessions"][0]
        self.assertEqual(first["rows"], [{"time_s": 0.0, "hr_bpm": 101.0},
                                         {"time_s": 5.0, "hr_bpm": 96.0}])
        self.assertEqual(first["coverage"]["duration_s"], 5.0)
        self.assertEqual(data["target_provenance"], "reference ECG")

    def test_requires_explicit_provenance_and_public_source(self):
        self.write_rows([self.row()])
        with self.assertRaisesRegex(ValueError, "target_provenance"):
            load_csv(self.path, origin="local", dataset="local file", target_provenance=" ")
        with self.assertRaisesRegex(ValueError, "source_url"):
            load_csv(self.path, origin="public", dataset="public file", target_provenance="device HR")

    def test_rejects_duplicate_timestamps_and_session_role_overlap(self):
        self.write_rows([self.row(time=0), self.row(time=0, hr=101)])
        with self.assertRaisesRegex(ValueError, "duplicate timestamp"):
            self.load()
        self.write_rows([self.row(session="same", role="support"),
                         self.row(session="same", time=1, role="test")])
        with self.assertRaisesRegex(ValueError, "reused"):
            self.load()

    def test_rejects_malformed_or_unlabeled_data(self):
        self.path.write_text("participant_id,session_id,time_s,hr_bpm\np1,s1,0,90\n")
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.load()
        self.write_rows([self.row(role="")])
        with self.assertRaisesRegex(ValueError, "unknown role"):
            self.load()

    def test_baseline_source_is_executable_and_fit_uses_only_support(self):
        support = [{"rows": [{"time_s": 0, "hr_bpm": 120},
                             {"time_s": 10, "hr_bpm": 100},
                             {"time_s": 20, "hr_bpm": 80}]}]
        env = {"fitted_slope": 0.0}
        exec(baseline_codes()["linear_recovery"], env)
        env["fitted_slope"] = env["fit"](support)
        pred = env["predict"]([{"time_s": 0, "hr_bpm": 90}], [10, 20])
        self.assertAlmostEqual(env["fitted_slope"], -2.0)
        self.assertAlmostEqual(pred[0]["mean"], 70.0)
        self.assertEqual(pred[1]["time_s"], 20.0)
        p = {}
        exec(baseline_codes()["persistence"], p)
        p["fit"](support)
        self.assertEqual(p["predict"]([{"time_s": 0, "hr_bpm": 90}], [10])[0]["mean"], 90)

    def test_make_episode_selects_one_participant_and_adds_acquisition_defaults(self):
        rows = []
        for session, role in [("support", "support"), ("acq", "acquisition"),
                              ("dev", "development"), ("test", "test")]:
            rows.extend([self.row("p1", session, 0, 100, role),
                         self.row("p1", session, 1, 99, role)])
        rows.extend([self.row("p2", "other", 0, 110, "support"),
                     self.row("p2", "other", 1, 109, "support")])
        self.write_rows(rows)
        collection = self.load()
        ep = make_episode(collection, participant_id="p1", episode_id="case-1")
        self.assertEqual({s["participant_id"] for s in ep["sessions"]}, {"p1"})
        self.assertEqual({s["role"] for s in ep["sessions"]},
                         {"support", "acquisition", "development", "test"})
        acquisition = next(s for s in ep["sessions"] if s["role"] == "acquisition")
        self.assertEqual(acquisition["cost"], 1)
        self.assertEqual(ep["budgets"], {"acquisition": 1, "evaluations": 2})
        self.assertEqual(ep["provenance"]["target_provenance"], "reference ECG")

    def test_make_episode_rejects_missing_required_roles(self):
        self.write_rows([self.row("p1", "only", 0, 100, "support"),
                         self.row("p1", "only", 1, 99, "support")])
        with self.assertRaisesRegex(ValueError, "support, development and test"):
            make_episode(self.load(), participant_id="p1", episode_id="incomplete")

    def test_select_baseline_uses_only_valid_finite_development_scores(self):
        results = {"linear": {"valid": True, "mae_bpm": 4.0},
                   "persistence": {"valid": True, "mae_bpm": 2.0},
                   "broken": {"valid": False, "mae_bpm": 0.0},
                   "nan": {"valid": True, "mae_bpm": float("nan")}}
        self.assertEqual(select_baseline(results), "persistence")
        self.assertEqual(select_baseline({"z": {"valid": True, "mae_bpm": 2},
                                          "a": {"valid": True, "mae_bpm": 2}}), "a")
        with self.assertRaisesRegex(ValueError, "no valid"):
            select_baseline({"test": {"valid": False, "mae_bpm": 0}})


if __name__ == "__main__":
    unittest.main()
