"""Heat RL task family: independent hand-computed checks and adversarial candidates.

Most checks use a small synthetic prepared root whose values are chosen so the
expected numbers can be computed by hand. Checks that need the real audited
workbook skip unless ``.research-data/heat/prepared`` exists.
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research_env.tasks.heat import requests as rq
from research_env.tasks.heat import rl_pool
from research_env.tasks.heat.rl_pool import HeatFamily, TaskSpec, OUTER_SPLITS

REPO = Path(__file__).resolve().parents[1]
REAL = REPO / ".research-data" / "heat" / "prepared"
SPLIT = json.loads((REPO / "research_env/tasks/heat/split_manifest.json").read_text())["groups"]
SUBJECTS = sorted(s for g in SPLIT.values() for s in g)
SANDBOX = sys.platform == "darwin" and shutil.which("sandbox-exec") is not None
SENTINEL = 91.2345       # forbidden-channel value that must never reach a request


def base(subject):
    return round(36.5 + 0.03 * subject, 4)


def make_prepared(root: Path, *, forbidden=False, alter_after=None, alter_subject=None):
    """Synthetic prepared root: rectal(s, m) = base(s) + 0.02 m.

    Subject 9 stops at minute 56; subject 24 has an ineligible rectal minute 40.
    ``alter_after`` perturbs every input and reference of ``alter_subject``
    stamped after that minute.
    """
    root.mkdir(parents=True, exist_ok=True)
    inputs, refs, context = [], [], {}
    for s in SUBJECTS:
        last = 56 if s == 9 else 60
        for m in range(1, last + 1):
            bump = 5.0 if (alter_after is not None and s == alter_subject and m > alter_after) else 0.0
            inputs.append({"subject": s, "minute": m, "field": "device_skin_c", "value": 35.0 + 0.01 * m + bump})
            inputs.append({"subject": s, "minute": m, "field": "work_rate_w", "value": 150.0 + s + bump})
            if m % 15 in range(6, 11):
                inputs.append({"subject": s, "minute": m, "field": "hr_bpm", "value": 120.0 + m + bump})
            if m % 8 == 0:
                inputs.append({"subject": s, "minute": m, "field": "ambient_c", "value": 36.0 + bump})
                inputs.append({"subject": s, "minute": m, "field": "ambient_rh_pct", "value": 20.0 + bump})
            if forbidden:
                for f in ("vendor_core_c", "rectal_c", "ysi_skin_c"):
                    inputs.append({"subject": s, "minute": m, "field": f, "value": SENTINEL})
            eligible = not (s == 24 and m == 40)
            refs.append({"subject": s, "minute": m, "rectal_c": round(base(s) + 0.02 * m, 4) + bump if eligible else None,
                         "eligible": eligible, "flags": [] if eligible else ["isolated_dropout"]})
        context[str(s)] = {"sex": {"value": "M" if s % 2 else "F"}, "age_yr": {"value": 20.0 + s},
                           "height_cm": {"value": 170.0}, "mass_kg": {"value": 70.0}, "bmi_kg_m2": {"value": 24.2}}
    (root / "inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs))
    (root / "references.jsonl").write_text("".join(json.dumps(r) + "\n" for r in refs))
    (root / "context.json").write_text(json.dumps(context))
    (root / "dataset_manifest.json").write_text(json.dumps({"split_manifest": {"groups": SPLIT}}))
    return root


def spec(fit, feedback, reward, tier="wearable", profile="fast"):
    return rl_pool._spec("agent_training", 0, [str(x) for x in fit], [str(x) for x in feedback],
                         [str(x) for x in reward], profile, {"tier": tier, "horizons_min": [5, 10, 20]})


def write_artifact(root: Path, body: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "inference.py").write_text(body)
    return root


CONSTANT = """
def predict(artifact_dir, request):
    return [{"horizon_min": h, "mean_c": 38.0, "lower_c": 37.9, "upper_c": 38.1} for h in request["horizons_min"]]
"""
WIDE = CONSTANT.replace("37.9", "-12.0").replace("38.1", "88.0")


class Fixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.prepared = make_prepared(self.tmp / "prepared")
        self.family = HeatFamily()

    def tearDown(self):
        self._tmp.cleanup()

    def replay(self, artifact, manifest, profile="benchmark"):
        return self.family.score(artifact, manifest, backend="macos-development", image=None,
                                 wall_s=120, profile=profile)


class PoolAndIsolation(Fixture):
    def test_pool_person_sets(self):
        pool = self.family.build_pool(self.prepared, seed=3)
        train, dev, test = (set(map(str, SPLIT[g])) for g in ("train", "development", "private_test"))
        self.assertEqual(len(pool), 20)
        for t in pool:
            self.assertIn(t.outer_split, OUTER_SPLITS)
            f, fb, rw = map(set, (t.fit_people, t.feedback_people, t.reward_people))
            self.assertFalse(f & fb or f & rw or fb & rw)
            if t.outer_split == "agent_training":
                self.assertEqual((len(f), len(fb), len(rw)), (8, 3, 3))
                self.assertLessEqual(f | fb | rw, train)
            elif t.outer_split == "agent_development":
                self.assertEqual(rw, dev)
                self.assertLessEqual(f | fb, train)
            else:
                self.assertEqual(rw, test)
            if t.outer_split != "agent_final_evaluation":
                self.assertFalse((f | fb | rw) & test, "private-test person in training/development task")
        self.assertEqual([t.task_id for t in pool], [t.task_id for t in self.family.build_pool(self.prepared, 3)])
        self.assertNotEqual({t.task_id for t in pool}, {t.task_id for t in self.family.build_pool(self.prepared, 4)})
        self.assertEqual(len({t.task_id for t in pool}), len(pool))

    def test_workspace_holds_only_fit_people(self):
        s = spec([1, 2, 3], [5, 7], [8, 12])
        ws = self.tmp / "ws"
        self.family.stage_workspace(s, self.prepared, ws, prebuild=False)
        train = json.loads((ws / "data/train_manifest.json").read_text())
        minute1 = sorted(r["rectal_c"] for p in train["people"] for r in p["reference"] if r["minute"] == 1)
        self.assertEqual(minute1, sorted(round(base(x) + 0.02, 4) for x in (1, 2, 3)))
        self.assertEqual(len(train["people"]), 3)
        example = json.loads((ws / "data/example_request.json").read_text())
        self.assertEqual(example["cutoff_min"], 30)
        self.assertNotIn("calibration", example)
        names = {p.relative_to(ws).as_posix() for p in ws.rglob("*") if p.is_file()}
        self.assertFalse({n for n in names if "private" in n or n.endswith("-reward.json")})
        for p in train["people"]:
            self.assertTrue(p["person"].startswith("p-") and len(p["person"]) == 12)

    def test_private_manifest_refused_inside_workspace(self):
        s = spec([1, 2, 3], [5], [8])
        ws = self.tmp / "ws"
        self.family.stage_workspace(s, self.prepared, ws, prebuild=False)
        with self.assertRaises(ValueError):
            self.family.private_manifest(s, self.prepared, "reward", ws / "hidden")
        with self.assertRaises(ValueError):
            self.family.private_manifest(s, self.prepared, "fit", self.tmp / "priv")

    def test_split_drift_rejected(self):
        manifest = json.loads((self.prepared / "dataset_manifest.json").read_text())
        manifest["split_manifest"]["groups"]["train"][0] = 6
        (self.prepared / "dataset_manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.family.build_pool(self.prepared, 0)


class Causality(Fixture):
    def test_post_cutoff_alteration_leaves_requests_unchanged(self):
        s = spec([1, 2, 3], [5], [8, 12], profile="benchmark")
        a = json.loads(self.family.private_manifest(s, self.prepared, "reward", self.tmp / "a").read_text())
        altered = make_prepared(self.tmp / "altered", alter_after=30, alter_subject=8)
        b = json.loads(self.family.private_manifest(s, altered, "reward", self.tmp / "b").read_text())
        for ga, gb in zip(a["groups"], b["groups"]):
            for ra, rb in zip(ga, gb):
                if ra["cutoff_min"] <= 30:
                    self.assertEqual(json.dumps(ra, sort_keys=True), json.dumps(rb, sort_keys=True))
        changed = [r for g in b["groups"] for r in g if r["cutoff_min"] > 30]
        self.assertTrue(changed and a["groups"][0][-1] != b["groups"][0][-1])  # sanity: alteration visible later
        self.assertNotEqual(a["targets"], b["targets"])

    def test_forbidden_channels_never_enter_requests(self):
        dirty = make_prepared(self.tmp / "dirty", forbidden=True)
        for tier in ("wearable", "calibrated"):
            s = spec([1, 2, 3], [5], [8, 24], tier=tier, profile="benchmark")
            m = json.loads(self.family.private_manifest(s, dirty, "reward", self.tmp / tier).read_text())
            text = json.dumps(m["groups"])
            self.assertNotIn(str(SENTINEL), text)
            for subject, group in zip((8, 24), m["groups"]):
              rectal_values = {str(round(base(subject) + 0.02 * mm, 4)) for mm in range(2, 61)}
              for r in group:
                self.assertEqual(set(r["history"]), set(rq.HISTORY_FIELDS))
                self.assertFalse(any(v in json.dumps(r) for v in rectal_values), "rectal history leaked")
                if tier == "wearable":
                    self.assertNotIn("calibration", r)
                else:
                    self.assertEqual(r["calibration"]["rectal_c_at_minute"], 1)
            ws = self.tmp / f"ws-{tier}"
            self.family.stage_workspace(s, dirty, ws, prebuild=False)
            self.assertNotIn(str(SENTINEL), (ws / "data/train_manifest.json").read_text())

    def test_horizon_masking(self):
        s = spec([1, 2, 3], [5], [9, 24], profile="benchmark")
        m = json.loads(self.family.private_manifest(s, self.prepared, "reward", self.tmp / "p").read_text())
        reqs = {r["cutoff_min"]: r for r in m["groups"][0]}           # subject 9 (ends minute 56)
        self.assertEqual(reqs[40]["horizons_min"], [5, 10, 20])
        self.assertEqual(reqs[41]["horizons_min"], [5, 10])
        self.assertEqual(reqs[51]["horizons_min"], [5])
        self.assertEqual(reqs[55]["horizons_min"], [5])
        t9 = [t for t in m["targets"] if t["request_id"] == reqs[40]["request_id"]]
        self.assertEqual([(t["horizon_min"], t["eligible"]) for t in t9], [(5, True), (10, True), (20, False)])
        r24 = {r["cutoff_min"]: r for r in m["groups"][1]}
        t24 = [t for t in m["targets"] if t["request_id"] == r24[30]["request_id"] and t["horizon_min"] == 10]
        self.assertEqual((t24[0]["target_minute"], t24[0]["eligible"]), (40, False))
        fast = json.loads(self.family.private_manifest(spec([1], [5], [9], profile="fast"), self.prepared,
                                                       "reward", self.tmp / "f").read_text())
        self.assertEqual([r["cutoff_min"] for r in fast["groups"][0]], list(range(5, 56, 5)))


def hand_manifest(path: Path) -> Path:
    """Two people. A: cutoff 40. B: cutoffs 35 and 40 (see test for arithmetic)."""
    def req(rid, c):
        return {"request_id": rid, "tier": "wearable", "cutoff_min": c, "horizons_min": rq.requested_horizons(c),
                "history": {f: [] for f in rq.HISTORY_FIELDS}, "context": {}}
    def tgt(rid, person, h, ref):
        return {"request_id": rid, "person": person, "horizon_min": h, "rectal_c": ref, "eligible": True}
    targets = [tgt("a40", "A", 5, 38.2), tgt("a40", "A", 10, 38.0), tgt("a40", "A", 20, 37.5),
               tgt("b35", "B", 5, 38.0), tgt("b35", "B", 10, 38.4), tgt("b35", "B", 20, 37.6),
               tgt("b40", "B", 5, 38.4), tgt("b40", "B", 10, 38.0), tgt("b40", "B", 20, 38.0)]
    path.write_text(json.dumps({"schema": rl_pool.MANIFEST_SCHEMA, "task_id": "hand", "people_role": "reward",
                                "tier": "wearable", "profile": "benchmark", "cutoffs_min": [35, 40],
                                "groups": [[req("a40", 40)], [req("b35", 35), req("b40", 40)]],
                                "targets": targets}))
    return path


@unittest.skipUnless(SANDBOX, "macOS sandbox-exec worker backend required")
class ReplayAndMetrics(Fixture):
    def test_hand_computed_metrics(self):
        m = hand_manifest(self.tmp / "hand.json")
        r = self.replay(write_artifact(self.tmp / "const", CONSTANT), m)
        self.assertTrue(r["execution_valid"], r["failure_reason"])
        # A: |38-38.2|, |38-38.0|, |38-37.5| -> horizon MAEs .2, 0, .5 -> person .2333...
        # B: h5 (0+.4)/2=.2, h10 (.4+0)/2=.2, h20 (.4+0)/2=.2 -> person .2
        # primary = (.23333 + .2)/2 = .216667 (a pooled mean over all 9 errors would be .2111)
        self.assertAlmostEqual(r["primary_loss"], (0.7 / 3 + 0.2) / 2, places=9)
        a = r["metrics"]["per_person"]["A"]["horizons"]
        self.assertAlmostEqual(a["20"]["is90"], 0.2 + 20 * 0.4, places=9)       # 8.2
        self.assertAlmostEqual(a["5"]["is90"], 0.2 + 20 * 0.1, places=9)        # 2.2
        self.assertEqual(a["5"]["coverage90"], 0.0)
        self.assertAlmostEqual(a["10"]["bias_c"], 0.0, places=9)
        self.assertAlmostEqual(r["metrics"]["by_horizon"]["20"]["bias_c"], ((0.5) + (0.4 + 0.0) / 2) / 2, places=9)
        wide = self.replay(write_artifact(self.tmp / "wide", WIDE), m)
        self.assertAlmostEqual(wide["primary_loss"], r["primary_loss"], places=9)
        self.assertEqual(wide["metrics"]["uncertainty"]["coverage90"], 1.0)
        self.assertGreater(wide["metrics"]["uncertainty"]["is90"], r["metrics"]["uncertainty"]["is90"])

    def test_invalid_outputs_are_invalid_not_zero(self):
        m = hand_manifest(self.tmp / "hand.json")
        bodies = {
            "missing_horizon": CONSTANT.replace('for h in request["horizons_min"]]', 'for h in request["horizons_min"][:1]]'),
            "nan": CONSTANT.replace("38.0,", "float('nan'),"),
            "unordered": CONSTANT.replace("37.9", "38.5"),
            "extra_key": CONSTANT.replace('"upper_c": 38.1', '"upper_c": 38.1, "person": 1'),
            "not_list": "def predict(a, r):\n    return {'mean_c': 38.0}\n",
            "crash": "def predict(a, r):\n    raise RuntimeError('boom')\n",
            "string": CONSTANT.replace("38.0,", "'38.0',"),
        }
        for name, body in bodies.items():
            r = self.replay(write_artifact(self.tmp / name, body), m)
            self.assertFalse(r["execution_valid"], name)
            self.assertIsNone(r["primary_loss"], name)
            self.assertTrue(r["failure_reason"], name)
        (self.tmp / "empty").mkdir()
        self.assertFalse(self.replay(self.tmp / "empty", m)["execution_valid"])

    def test_participant_stdout_cannot_corrupt_protocol(self):
        m = hand_manifest(self.tmp / "hand.json")
        noisy = "print('debug noise')\n" + CONSTANT.replace("    return", "    print('x')\n    return")
        r = self.replay(write_artifact(self.tmp / "noisy", noisy), m)
        self.assertTrue(r["execution_valid"], r["failure_reason"])

    def test_leaky_candidate_cannot_read_controller_files(self):
        m = hand_manifest(self.tmp / "hand.json")
        leaky = f"open({str(m)!r}).read()\n" + CONSTANT
        r = self.replay(write_artifact(self.tmp / "leaky", leaky), m)
        self.assertFalse(r["execution_valid"])

    def test_state_resets_between_people(self):
        # Counts calls in a module global; B's first request would see 1 if the worker were shared.
        body = ("N = [0]\ndef predict(a, r):\n    N[0] += 1\n    v = 38.0 if N[0] == 1 else 30.0\n"
                "    return [{'horizon_min': h, 'mean_c': v, 'lower_c': v, 'upper_c': v} for h in r['horizons_min']]\n")
        r = self.replay(write_artifact(self.tmp / "counter", body), hand_manifest(self.tmp / "hand.json"))
        b5 = r["metrics"]["per_person"]["B"]["horizons"]["5"]["mae_c"]
        self.assertAlmostEqual(b5, (0.0 + 8.4) / 2, places=9)   # b35 fresh (38.0 vs 38.0), b40 second call (30 vs 38.4)

    def test_baseline_replay_matches_frozen_baseline(self):
        s = spec([1, 2, 3, 5], [7], [8, 12], profile="fast")
        ws = self.tmp / "ws"
        self.family.stage_workspace(s, self.prepared, ws)
        manifest = self.family.private_manifest(s, self.prepared, "reward", self.tmp / "priv")
        r = self.replay(ws / "artifact", manifest, profile="fast")
        self.assertTrue(r["execution_valid"], r["failure_reason"])
        # Elapsed-time on rectal = base + .02 m predicts mean(base_fit) + .02 m, so every
        # target's error is |mean(base_fit) - base(p)|, independent of minute and horizon.
        fit_mean = sum(base(x) for x in (1, 2, 3, 5)) / 4
        expected = sum(abs(fit_mean - base(p)) for p in (8, 12)) / 2
        self.assertAlmostEqual(r["primary_loss"], expected, places=6)
        self.assertAlmostEqual(self.family.baseline_loss(s, self.prepared), expected, places=6)
        built = subprocess.run([sys.executable, "tools/build_artifact.py", "baselines/ridge_lag.py", "--out", "ridge"],
                               cwd=ws, capture_output=True, text=True, timeout=120)
        self.assertEqual(built.returncode, 0, built.stderr)
        ridge = self.replay(ws / "ridge", manifest, profile="fast")
        self.assertTrue(ridge["execution_valid"], ridge["failure_reason"])
        self.assertTrue(math.isfinite(ridge["primary_loss"]))

    def test_fast_profile_subsamples_benchmark_manifest(self):
        s = spec([1, 2, 3, 5], [7], [8], profile="benchmark")
        manifest = self.family.private_manifest(s, self.prepared, "reward", self.tmp / "priv")
        art = write_artifact(self.tmp / "const", CONSTANT)
        full, fast = self.replay(art, manifest, "benchmark"), self.replay(art, manifest, "fast")
        self.assertGreater(full["metrics"]["requested_targets"], fast["metrics"]["requested_targets"])
        self.assertEqual(fast["metrics"]["requested_targets"], 3 * 8 + 2 + 2 + 1)  # cutoffs 5..40, 45, 50, 55


@unittest.skipUnless((REAL / "dataset_manifest.json").exists(), "real prepared heat data not present")
class RealData(unittest.TestCase):
    def test_audit_and_ak9_mask(self):
        manifest = json.loads((REAL / "dataset_manifest.json").read_text())
        self.assertTrue(manifest["source"]["verification"]["verified"])
        self.assertEqual(manifest["source"]["sha256"],
                         "499f546648822d99465c476702183cd3e37bdbaa6274de383f542708b052d531")
        self.assertEqual({g: len(v) for g, v in manifest["split_manifest"]["groups"].items()},
                         {"train": 14, "development": 5, "private_test": 5})
        log = [json.loads(l) for l in (REAL / "quality_log.jsonl").read_text().splitlines() if l]
        ak9 = [e for e in log if e["sheet"] == "Exercise Data" and e["cell"] == "AK9"]
        self.assertEqual((ak9[0]["raw"], ak9[0]["subject"], ak9[0]["action"]), ("1176", 10, "masked (no replacement)"))
        family = HeatFamily()
        dev = [t for t in family.build_pool(REAL, 0) if t.outer_split == "agent_development"][0]
        with tempfile.TemporaryDirectory() as tmp:
            m = json.loads(family.private_manifest(dev, REAL, "reward", Path(tmp)).read_text())
        self.assertNotIn("1176", json.dumps(m["groups"]))


if __name__ == "__main__":
    unittest.main()
