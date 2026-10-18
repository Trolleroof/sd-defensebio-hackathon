import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark.executor import execute


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("sandbox-exec"),
                     "macOS sandbox-exec is required for isolation probes")
class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.support = [{"id": "cal", "rows": [{"time_s": 0, "hr_bpm": 100}]}]
        self.requests = [{"id": "held", "history": [{"time_s": 0, "hr_bpm": 110}],
                          "forecast_times_s": [30, 60]}]

    def test_runs_predictor_and_optional_fit(self):
        code = '''
level = 0
def fit(support):
    global level
    level = support[0]["rows"][0]["hr_bpm"]
def predict(history, forecast_times_s):
    start = history[-1]["hr_bpm"]
    return [{"time_s": t, "mean": start - level / 100,
             "lower": start - level / 100 - 2, "upper": start - level / 100 + 2}
            for t in forecast_times_s]
'''
        result = execute(code, self.support, self.requests)
        self.assertTrue(result["execution"]["valid"], result)
        self.assertEqual(result["execution"]["executor"], "macos-sandbox-exec")
        self.assertEqual([r["time_s"] for r in result["predictions"]["held"]], [30, 60])
        self.assertEqual(result["predictions"]["held"][0]["mean"], 109)

    def test_hidden_sentinel_is_not_readable(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            secret = Path(tmp) / "private-sentinel"
            secret.write_text("must-not-leak", encoding="utf-8")
            code = f'''\ndef predict(history, forecast_times_s):
    try:
        with open({str(secret)!r}) as f:
            f.read()
        allowed = 1
    except PermissionError:
        allowed = 0
    return [{{"time_s": t, "mean": allowed, "lower": 0, "upper": 1}}
            for t in forecast_times_s]
'''
            result = execute(code, self.support, self.requests)
            self.assertTrue(result["execution"]["valid"], result)
            self.assertEqual(result["predictions"]["held"][0]["mean"], 0)

    def test_controller_private_root_is_not_readable(self):
        with tempfile.TemporaryDirectory(dir="/private/var/tmp") as tmp:
            secret = Path(tmp) / "private-sentinel"
            secret.write_text("hidden-label", encoding="utf-8")
            code = f'''\ndef predict(history, forecast_times_s):
    try:
        with open({str(secret)!r}) as f:
            f.read()
        allowed = 1
    except PermissionError:
        allowed = 0
    return [{{"time_s": t, "mean": allowed, "lower": 0, "upper": 1}}
            for t in forecast_times_s]
'''
            result = execute(code, self.support, self.requests, forbidden_paths=[tmp])
            self.assertTrue(result["execution"]["valid"], result)
            self.assertEqual(result["predictions"]["held"][0]["mean"], 0)

    def test_network_is_denied(self):
        code = '''
import socket
def predict(history, forecast_times_s):
    try:
        socket.create_connection(("1.1.1.1", 443), timeout=1)
        denied = 0
    except PermissionError:
        denied = 1
    return [{"time_s": t, "mean": denied, "lower": 0, "upper": 1} for t in forecast_times_s]
'''
        result = execute(code, self.support, self.requests)
        self.assertTrue(result["execution"]["valid"], result)
        self.assertEqual(result["predictions"]["held"][0]["mean"], 1)

    def test_timeout_is_bounded(self):
        code = '''
def predict(history, forecast_times_s):
    while True:
        pass
'''
        result = execute(code, self.support, self.requests)
        self.assertFalse(result["execution"]["valid"], result)
        self.assertTrue(result["execution"].get("timed_out", False), result)


if __name__ == "__main__":
    unittest.main()
