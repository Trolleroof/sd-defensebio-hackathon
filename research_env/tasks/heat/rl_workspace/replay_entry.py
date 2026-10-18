"""Copied into a fresh heat replay worker: only artifact/ and one person's causal requests.

Reads one JSON request per stdin line and writes one JSON response per line.
Participant stdout is redirected to stderr so it cannot corrupt the protocol.
"""
import importlib.util
import json
import sys
from pathlib import Path

protocol = sys.stdout
sys.stdout = sys.stderr
artifact = Path("artifact")
sys.path.insert(0, str(artifact.resolve()))
spec = importlib.util.spec_from_file_location("candidate", artifact / "inference.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for line in sys.stdin:
    request = json.loads(line)
    result = module.predict("artifact", request)
    protocol.write(json.dumps(result, allow_nan=False) + "\n")
    protocol.flush()
