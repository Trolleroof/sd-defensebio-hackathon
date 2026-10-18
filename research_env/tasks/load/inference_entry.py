"""Copied into a fresh worker; only the artifact and the current request are readable.

Stateless task: no state is injected or returned. One worker serves one person.
"""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('candidate', Path('artifact')/'inference.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for line in sys.stdin:
    result = module.predict('artifact', json.loads(line))
    sys.stdout.write(json.dumps(result, allow_nan=False)+'\n')
    sys.stdout.flush()
