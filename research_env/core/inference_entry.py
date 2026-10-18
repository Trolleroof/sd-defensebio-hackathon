"""Copied into fresh worker; only artifact and current causal request are readable."""
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('candidate', Path('artifact')/'inference.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
state = None
for line in sys.stdin:
    request = json.loads(line)
    request['state'] = state
    result = module.predict('artifact', request)
    state = result['state']
    sys.stdout.write(json.dumps(result, allow_nan=False)+'\n')
    sys.stdout.flush()
