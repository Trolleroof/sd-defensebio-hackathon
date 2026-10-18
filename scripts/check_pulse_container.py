"""Explicit Docker gates using cached image, never paid inference."""
import argparse
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.core.worker import execute
from research_env.core.common import write

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image')
parser.add_argument('report')
args = parser.parse_args()
with tempfile.TemporaryDirectory() as temp:
    parent = Path(temp)
    work = parent/'public'
    work.mkdir()
    private = parent/'private-label.txt'
    private.write_text('HIDDEN_SENTINEL')
    checks = {}
    checks['numerical_training'] = execute(work, ['python', '-c',
        'import numpy as np;from sklearn.linear_model import Ridge;'
        'm=Ridge().fit(np.arange(10).reshape(-1,1),np.arange(10));print(m.predict([[4]])[0])'], timeout_s=60, image=args.image)
    checks['private_file_denied'] = execute(work, ['python', '-c', f'open({str(private)!r}).read()'], timeout_s=20, image=args.image)
    checks['network_denied'] = execute(work, ['python', '-c',
        'import socket;socket.create_connection(("1.1.1.1",443),timeout=2)'], timeout_s=20, image=args.image)
    checks['timeout'] = execute(work, ['python', '-c', 'while True: pass'], timeout_s=1, image=args.image)
    checks['memory_limit'] = execute(work, ['python', '-c', 'x=bytearray(512*1024*1024)'],
                                     timeout_s=20, memory_gb=.0625, image=args.image)
    passed = (checks['numerical_training']['execution_valid']
              and 'FileNotFoundError' in checks['private_file_denied']['stderr']
              and 'Network is unreachable' in checks['network_denied']['stderr']
              and checks['timeout']['failure_reason'] == 'wall_budget_exhausted'
              and checks['memory_limit']['exit_code'] == 137)
    write(args.report, {'all_passed': passed, 'checks': checks, 'image': args.image})
    print('Docker gates:', passed)
    if not passed:
        raise SystemExit(1)
