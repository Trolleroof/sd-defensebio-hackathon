"""Prepare the EVOBREATH 2015 load task from verified originals (optionally fetch)."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_env.tasks.load.data import fetch, prepare, verify_originals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('originals', help='directory holding the original XLS and protocol PDF')
parser.add_argument('output', help='prepared root (git-ignored, e.g. .research-data/load/prepared)')
parser.add_argument('--fetch', action='store_true', help='download missing originals from Mendeley')
parser.add_argument('--split', default=None, help='explicit split JSON covering all 48 people')
args = parser.parse_args()
hashes = fetch(args.originals) if args.fetch else verify_originals(args.originals)
manifest = prepare(Path(args.originals), Path(args.output), args.split)
print(json.dumps({'source_sha256': hashes, 'audit': manifest['audit']}, indent=1))
