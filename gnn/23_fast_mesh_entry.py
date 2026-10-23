"""Fast entry for single cases and the original 12+4 suite."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/mesh_fast')
parser.add_argument('--case-name', default='fast_case')
args, _ = parser.parse_known_args()
forwarded = sys.argv[1:]
if '--output' not in forwarded:
    forwarded += ['--output', str(args.output)]
if '--case-name' not in forwarded:
    forwarded += ['--case-name', args.case_name]
sys.argv = [str(ROOT / '21_profile_mesh_generation.py'), '--fast',
            '--timing-output', str(args.output.resolve() / 'timings' / (args.case_name + '.json'))] + forwarded
runpy.run_path(str(ROOT / '21_profile_mesh_generation.py'), run_name='__main__')
