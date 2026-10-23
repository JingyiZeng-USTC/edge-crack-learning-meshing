"""Self-contained final workflow: learned size -> fast GNN mesh -> FCNN-SM."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(arguments, cwd=ROOT):
    print('\nRUN: ' + subprocess.list2cmdline([str(x) for x in arguments]), flush=True)
    subprocess.run([str(x) for x in arguments], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge', default='left', choices=['left', 'right', 'bottom', 'top'])
    parser.add_argument('--origin', type=float, default=1.07)
    parser.add_argument('--length', type=float, default=.32)
    parser.add_argument('--angle-deg', type=float, default=11.)
    parser.add_argument('--geometry-json', type=Path)
    parser.add_argument('--case-name', default='final_straight')
    parser.add_argument('--seed', type=int, default=74032)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=2560)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--mesh-only', action='store_true', help='Generate and export the mesh; do not train FCNN-SM')
    parser.add_argument('--suite', action='store_true', help='Run the original 12 straight + 4 curved cases')
    parser.add_argument('--size-model', type=Path, default=ROOT / 'module30_checkpoints/single_straight_crack_local_v4.pt')
    parser.add_argument('--gnn-model', type=Path, default=ROOT / 'gnn/artifacts/single_straight_200_strict_model.pt')
    args = parser.parse_args()
    if args.epochs < 1 or args.samples < 3:
        parser.error('epochs >=1 and samples >=3 are required')
    if not args.case_name or Path(args.case_name).name != args.case_name or any(c in args.case_name for c in '<>:"/\\|?*') or args.case_name in ['.', '..']:
        parser.error('case-name must be one safe file name, without a path')
    args.output = args.output.resolve()
    mesh_root = args.output / ('suite_12straight_4curve' if args.suite else args.case_name)
    if mesh_root.exists() and any(mesh_root.iterdir()):
        parser.error(f'Output already exists: {mesh_root}. Choose another --case-name or --output; existing results are preserved.')
    if not args.size_model.is_file() or not args.gnn_model.is_file():
        parser.error('Missing pretrained size/GNN checkpoint')
    if args.suite:
        if args.size_model.resolve() != (ROOT / 'module30_checkpoints/single_straight_crack_local_v4.pt').resolve() or args.gnn_model.resolve() != (ROOT / 'gnn/artifacts/single_straight_200_strict_model.pt').resolve():
            parser.error('Suite uses the packaged final checkpoints; use single cases for custom checkpoints')
        run([sys.executable, '-B', '-u', ROOT / 'gnn/17_test_quality20_straight_curved.py',
             '--pipeline', ROOT / 'gnn/23_fast_mesh_entry.py', '--output', mesh_root, '--device', args.device], ROOT / 'gnn')
        if not args.mesh_only:
            run([sys.executable, '-B', '-u', ROOT / 'gnn/18_run_quality20_fcnn_suite.py',
                 '--suite-root', mesh_root, '--python', sys.executable, '--epochs', args.epochs,
                 '--samples', args.samples, '--device', args.device], ROOT / 'gnn')
        else:
            run([sys.executable, '-B', ROOT / 'gnn/09_export_gnn_mesh_for_fcnn.py',
                 '--recovery', mesh_root / 'meshes', '--candidates', mesh_root / 'candidates',
                 '--output', mesh_root / 'fcnn_meshes'], ROOT / 'gnn')
    else:
        geometry_args = (['--geometry-json', args.geometry_json.resolve()] if args.geometry_json else
                         ['--edge', args.edge, '--origin', args.origin, '--length', args.length, '--angle-deg', args.angle_deg])
        run([sys.executable, '-B', '-u', ROOT / 'gnn/23_fast_mesh_entry.py',
             '--output', mesh_root, '--case-name', args.case_name, '--seed', args.seed,
             '--device', args.device, '--size-model', args.size_model.resolve(),
             '--gnn-model', args.gnn_model.resolve()] + geometry_args, ROOT / 'gnn')
        run([sys.executable, '-B', ROOT / 'gnn/09_export_gnn_mesh_for_fcnn.py',
             '--recovery', mesh_root / 'meshes', '--candidates', mesh_root / 'candidates',
             '--output', mesh_root / 'fcnn_meshes', '--name', args.case_name], ROOT / 'gnn')
        if not args.mesh_only:
            run([sys.executable, '-B', '-u', '-m', 'module44_load_surrogate_fcnn_sm.train_solve',
                 '--mesh', mesh_root / 'fcnn_meshes' / (args.case_name + '.npz'),
                 '--output', mesh_root / 'fcnn_results' / args.case_name,
                 '--E', '2.10e11', '--nu', '.30', '--thickness', '.01', '--behavior', 'plane-stress',
                 '--loading-control', 'top-line-load', '--top-line-load', '5e5', '--bottom-uy', '0',
                 '--load-family', 'uniform-amplitude', '--samples', args.samples, '--batch-size', '64',
                 '--epochs', args.epochs, '--learning-rate', '1e-3', '--loss', 'ritz-hybrid',
                 '--ritz-residual-weight', '1e-2', '--global-balance-weight', '1.0',
                 '--width-multiplier', '1', '--low-modes', '16', '--activation', 'silu',
                 '--patience', '800', '--dtype', 'float64', '--device', args.device, '--seed', '78001'])
    print(f'Complete. Results: {mesh_root}', flush=True)


if __name__ == '__main__':
    main()
