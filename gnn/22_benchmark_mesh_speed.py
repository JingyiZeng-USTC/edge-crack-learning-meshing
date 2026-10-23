"""Paired timing and mesh-equivalence tests; never trains a network."""
from pathlib import Path
import argparse
import csv
import importlib.util
import json
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', action='store_true', help='Use all original 12+4 cases; default: one straight and one curved')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--resume', action='store_true', help='Reuse completed timing files only when all forwarded arguments and mode match')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'mesh_speed_benchmark')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--figure', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    spec = importlib.util.spec_from_file_location('suite_cases', ROOT / '17_test_quality20_straight_curved.py')
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    cases = suite.load_cases(ROOT / 'unseen_single_crack_cases_k20.json', ROOT / 'quality20_curved_cases.json')
    if not args.suite:
        cases = [cases[1], cases[12]]
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    timings = []
    for case in cases:
        name = case['case_name']
        geometry = []
        if case['family'] == 'straight':
            for key in ['edge', 'origin', 'length', 'angle_deg']:
                geometry += ['--' + key.replace('_', '-'), str(case[key])]
        else:
            geometry_path = args.output / (name + '_geometry.json')
            geometry_path.write_text(json.dumps({'W': 1., 'H': 2., 'crack_polylines': [suite.curved_polyline(case).tolist()]}), encoding='utf-8')
            geometry = ['--geometry-json', str(geometry_path)]
        for repeat in range(args.repeats):
            reports = {}
            # Alternate order to reduce systematic cache/order bias.
            for mode in (['baseline', 'fast'] if repeat % 2 == 0 else ['fast', 'baseline']):
                folder = args.output / name / str(repeat) / mode
                folder.mkdir(parents=True, exist_ok=True)
                timing_path = folder / 'timing.json'
                command = [sys.executable, '-B', '-u', str(ROOT / '21_profile_mesh_generation.py'),
                           '--timing-output', str(timing_path), '--output', str(folder), '--case-name', name,
                           '--seed', str(case['seed']), '--candidate-k', '20', '--device', args.device,
                           '--figure' if args.figure else '--no-figure'] + geometry
                if mode == 'fast':
                    command += ['--fast']
                expected = [value for value in command[command.index('--output'):] if value != '--fast']
                timing = json.loads(timing_path.read_text(encoding='utf-8')) if options_exist(args.resume, timing_path, folder / (name + '.json')) else None
                if timing is not None and timing['pipeline_arguments'] == expected and timing['fast_kernels'] == (mode == 'fast') and 'process_wall_seconds' in timing:
                    wall = timing['process_wall_seconds']
                    print(f'REUSE {name} {mode}', flush=True)
                else:
                    print(f'START {name} repeat={repeat+1} {mode}', flush=True)
                    began = time.perf_counter()
                    with (folder / 'run.log').open('w', encoding='utf-8') as log:
                        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
                    wall = time.perf_counter() - began
                    timing = json.loads(timing_path.read_text(encoding='utf-8'))
                    timing['process_wall_seconds'] = wall
                    timing_path.write_text(json.dumps(timing, indent=2), encoding='utf-8')
                reports[mode] = (folder, timing)
                for stage, item in timing['stages'].items():
                    timings.append({'case': name, 'repeat': repeat+1, 'mode': mode, 'stage': stage, **item})
                print(f'DONE {mode} {wall:.3f}s', flush=True)
            same = {}
            probability_delta = 0.0
            probability_close = False
            for archive, keys in [('candidates', ['nodes', 'candidates', 'features', 'line_edge_index', 'probability']),
                                  ('meshes', ['nodes', 'triangles', 'gnn_selected_edges'])]:
                with np.load(reports['baseline'][0] / archive / (name + '.npz')) as old, np.load(reports['fast'][0] / archive / (name + '.npz')) as new:
                    for key in keys:
                        same[archive + '/' + key] = bool(np.array_equal(old[key], new[key]))
                        if key == 'probability':
                            probability_delta = float(np.max(np.abs(old[key] - new[key])))
                            probability_close = bool(np.allclose(old[key], new[key], rtol=0.0, atol=5e-6))
            baseline = reports['baseline'][1]['process_wall_seconds']
            fast = reports['fast'][1]['process_wall_seconds']
            final = json.loads((reports['fast'][0] / (name + '.json')).read_text(encoding='utf-8'))
            topology = final['recovery']['topology']
            quality = final['recovery']['quality']
            accepted = (quality['min_angle_deg'] >= 20.0 and topology['components'] == 1
                        and all(topology[key] == 0 for key in ['isolated_nodes', 'nonmanifold_edges', 'missing_constraint_edges', 'duplicate_triangles'])
                        and quality['degenerate_triangles'] == 0 and final['recovery']['inserted_nodes'] == 0
                        and not final['teacher_used_at_inference'] and not final['cdt_fallback_used'])
            result = {'case': name, 'repeat': repeat+1, 'baseline_seconds': baseline, 'fast_seconds': fast,
                      'speedup': baseline / fast, 'reduction_percent': 100 * (1 - fast / baseline),
                      'exact_equal': same, 'all_exact_equal': all(same.values()),
                      'all_geometry_exact_equal': all(value for key, value in same.items() if key != 'candidates/probability'),
                      'probability_max_abs_delta': probability_delta, 'probability_close_atol_5e_6': probability_close,
                      'quality_and_topology_pass': accepted,
                      'quality': final['recovery']['quality'], 'topology': final['recovery']['topology']}
            result['stage_comparison_seconds'] = {
                stage: {'baseline': reports['baseline'][1]['stages'].get(stage, {}).get('inclusive_seconds'),
                        'fast': reports['fast'][1]['stages'].get(stage, {}).get('inclusive_seconds')}
                for stage in sorted(set(reports['baseline'][1]['stages']) | set(reports['fast'][1]['stages']))}
            for mode in ['baseline', 'fast']:
                stage_values = reports[mode][1]['stages']
                core = sum(value['inclusive_seconds'] for stage, value in stage_values.items() if '/' not in stage)
                core -= sum(stage_values.get(key, {}).get('inclusive_seconds', 0.) for key in [
                    'gnn_model_load', 'size_field_setup/size_model_load', 'compressed_data_export', 'plot_export'])
                result[mode + '_core_stage_seconds'] = core
            results.append(result)
            (args.output / 'summary.json').write_text(json.dumps({'cases': results}, indent=2), encoding='utf-8')
            with (args.output / 'stage_timings.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['case', 'repeat', 'mode', 'stage', 'calls', 'inclusive_seconds', 'exclusive_seconds'])
                writer.writeheader()
                writer.writerows(timings)
            with (args.output / 'stage_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['case', 'repeat', 'stage', 'baseline_seconds', 'fast_seconds'])
                for item in results:
                    writer.writerow([item['case'], item['repeat'], 'FULL_PROCESS', item['baseline_seconds'], item['fast_seconds']])
                    writer.writerow([item['case'], item['repeat'], 'CORE_STAGES_EXCLUDING_LOADS_AND_OUTPUT', item['baseline_core_stage_seconds'], item['fast_core_stage_seconds']])
                    for stage, values in item['stage_comparison_seconds'].items():
                        writer.writerow([item['case'], item['repeat'], stage, values['baseline'], values['fast']])
            print(f'COMPARE geometry_exact={result["all_geometry_exact_equal"]} probability_delta={probability_delta:.3g} speedup={result["speedup"]:.3f}', flush=True)
            if not (result['all_geometry_exact_equal'] and probability_close and accepted):
                raise RuntimeError('Mesh or prediction changed; inspect exact_equal before enabling fast kernels')
    print(f'Saved: {args.output}', flush=True)


def options_exist(resume, *paths):
    return resume and all(path.is_file() for path in paths)


if __name__ == '__main__':
    main()
