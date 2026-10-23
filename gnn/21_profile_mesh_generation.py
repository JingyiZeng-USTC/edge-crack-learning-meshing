"""Stage timing for the existing mesh generator, without any model training.

Example: python 21_profile_mesh_generation.py --timing-output artifacts/timing/baseline.json
         --edge left --origin 1.07 --length .32 --angle-deg 11 --seed 2102
Use --fast for equivalent scalar kernels. All pipeline arguments are forwarded.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--timing-output', type=Path, required=True)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--call-profile', action='store_true')
    options, forwarded = parser.parse_known_args()
    began = time.perf_counter()
    pipeline = runpy.run_path(str(ROOT / '10_learned_size_to_gnn.py'), run_name='timed_pipeline')
    imported = time.perf_counter() - began
    import torch
    import numpy as np
    from common import mesh_validation, meshing
    from module30_learned_size_field import single_crack_local_field
    if options.fast:
        from common.fast_inference import install_fast_kernels
        install_fast_kernels()

    records = {}
    stack = []

    def wrap(container, name, label, synchronize=False):
        original = container[name] if isinstance(container, dict) else getattr(container, name)

        @functools.wraps(original)
        def measured(*args, **kwargs):
            if synchronize and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            path = '/'.join([frame[0] for frame in stack] + [label])
            start = time.perf_counter()
            frame = [label, 0.0]
            stack.append(frame)
            try:
                return original(*args, **kwargs)
            finally:
                if synchronize and torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                stack.pop()
                if stack:
                    stack[-1][1] += elapsed
                record = records.setdefault(path, {'calls': 0, 'inclusive_seconds': 0.0, 'exclusive_seconds': 0.0})
                record['calls'] += 1
                record['inclusive_seconds'] += elapsed
                record['exclusive_seconds'] += elapsed - frame[1]
                if not stack:
                    print(f'[TIMING] {label}: {elapsed:.6f} s', flush=True)

        if isinstance(container, dict):
            container[name] = measured
        else:
            setattr(container, name, measured)

    # Function globals, not the dictionary copy returned by run_path.
    namespace = pipeline['main'].__globals__
    for name, label, sync in [
        ('_geometry', 'geometry', False),
        ('_device', 'device_initialization', True),
        ('project_candidate_edges', 'candidate_edges', False),
        ('edge_features', 'edge_features', False),
        ('line_graph_edge_index', 'line_graph', False),
        ('_load_connectivity_model', 'gnn_model_load', True),
        ('recover_strict_gnn_project_mesh', 'structured_decode_and_checks', False),
        ('constrained_quality_smoothing', 'node_quality_smoothing', False),
        ('triangle_quality', 'final_angle_quality', False),
        ('topology_metrics', 'final_topology_checks', False),
        ('_draw', 'plot_export', False),
    ]:
        wrap(namespace, name, label, sync)
    wrap(namespace['SingleCrackLocalSizeField'], '__init__', 'size_field_setup', True)
    wrap(single_crack_local_field, '_load_runtime', 'size_model_load', True)
    wrap(single_crack_local_field, 'build_single_crack_local_features', 'size_features')
    wrap(single_crack_local_field.DirectNormalizedLogSizeMLP, 'forward', 'size_network_forward', True)
    wrap(namespace['SingleCrackLocalSizeField'], '_build_grid', 'size_prediction_and_grid', True)
    wrap(namespace['SingleCrackLocalSizeField'], '_rescale_to_complexity', 'complexity_rescale')
    # Avoid timer overhead on the hundreds of thousands of scalar interpolation calls.
    # Scalar size queries inside sampling are deliberately not individually timed.
    wrap(namespace['AdaptivePoissonSampler'], 'sample', 'adaptive_sampling')
    wrap(namespace['EdgeGraphSAGE'], 'forward', 'gnn_forward', True)
    wrap(mesh_validation, 'select_non_crossing_project_edges', 'edge_legality_filter')
    wrap(mesh_validation, '_structured_quality_redecode', 'candidate_diagonal_quality')
    wrap(mesh_validation, 'strict_triangular_faces', 'triangle_recovery')
    wrap(mesh_validation, 'topology_metrics', 'decoder_topology_checks')
    wrap(np, 'savez_compressed', 'compressed_data_export')
    sys.argv = [str(ROOT / '10_learned_size_to_gnn.py')] + forwarded
    profiler = None
    if options.call_profile:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
    failure = None
    try:
        pipeline['main']()
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        if profiler:
            profiler.disable()
        total = time.perf_counter() - began
        root_sum = sum(v['inclusive_seconds'] for k, v in records.items() if '/' not in k)
        report = {
            'format': 'mesh-stage-timing-v1', 'fast_kernels': options.fast,
            'pipeline_arguments': forwarded, 'failure': failure,
            'offline_training_included': False,
            'imports_seconds': imported, 'script_wall_seconds': total,
            'other_setup_and_reporting_seconds': max(0.0, total - imported - root_sum),
            'cuda_synchronized': True, 'call_profile_enabled': bool(profiler),
            'torch_version': torch.__version__,
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_initialized() else None,
            'torch_cpu_threads': torch.get_num_threads(),
            'stages': records,
            'note': 'Inclusive nested times overlap; sum root stages only. Imports, model loads, plots and exports are reported separately. cProfile changes runtime; do not use profiled runs for speed comparisons.',
        }
        options.timing_output.parent.mkdir(parents=True, exist_ok=True)
        options.timing_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        with options.timing_output.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['stage', 'calls', 'inclusive_seconds', 'exclusive_seconds'])
            for key, value in records.items():
                writer.writerow([key, value['calls'], value['inclusive_seconds'], value['exclusive_seconds']])
        if profiler:
            profiler.dump_stats(str(options.timing_output.with_suffix('.prof')))
        print(f'[TIMING] Wall={total:.6f} s; saved {options.timing_output}', flush=True)


if __name__ == '__main__':
    main()
