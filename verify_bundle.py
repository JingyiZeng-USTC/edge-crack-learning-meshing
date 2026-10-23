"""Read-only syntax/import/checkpoint/dataset validation."""
from pathlib import Path
import ast
import csv
import importlib
import importlib.util
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'gnn'))


def main():
    files = [p for p in ROOT.rglob('*.py') if 'results' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts]
    for p in files:
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    dependencies = ['numpy', 'scipy', 'matplotlib', 'meshio', 'sklearn', 'torch', 'torch_geometric']
    versions = {}
    for name in dependencies:
        module = importlib.import_module(name)
        versions[name] = getattr(module, '__version__', 'unknown')
    import torch
    size = torch.load(ROOT / 'module30_checkpoints/single_straight_crack_local_v4.pt', map_location='cpu', weights_only=False)
    graph = torch.load(ROOT / 'gnn/artifacts/single_straight_200_strict_model.pt', map_location='cpu', weights_only=False)
    assert size['model_family'] == 'single-crack-local-size-field-v4'
    assert len(size['feature_names']) == 14
    assert graph['in_channels'] == 10 and graph['layers'] == 4
    for name in ['sampling.sampling_pipeline', 'module44_load_surrogate_fcnn_sm.train_solve',
                 'module30_learned_size_field.train_single_crack_local', 'common.fast_inference']:
        module = importlib.import_module(name)
        assert Path(module.__file__).resolve().is_relative_to(ROOT), (name, module.__file__)
    for name in ['10_learned_size_to_gnn.py', '09_export_gnn_mesh_for_fcnn.py', '12_train_structured_graphsage.py']:
        runpy.run_path(str(ROOT / 'gnn' / name), run_name='import_check')
    dataset = ROOT / 'datasets/module30_single_straight_v1'
    rows = list(csv.DictReader((dataset / 'crack_graph_manifest.csv').open(encoding='utf-8', newline='')))
    counts = {split: sum(row['split'] == split for row in rows) for split in ['train', 'validation', 'test']}
    assert len(rows) == 200 and counts == {'train': 120, 'validation': 40, 'test': 40}, counts
    assert all((dataset / row['file']).is_file() for row in rows)
    print(json.dumps({'status': 'PASS', 'syntax_files': len(files), 'versions': versions,
                      'cuda_available': torch.cuda.is_available(), 'reference_meshes': len(rows),
                      'split': counts, 'imports_resolve_inside_bundle': True}, indent=2))


if __name__ == '__main__':
    main()
