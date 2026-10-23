"""Optional offline retraining; bundled final checkpoints are never overwritten."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(args, cwd):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['size', 'gnn', 'both'], default='both')
    parser.add_argument('--size-epochs', type=int, default=300)
    parser.add_argument('--gnn-epochs', type=int, default=80)
    parser.add_argument('--candidate-k', type=int, default=16, help='Historical training graphs used k=16; final inference uses k=20')
    args = parser.parse_args()
    raw = ROOT / 'datasets/module30_single_straight_v1'
    if args.stage in ['size', 'both']:
        dataset = ROOT / 'datasets/module30_single_straight_local_v4'
        if not (dataset / 'metadata.json').is_file():
            run([sys.executable, '-B', '-m', 'module30_learned_size_field.prepare_single_crack_local',
                 '--source', raw, '--output', dataset], ROOT)
        run([sys.executable, '-B', '-u', '-m', 'module30_learned_size_field.train_single_crack_local',
             '--dataset', dataset, '--output', ROOT / 'module30_checkpoints/retrained_size_field.pt',
             '--epochs', args.size_epochs], ROOT)
    if args.stage in ['gnn', 'both']:
        plain = ROOT / 'gnn/data' / f'single_straight_200_line_graph_k{args.candidate_k}'
        structured = ROOT / 'gnn/data' / f'single_straight_200_structured_line_graph_k{args.candidate_k}'
        if not (plain / 'summary.json').is_file():
            run([sys.executable, '-B', '-u', ROOT / 'gnn/00_prepare_project_line_graphs.py',
                 '--source', raw, '--output', plain, '--k', args.candidate_k, '--single-straight-only'], ROOT / 'gnn')
        if not (structured / 'summary.json').is_file():
            run([sys.executable, '-B', '-u', ROOT / 'gnn/11_add_structured_ranking_pairs.py',
                 '--source', plain, '--output', structured], ROOT / 'gnn')
        model = ROOT / 'gnn/artifacts/retrained_strict_model.pt'
        run([sys.executable, '-B', '-u', ROOT / 'gnn/12_train_structured_graphsage.py',
             '--graphs', structured / 'graphs', '--output', model,
             '--init-model', ROOT / 'gnn/artifacts/single_straight_200_model.pt',
             '--epochs', args.gnn_epochs], ROOT / 'gnn')
        run([sys.executable, '-B', ROOT / 'gnn/05_test_graphsage.py', '--graphs', structured / 'graphs',
             '--candidates', plain / 'candidates', '--model', model,
             '--output', ROOT / 'gnn/artifacts/retrained_test_metrics.json'], ROOT / 'gnn')


if __name__ == '__main__':
    main()
