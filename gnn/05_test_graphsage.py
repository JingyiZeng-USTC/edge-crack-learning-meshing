from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from common.metrics import classification_metrics, set_recovery_metrics
from common.model import EdgeGraphSAGE


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out edge cracks.")
    parser.add_argument("--graphs", type=Path, default=Path("data/cdt_graphs"))
    parser.add_argument("--candidates", type=Path, default=Path("data/cdt_candidates"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/cdt_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/cdt_test_metrics.json"))
    parser.add_argument("--threshold", type=float, default=None, help="Default: validation-selected checkpoint threshold")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    threshold = checkpoint["threshold"] if args.threshold is None else args.threshold
    model = EdgeGraphSAGE(
        in_channels=checkpoint["in_channels"], hidden_channels=checkpoint["hidden"], layers=checkpoint["layers"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    manifest = json.loads((args.graphs / "splits.json").read_text(encoding="utf-8"))
    labels, probabilities = [], []
    ranking_correct, ranking_total = 0, 0
    recovery = []
    for graph_name in manifest["test"]:
        graph = torch.load(args.graphs / graph_name, weights_only=False).to(device)
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            prob = logits.softmax(dim=1)[:, 1].cpu().numpy()
            if hasattr(graph, "rank_pair_index") and graph.rank_pair_index.numel():
                score = logits[:, 1] - logits[:, 0]
                pairs = graph.rank_pair_index
                ranking_correct += int(torch.count_nonzero(score[pairs[0]] > score[pairs[1]]))
                ranking_total += int(pairs.shape[1])
        labels.append(graph.y.cpu().numpy())
        probabilities.append(prob)
        item = np.load(args.candidates / Path(graph.source_file).name)
        predicted = item["candidates"][prob >= threshold]
        recovery.append(set_recovery_metrics(item["gt_edges"], predicted, "edge"))
    result = classification_metrics(np.concatenate(labels), np.concatenate(probabilities), threshold)
    result["threshold"] = float(threshold)
    if ranking_total:
        result["crossing_rank_accuracy"] = float(ranking_correct / ranking_total)
        result["crossing_rank_pairs"] = int(ranking_total)
    for key in recovery[0]:
        result[key] = float(np.mean([row[key] for row in recovery]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
