"""Retrain GraphSAGE on the original 200 meshes with structured ranking."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from common.metrics import best_f1_threshold, classification_metrics
from common.model import EdgeGraphSAGE


class GraphFileDataset(Dataset):
    def __init__(self, folder: Path, names: list[str]):
        self.folder = folder
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        graph = torch.load(self.folder / self.names[index], weights_only=False)
        if not hasattr(graph, "rank_pair_index"):
            raise RuntimeError(f"{self.names[index]} has no structured ranking pairs")
        return graph


def count_labels(dataset: GraphFileDataset) -> torch.Tensor:
    counts = torch.zeros(2, dtype=torch.long)
    for index in range(len(dataset)):
        counts += torch.bincount(dataset[index].y, minlength=2)
    return counts


def edge_score(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 1] - logits[:, 0]


def ranking_loss(logits: torch.Tensor, pair_index: torch.Tensor, margin: float) -> torch.Tensor:
    if pair_index.numel() == 0:
        return logits.sum() * 0.0
    score = edge_score(logits)
    positive, negative = pair_index[0], pair_index[1]
    return F.softplus(margin - (score[positive] - score[negative])).mean()


@torch.no_grad()
def predictions(
    model: EdgeGraphSAGE,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    ranking_correct = 0
    ranking_total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
        probability = logits.softmax(dim=1)[:, 1]
        labels.append(batch.y.cpu().numpy())
        probabilities.append(probability.cpu().numpy())
        pairs = batch.rank_pair_index
        if pairs.numel():
            score = edge_score(logits)
            ranking_correct += int(torch.count_nonzero(score[pairs[0]] > score[pairs[1]]))
            ranking_total += int(pairs.shape[1])
    accuracy = ranking_correct / ranking_total if ranking_total else 0.0
    return np.concatenate(labels), np.concatenate(probabilities), float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE with CE plus crossing-edge ranking loss.",
    )
    parser.add_argument(
        "--graphs", type=Path,
        default=Path("data/single_straight_200_structured_line_graph/graphs"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/single_straight_200_strict_model.pt"),
    )
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ranking-weight", type=float, default=0.35)
    parser.add_argument("--ranking-margin", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=64031)
    parser.add_argument(
        "--allow-incomplete-split", action="store_true",
        help="Only for a small pipeline smoke test; production requires 120/40/40.",
    )
    args = parser.parse_args()
    if args.ranking_weight < 0.0 or args.ranking_margin < 0.0:
        raise ValueError("ranking settings must be non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.graphs / "splits.json").read_text(encoding="utf-8"))
    split_counts = {key: len(manifest[key]) for key in ("train", "val", "test")}
    if not args.allow_incomplete_split and split_counts != {
        "train": 120, "val": 40, "test": 40,
    }:
        raise RuntimeError("Structured dataset must preserve the original 120/40/40 split")
    if min(split_counts.values()) < 1:
        raise RuntimeError("Training, validation, and test splits must all be non-empty")
    train_dataset = GraphFileDataset(args.graphs, manifest["train"])
    val_dataset = GraphFileDataset(args.graphs, manifest["val"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    in_channels = int(train_dataset[0].x.shape[1])
    model = EdgeGraphSAGE(
        in_channels=in_channels, hidden_channels=args.hidden, layers=args.layers,
    ).to(device)
    initialized_from = None
    if args.init_model is not None:
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        expected = (in_channels, args.hidden, args.layers)
        actual = (
            int(checkpoint["in_channels"]), int(checkpoint["hidden"]),
            int(checkpoint["layers"]),
        )
        if actual != expected:
            raise RuntimeError(f"Initialization model schema {actual} != requested {expected}")
        model.load_state_dict(checkpoint["model_state"])
        initialized_from = str(args.init_model.resolve())

    counts = count_labels(train_dataset).float()
    class_weights = (counts.sum() / (2.0 * counts.clamp_min(1.0))).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_selection_score = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_ce = 0.0
        total_rank = 0.0
        total_edges = 0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)
            ce = F.cross_entropy(logits, batch.y, weight=class_weights)
            rank = ranking_loss(logits, batch.rank_pair_index, args.ranking_margin)
            loss = ce + args.ranking_weight * rank
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_ce += float(ce.detach()) * int(batch.num_nodes)
            total_rank += float(rank.detach())
            total_edges += int(batch.num_nodes)
            total_graphs += int(batch.num_graphs)

        val_labels, val_probs, val_ranking_accuracy = predictions(model, val_loader, device)
        threshold, _ = best_f1_threshold(val_labels, val_probs)
        val = classification_metrics(val_labels, val_probs, threshold)
        selection_score = math.sqrt(max(val["auprc"], 0.0) * max(val_ranking_accuracy, 0.0))
        row: dict[str, float | int] = {
            "epoch": epoch,
            "ce_loss": total_ce / max(total_edges, 1),
            "ranking_loss": total_rank / max(total_graphs, 1),
            "val_ranking_accuracy": val_ranking_accuracy,
            "selection_score": selection_score,
            **val,
        }
        history.append(row)
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "in_channels": in_channels,
                    "threshold": threshold,
                    "epoch": epoch,
                    "val": val,
                    "val_ranking_accuracy": val_ranking_accuracy,
                    "selection_score": selection_score,
                    "training_origin": "original-200-meshes-no-node-resampling",
                    "split": split_counts,
                    "loss": {
                        "cross_entropy": True,
                        "ranking_weight": args.ranking_weight,
                        "ranking_margin": args.ranking_margin,
                        "ranking_pairs": "teacher-positive-above-crossing-negative",
                    },
                    "initialized_from": initialized_from,
                },
                args.output,
            )
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} CE={row['ce_loss']:.4f} rank={row['ranking_loss']:.4f} "
                f"val_AUPRC={val['auprc']:.4f} rank_acc={val_ranking_accuracy:.4f} "
                f"score={selection_score:.4f} threshold={threshold:.3f}"
            )

    args.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8",
    )
    print(f"best structured checkpoint: {args.output} (score={best_selection_score:.6f})")


if __name__ == "__main__":
    main()
