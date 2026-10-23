from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np
import torch

from common.mesh_validation import recover_strict_gnn_project_mesh, topology_metrics, triangle_quality
from common.metrics import classification_metrics, set_recovery_metrics
from common.model import EdgeGraphSAGE


def _prefixed(prefix: str, values: dict[str, object]) -> dict[str, object]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _numeric_mean(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}


def draw_comparison(path: Path, reference_points: np.ndarray, reference: np.ndarray,
                    recovered_points: np.ndarray, recovered: np.ndarray,
                    crack_segments: np.ndarray, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 9), constrained_layout=True)
    for ax, plot_points, triangles, name in zip(
        axes, (reference_points, recovered_points), (reference, recovered),
        ("Project teacher", "Strict GraphSAGE recovery (no CDT)")
    ):
        if len(triangles):
            ax.triplot(Triangulation(plot_points[:, 0], plot_points[:, 1], triangles), lw=0.35, color="#2f6690")
        for segment in crack_segments:
            ax.plot(segment[:, 0], segment[:, 1], color="#d1495b", lw=1.8, zorder=4)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(title)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover and validate all held-out project meshes.")
    parser.add_argument("--data", type=Path, default=Path("data/project_line_graph"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/project_200_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/project_200_recovery"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-figures", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Optional number of test meshes for a smoke test.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recovered_dir, figures_dir = args.output / "meshes", args.output / "figures"
    recovered_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = EdgeGraphSAGE(in_channels=checkpoint["in_channels"], hidden_channels=checkpoint["hidden"],
                          layers=checkpoint["layers"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint["threshold"])
    split_manifest = json.loads((args.data / "graphs" / "splits.json").read_text(encoding="utf-8"))
    graph_names = split_manifest["test"][: args.limit] if args.limit is not None else split_manifest["test"]
    rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for index, graph_name in enumerate(graph_names, start=1):
        graph = torch.load(args.data / "graphs" / graph_name, weights_only=False).to(device)
        with torch.no_grad():
            probability = model(graph.x, graph.edge_index).softmax(dim=1)[:, 1].cpu().numpy()
        item = np.load(args.data / "candidates" / graph.source_file, allow_pickle=False)
        points, reference = item["nodes"], item["triangles"]
        candidates, gt_edges = item["candidates"], item["gt_edges"]
        crack_segments = item["crack_segments"]
        width, height = float(item["width"]), float(item["height"])
        positive_mask = probability >= threshold
        predicted = candidates[positive_mask]
        try:
            recovered_points, recovered, recovery_constraints, filter_stats = recover_strict_gnn_project_mesh(
                points, candidates, probability, width, height, crack_segments, threshold=threshold,
            )
        except RuntimeError as exc:
            failure = {
                "name": Path(graph_name).stem,
                "family": str(item["crack_family"]),
                "edge_side": str(item["edge_side"]),
                "error": str(exc),
            }
            failure_rows.append(failure)
            print(
                f"[{index:02d}/{len(graph_names)}] {failure['family']}/{failure['edge_side']} "
                f"STRICT_FAILURE: {exc}"
            )
            continue
        class_metrics = classification_metrics(item["labels"], probability, threshold)
        class_metrics.update(set_recovery_metrics(gt_edges, predicted, "raw_edge"))
        class_metrics.update(set_recovery_metrics(gt_edges, recovery_constraints, "constraint_edge"))
        comparison = set_recovery_metrics(reference, recovered, "original_triangle")
        teacher_quality = triangle_quality(points, reference)
        recovered_quality = triangle_quality(recovered_points, recovered)
        topology = topology_metrics(recovered_points, recovered, width, height, crack_segments)
        row: dict[str, object] = {
            "name": Path(graph_name).stem, "family": str(item["crack_family"]),
            "edge_side": str(item["edge_side"]), "nodes": len(points), "threshold": threshold,
            "recovered_nodes": len(recovered_points), "inserted_nodes": 0,
            **filter_stats,
            **class_metrics, **comparison, **_prefixed("teacher", teacher_quality),
            **_prefixed("recovered", recovered_quality), **_prefixed("topology", topology),
        }
        rows.append(row)
        np.savez_compressed(recovered_dir / f"{Path(graph_name).stem}.npz", nodes=recovered_points,
                            triangles=recovered, recovery_constraints=recovery_constraints, probability=probability,
                            crack_segments=crack_segments)
        print(f"[{index:02d}/{len(graph_names)}] {row['family']}/{row['edge_side']} "
              f"tri_F1={comparison['original_triangle_f1']:.4f} min_angle={recovered_quality['min_angle_deg']:.2f} "
              "strict_no_CDT=pass inserted=0")

    if not rows:
        raise RuntimeError(f"All {len(graph_names)} strict recoveries failed: {failure_rows}")
    fieldnames = list(rows[0])
    with (args.output / "per_mesh_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    headline_keys = [
        "accuracy", "precision", "recall", "f1", "auprc", "auroc", "raw_edge_f1", "constraint_edge_f1",
        "original_triangle_precision", "original_triangle_recall", "original_triangle_f1", "inserted_nodes",
        "predicted_edges", "kept_predicted_edges", "selected_edges",
        "selected_below_threshold_edges", "relative_area_error", "rejected_crossing_edges",
        "rejected_pass_through_edges", "rejected_invalid_edges",
        "teacher_min_angle_deg",
        "recovered_min_angle_deg", "teacher_fraction_below_20deg", "recovered_fraction_below_20deg",
        "teacher_quality_mean", "recovered_quality_mean", "topology_missing_constraint_edges",
        "topology_nonmanifold_edges", "topology_isolated_nodes",
    ]
    grouped: dict[str, dict[str, dict[str, float]]] = {"family": {}, "edge_side": {}}
    for group_key in grouped:
        buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            buckets[str(row[group_key])].append(row)
        grouped[group_key] = {name: _numeric_mean(bucket, headline_keys) for name, bucket in sorted(buckets.items())}
    summary = {
        "model": str(args.model), "device": str(device), "test_meshes": len(graph_names), "threshold": threshold,
        "successful_meshes": len(rows), "failed_meshes": len(failure_rows),
        "recovery_mode": "strict-gnn-no-cdt-no-ear-clipping",
        "overall_mean": _numeric_mean(rows, headline_keys), "by_group": grouped,
        "worst_triangle_f1": min(rows, key=lambda row: float(row["original_triangle_f1"]))["name"],
        "worst_recovered_min_angle": min(rows, key=lambda row: float(row["recovered_min_angle_deg"]))["name"],
        "meshes_with_topology_failures": len(failure_rows) + sum(
            int(row["topology_missing_constraint_edges"]) > 0
            or int(row["topology_nonmanifold_edges"]) > 0
            or int(row["topology_isolated_nodes"]) > 0 for row in rows
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "failures.json").write_text(
        json.dumps(failure_rows, indent=2), encoding="utf-8",
    )

    figure_rows: list[dict[str, object]] = []
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        figure_rows.append(min(family_rows, key=lambda row: float(row["original_triangle_f1"])))
        figure_rows.append(max(family_rows, key=lambda row: float(row["original_triangle_f1"])))
    unique_rows = {str(row["name"]): row for row in figure_rows}
    for row in list(unique_rows.values())[: args.max_figures]:
        item = np.load(args.data / "candidates" / f"{row['name']}.npz", allow_pickle=False)
        recovered_item = np.load(recovered_dir / f"{row['name']}.npz", allow_pickle=False)
        draw_comparison(figures_dir / f"{row['name']}.png", item["nodes"], item["triangles"],
                        recovered_item["nodes"], recovered_item["triangles"], item["crack_segments"],
                        f"{row['family']} / {row['edge_side']} / triangle F1={float(row['original_triangle_f1']):.3f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
