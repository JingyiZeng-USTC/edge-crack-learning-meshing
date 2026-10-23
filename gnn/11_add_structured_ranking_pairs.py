"""Add structured edge-ranking supervision to the existing 200 Line Graphs.

This script never regenerates nodes or triangles.  It reads the original
single-straight-crack candidates/labels and records pairs

    (teacher-positive edge, crossing teacher-negative edge)

so training can rank a legal teacher edge above an alternative that would block
it during confidence-ordered planar recovery.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

def _spatial_cells(a: np.ndarray, b: np.ndarray, cell_size: float) -> list[tuple[int, int]]:
    lower = np.floor(np.minimum(a, b) / cell_size).astype(np.int64)
    upper = np.floor(np.maximum(a, b) / cell_size).astype(np.int64)
    return [
        (x, y)
        for x in range(int(lower[0]), int(upper[0]) + 1)
        for y in range(int(lower[1]), int(upper[1]) + 1)
    ]


def crossing_rank_pairs(
    points: np.ndarray,
    candidates: np.ndarray,
    labels: np.ndarray,
    width: float,
    height: float,
    max_positive_per_negative: int = 2,
) -> np.ndarray:
    """Return [2, P] positive/negative candidate-index ranking pairs."""
    points = np.asarray(points, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    positive_ids = np.flatnonzero(labels == 1)
    negative_ids = np.flatnonzero(labels == 0)
    if len(positive_ids) == 0 or len(negative_ids) == 0:
        return np.empty((2, 0), dtype=np.int64)

    positive_edges = candidates[positive_ids]
    lengths = np.linalg.norm(
        points[positive_edges[:, 1]] - points[positive_edges[:, 0]], axis=1,
    )
    positive_lengths = lengths[lengths > 0.0]
    median_length = (
        float(np.median(positive_lengths))
        if len(positive_lengths)
        else min(width, height) / 100.0
    )
    cell_size = max(median_length, min(width, height) / 250.0)
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for local_id, edge in enumerate(positive_edges):
        a, b = points[edge]
        for cell in _spatial_cells(a, b, cell_size):
            grid[cell].append(local_id)

    pairs: list[tuple[int, int]] = []
    area_tolerance = 1.0e-12 * max(width, height, 1.0) ** 2

    def cross(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]

    for negative_id in negative_ids:
        u, v = map(int, candidates[negative_id])
        a, b = points[[u, v]]
        nearby: set[int] = set()
        for cell in _spatial_cells(a, b, cell_size):
            nearby.update(grid.get(cell, ()))
        if not nearby:
            continue
        local_ids = np.fromiter(nearby, dtype=np.int64)
        local_edges = positive_edges[local_ids]
        disjoint = (
            (local_edges[:, 0] != u) & (local_edges[:, 1] != u)
            & (local_edges[:, 0] != v) & (local_edges[:, 1] != v)
        )
        local_ids = local_ids[disjoint]
        if len(local_ids) == 0:
            continue
        local_edges = positive_edges[local_ids]
        c = points[local_edges[:, 0]]
        d = points[local_edges[:, 1]]
        ab = np.broadcast_to(b - a, c.shape)
        cd = d - c
        o1 = cross(ab, c - a)
        o2 = cross(ab, d - a)
        o3 = cross(cd, np.broadcast_to(a, c.shape) - c)
        o4 = cross(cd, np.broadcast_to(b, c.shape) - c)
        first_opposite = ((o1 > area_tolerance) & (o2 < -area_tolerance)) | (
            (o1 < -area_tolerance) & (o2 > area_tolerance)
        )
        second_opposite = ((o3 > area_tolerance) & (o4 < -area_tolerance)) | (
            (o3 < -area_tolerance) & (o4 > area_tolerance)
        )
        crossing_local = local_ids[first_opposite & second_opposite]
        if len(crossing_local) == 0:
            continue
        order = np.lexsort((positive_ids[crossing_local], lengths[crossing_local]))
        for local_id in crossing_local[order[:max_positive_per_negative]]:
            pairs.append((int(positive_ids[local_id]), int(negative_id)))
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64).T


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add crossing-edge ranking pairs to the original 200 Line Graphs.",
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/single_straight_200_line_graph"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/single_straight_200_structured_line_graph"),
    )
    parser.add_argument("--max-positive-per-negative", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.max_positive_per_negative < 1:
        raise ValueError("max-positive-per-negative must be at least 1")

    source_graphs = args.source / "graphs"
    source_candidates = args.source / "candidates"
    manifest = json.loads((source_graphs / "splits.json").read_text(encoding="utf-8"))
    all_ordered = [(split, name) for split in ("train", "val", "test") for name in manifest[split]]
    if len(all_ordered) != 200:
        raise RuntimeError(f"Expected the original 200 graphs, found {len(all_ordered)}")
    if args.limit is not None:
        ordered = [
            (split, name)
            for split in ("train", "val", "test")
            for name in manifest[split][: args.limit]
        ]
    else:
        ordered = all_ordered

    output_graphs = args.output / "graphs"
    output_graphs.mkdir(parents=True, exist_ok=True)
    output_manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    report: list[dict[str, int | str]] = []
    total_pairs = 0
    for index, (split, graph_name) in enumerate(ordered, start=1):
        source_graph = torch.load(source_graphs / graph_name, weights_only=False)
        candidate_path = source_candidates / Path(source_graph.source_file).name
        item = np.load(candidate_path, allow_pickle=False)
        if not np.array_equal(source_graph.y.numpy(), item["labels"]):
            raise RuntimeError(f"Graph/candidate labels differ for {graph_name}")
        pair_index = crossing_rank_pairs(
            item["nodes"], item["candidates"], item["labels"],
            float(item["width"]), float(item["height"]),
            max_positive_per_negative=args.max_positive_per_negative,
        )
        source_graph.rank_pair_index = torch.from_numpy(pair_index).long()
        source_graph.training_sample_origin = "original-200-meshes-no-resampling"
        torch.save(source_graph, output_graphs / graph_name)
        output_manifest[split].append(graph_name)
        total_pairs += int(pair_index.shape[1])
        report.append({
            "name": Path(graph_name).stem,
            "split": split,
            "line_graph_nodes": int(source_graph.num_nodes),
            "ranking_pairs": int(pair_index.shape[1]),
        })
        if index == 1 or index % 10 == 0 or index == len(ordered):
            print(
                f"structured {index:03d}/{len(ordered)} "
                f"pairs={pair_index.shape[1]} total={total_pairs}"
            )

    (output_graphs / "splits.json").write_text(
        json.dumps(output_manifest, indent=2), encoding="utf-8",
    )
    summary = {
        "format": "original-200-line-graphs-with-crossing-ranking-v1",
        "source": str(args.source.resolve()),
        "nodes_regenerated": False,
        "triangles_regenerated": False,
        "meshes": len(ordered),
        "split_counts": {key: len(value) for key, value in output_manifest.items()},
        "ranking_pairs": total_pairs,
        "mean_pairs_per_mesh": total_pairs / max(len(ordered), 1),
        "max_positive_per_negative": args.max_positive_per_negative,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
