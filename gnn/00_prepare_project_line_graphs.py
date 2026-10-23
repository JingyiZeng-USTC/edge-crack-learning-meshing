from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data

from common.project_data import convert_project_sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the existing 200 project meshes into Line-Graph training data.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(r"D:\itl\eg\3\datasets\module30_crack_graph_v1"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/project_line_graph"))
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit; preserves manifest order.")
    parser.add_argument(
        "--single-straight-only",
        action="store_true",
        help="Reject the source unless every manifest row is one unbranched, one-segment straight crack.",
    )
    args = parser.parse_args()

    manifest_path = args.source / "crack_graph_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 project meshes, found {len(rows)}")
    if args.single_straight_only:
        invalid_rows = [
            row for row in rows
            if row.get("family") != "straight"
            or int(row.get("n_polylines", -1)) != 1
            or int(row.get("n_graph_segments", -1)) != 1
            or int(row.get("n_branches", -1)) != 0
        ]
        if invalid_rows:
            example = invalid_rows[0]
            raise RuntimeError(
                "--single-straight-only validation failed: "
                f"{len(invalid_rows)} of {len(rows)} rows are not one-segment, unbranched straight cracks; "
                f"first invalid case={example.get('case_id', '<unknown>')}"
            )
        print("validated source: 200 one-segment, unbranched straight cracks")
    if args.limit is not None:
        rows = rows[: args.limit]

    candidates_dir = args.output / "candidates"
    graphs_dir = args.output / "graphs"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    split_names = {"train": "train", "validation": "val", "test": "test"}
    split_manifest: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    report_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        source_file = args.source / row["file"]
        item = convert_project_sample(source_file, k=args.k)
        name = source_file.stem
        candidate_name, graph_name = f"{name}.npz", f"{name}.pt"
        split = split_names[row["split"]]
        np.savez_compressed(
            candidates_dir / candidate_name,
            nodes=item["nodes"], triangles=item["triangles"], gt_edges=item["gt_edges"],
            candidates=item["candidates"], labels=item["labels"], features=item["features"],
            width=item["width"], height=item["height"], crack_segments=item["crack_segments"],
            crack_family=item["crack_family"], edge_side=item["edge_side"], split=split,
        )
        graph = Data(
            x=torch.from_numpy(item["features"]).float(),
            edge_index=torch.from_numpy(item["line_edge_index"]).long(),
            y=torch.from_numpy(item["labels"]).long(),
        )
        graph.source_file = candidate_name
        torch.save(graph, graphs_dir / graph_name)
        split_manifest[split].append(graph_name)
        report_rows.append({
            "case_id": row["case_id"], "split": split, "family": row["family"], "edge": row["edge"],
            "nodes": len(item["nodes"]), "triangles": len(item["triangles"]),
            "gt_edges": len(item["gt_edges"]), "candidates": len(item["candidates"]),
            "line_graph_edges": int(item["line_edge_index"].shape[1]), "coverage": item["coverage"],
        })
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"converted {index:03d}/{len(rows)} coverage={item['coverage']:.6f}")

    (graphs_dir / "splits.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    with (args.output / "conversion_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    summary = {
        "source": str(args.source), "k": args.k, "meshes": len(rows),
        "single_straight_only": bool(args.single_straight_only),
        "split_counts": {key: len(value) for key, value in split_manifest.items()},
        "coverage_mean": float(np.mean([float(row["coverage"]) for row in report_rows])),
        "coverage_min": float(np.min([float(row["coverage"]) for row in report_rows])),
        "total_line_graph_nodes": int(sum(int(row["candidates"]) for row in report_rows)),
        "total_line_graph_edges": int(sum(int(row["line_graph_edges"]) for row in report_rows)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
