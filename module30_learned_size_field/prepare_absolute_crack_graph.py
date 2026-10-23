"""Convert existing pure crack meshes into absolute-size-field v2 samples.

The source meshes may have been produced by any offline reference method.  The
new samples retain only geometry, triangles, geometry-only features and the
mesh-derived absolute target size.  No analytic size value is exposed to the
v2 model, and no FEM quantity is read or written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .crack_graph_features import (
    ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES,
    CRACK_GRAPH_FEATURE_NAMES,
)


MODEL_FAMILY = "absolute-crack-graph-size-field-v2"
DROP_FEATURE = "analytic_h_over_hmax"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pure crack meshes to 21-feature absolute-size samples",
    )
    parser.add_argument("--source", default="datasets/module30_crack_graph_v1")
    parser.add_argument("--output", default="datasets/module30_crack_graph_absolute_v2")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.source)
    output = Path(args.output)
    source_metadata = json.loads(
        (source / "metadata.json").read_text(encoding="utf-8"),
    )
    source_features = list(source_metadata.get("feature_names", ()))
    if source_features != CRACK_GRAPH_FEATURE_NAMES:
        raise ValueError("Source dataset is not the expected 22-feature crack-graph dataset")
    if bool(source_metadata.get("uses_fem_fields", True)):
        raise ValueError("Source dataset is not geometry-only")
    if (output / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite")

    manifest_name = str(source_metadata.get("manifest", "crack_graph_manifest.csv"))
    with (source / manifest_name).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Source manifest is empty")

    h_min = float(source_metadata["h_min"])
    h_max = float(source_metadata["h_max"])
    log_min = float(np.log(h_min))
    log_range = float(np.log(h_max) - np.log(h_min))
    keep_indices = [
        index for index, name in enumerate(source_features)
        if name != DROP_FEATURE
    ]
    if len(keep_indices) != len(ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES):
        raise RuntimeError("Absolute feature projection has the wrong width")

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    converted_rows: list[dict] = []
    total_points = 0
    forbidden = set(source_metadata.get("forbidden_labels", ()))
    for row_index, row in enumerate(rows):
        source_path = source / row["file"]
        destination = mesh_dir / Path(row["file"]).name
        with np.load(source_path, allow_pickle=False) as data:
            available = set(data.files)
            if forbidden.intersection(available):
                raise ValueError(f"{source_path} contains a forbidden FEM/GNN label")
            required = {
                "points", "triangles", "features", "target_h", "sample_weight",
                "W", "H", "crack_polyline_points", "crack_polyline_offsets",
                "crack_graph_vertices", "crack_graph_segments", "crack_tips",
                "crack_entries", "branch_points", "crack_family", "edge_side",
            }
            missing = required - available
            if missing:
                raise ValueError(f"{source_path} misses fields: {sorted(missing)}")
            features = np.asarray(data["features"], dtype=np.float32)[:, keep_indices]
            target_h = np.clip(
                np.asarray(data["target_h"], dtype=np.float32), h_min, h_max,
            )
            target_normalized_log_h = np.clip(
                (np.log(target_h) - log_min) / log_range, 0.0, 1.0,
            ).astype(np.float32)
            payload = {
                "points": np.asarray(data["points"], dtype=np.float32),
                "triangles": np.asarray(data["triangles"], dtype=np.int64),
                "features": features,
                "target_h": target_h,
                "target_normalized_log_h": target_normalized_log_h,
                "sample_weight": np.asarray(data["sample_weight"], dtype=np.float32),
                "W": np.asarray(data["W"]),
                "H": np.asarray(data["H"]),
                "crack_polyline_points": np.asarray(data["crack_polyline_points"], dtype=np.float32),
                "crack_polyline_offsets": np.asarray(data["crack_polyline_offsets"], dtype=np.int64),
                "crack_graph_vertices": np.asarray(data["crack_graph_vertices"], dtype=np.float32),
                "crack_graph_segments": np.asarray(data["crack_graph_segments"], dtype=np.int64),
                "crack_tips": np.asarray(data["crack_tips"], dtype=np.float32),
                "crack_entries": np.asarray(data["crack_entries"], dtype=np.float32),
                "branch_points": np.asarray(data["branch_points"], dtype=np.float32),
                "crack_family": np.asarray(data["crack_family"]),
                "edge_side": np.asarray(data["edge_side"]),
            }
        if features.shape[1] != 21 or len(features) != len(target_h):
            raise ValueError(f"Invalid converted tensor shapes in {source_path}")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target_h)):
            raise ValueError(f"Non-finite converted values in {source_path}")
        np.savez_compressed(destination, **payload)
        converted = dict(row)
        converted["file"] = f"meshes/{destination.name}"
        converted_rows.append(converted)
        total_points += len(target_h)
        if (row_index + 1) % 20 == 0 or row_index + 1 == len(rows):
            print(f"Converted {row_index + 1}/{len(rows)} pure meshes", flush=True)

    output_manifest = "absolute_crack_graph_manifest.csv"
    with (output / output_manifest).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(converted_rows[0]))
        writer.writeheader()
        writer.writerows(converted_rows)

    metadata = {
        "format_version": 2,
        "model_family": MODEL_FAMILY,
        "manifest": output_manifest,
        "target": "absolute_normalized_log_mesh_size",
        "target_source": "points_and_triangles_only",
        "offline_teacher": source_metadata.get("reference_size_field", "existing_mesh_method"),
        "runtime_uses_analytic_size_field": False,
        "uses_fem_fields": False,
        "uses_gnn": False,
        "feature_names": ABSOLUTE_CRACK_GRAPH_FEATURE_NAMES,
        "dropped_feature": DROP_FEATURE,
        "meshes": len(converted_rows),
        "points": int(total_points),
        "split_counts": source_metadata["split_counts"],
        "family_counts": source_metadata["family_counts"],
        "edge_counts": source_metadata["edge_counts"],
        "h_min": h_min,
        "h_max": h_max,
        "parameterization": "sigmoid_normalized_log_h",
        "source_dataset": str(source.resolve()),
        "forbidden_labels": sorted(forbidden),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"Saved: {output}")
    print(f"Meshes: {len(converted_rows)} | points: {total_points}")
    print("Features: 21 geometry-only | analytic runtime dependency: none")
    print("FEM/GNN labels: none")


if __name__ == "__main__":
    main()
