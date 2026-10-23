"""Re-extract local-invariant inputs from pure straight-crack meshes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from crack_graph_geometry import CrackGraphGeometry

from .single_crack_local_features import (
    SINGLE_CRACK_LOCAL_FEATURE_NAMES,
    SINGLE_CRACK_LOCAL_MODEL_FAMILY,
    build_single_crack_local_features,
)


def _polylines(data) -> list[np.ndarray]:
    points = np.asarray(data["crack_polyline_points"], dtype=float)
    offsets = np.asarray(data["crack_polyline_offsets"], dtype=np.int64)
    return [points[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build straight-only local-invariant size-field supervision",
    )
    parser.add_argument("--source", default="datasets/module30_single_straight_v1")
    parser.add_argument("--output", default="datasets/module30_single_straight_local_v4")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = Path(args.source)
    output = Path(args.output)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("family_counts") != {"straight": metadata.get("meshes")}:
        raise ValueError("Source dataset must contain straight cracks only")
    if bool(metadata.get("uses_fem_fields", True)) or bool(metadata.get("uses_gnn", True)):
        raise ValueError("Source dataset must be a pure geometry/mesh dataset")
    if (output / "metadata.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; use --overwrite")
    with (source / metadata["manifest"]).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Source manifest is empty")

    mesh_dir = output / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    h_min = float(metadata["h_min"])
    h_max = float(metadata["h_max"])
    log_min = float(np.log(h_min))
    log_range = float(np.log(h_max) - np.log(h_min))
    converted = []
    feature_min = np.full(len(SINGLE_CRACK_LOCAL_FEATURE_NAMES), np.inf)
    feature_max = np.full(len(SINGLE_CRACK_LOCAL_FEATURE_NAMES), -np.inf)
    total_points = 0

    forbidden = set(metadata.get("forbidden_labels", ()))
    for index, row in enumerate(rows):
        if row.get("family") != "straight":
            raise ValueError(f"Non-straight case found in source: {row}")
        source_path = source / row["file"]
        with np.load(source_path, allow_pickle=False) as data:
            if forbidden.intersection(data.files):
                raise ValueError(f"{source_path} contains a forbidden physics label")
            geometry = CrackGraphGeometry(
                float(data["W"]),
                float(data["H"]),
                _polylines(data),
            )
            if len(geometry.crack_polylines[0]) != 2:
                raise ValueError(f"Training crack is not one straight segment: {source_path}")
            points = np.asarray(data["points"], dtype=np.float32)
            features = build_single_crack_local_features(
                points, geometry, h_min=h_min,
            )
            target_h = np.clip(
                np.asarray(data["target_h"], dtype=np.float32), h_min, h_max,
            )
            target = np.clip(
                (np.log(target_h) - log_min) / log_range, 0.0, 1.0,
            ).astype(np.float32)
            filename = Path(row["file"]).name
            np.savez_compressed(
                mesh_dir / filename,
                points=points,
                triangles=np.asarray(data["triangles"], dtype=np.int64),
                features=features,
                target_h=target_h,
                target_normalized_log_h=target,
                sample_weight=np.asarray(data["sample_weight"], dtype=np.float32),
                W=np.asarray(data["W"]),
                H=np.asarray(data["H"]),
                crack_polyline_points=np.asarray(data["crack_polyline_points"], dtype=np.float32),
                crack_polyline_offsets=np.asarray(data["crack_polyline_offsets"], dtype=np.int64),
                crack_tips=np.asarray(data["crack_tips"], dtype=np.float32),
                crack_entries=np.asarray(data["crack_entries"], dtype=np.float32),
                crack_family=np.asarray(data["crack_family"]),
                edge_side=np.asarray(data["edge_side"]),
            )
        if row["split"] == "train":
            feature_min = np.minimum(feature_min, features.min(axis=0))
            feature_max = np.maximum(feature_max, features.max(axis=0))
        converted_row = dict(row)
        converted_row["file"] = f"meshes/{filename}"
        converted.append(converted_row)
        total_points += len(points)
        if (index + 1) % 25 == 0:
            print(f"Converted {index + 1}/{len(rows)} straight-crack meshes")

    manifest = "single_crack_local_manifest.csv"
    with (output / manifest).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    output_metadata = {
        "format_version": 3,
        "model_family": SINGLE_CRACK_LOCAL_MODEL_FAMILY,
        "manifest": manifest,
        "target": "absolute_normalized_log_h",
        "target_source": "reference_mesh_points_and_triangles_only",
        "runtime_uses_analytic_size_field": False,
        "uses_fem_fields": False,
        "uses_gnn": False,
        "training_scope": "one_segment_straight_single_edge_cracks_only",
        "feature_names": SINGLE_CRACK_LOCAL_FEATURE_NAMES,
        "feature_normalization": "fixed_physical_bounded_no_dataset_zscore",
        "feature_train_min": feature_min.tolist(),
        "feature_train_max": feature_max.tolist(),
        "meshes": len(converted),
        "points": int(total_points),
        "split_counts": metadata["split_counts"],
        "family_counts": metadata["family_counts"],
        "edge_counts": metadata["edge_counts"],
        "h_min": h_min,
        "h_max": h_max,
        "mesh_size_definition": metadata.get("mesh_size_definition"),
        "source_dataset": str(source),
        "forbidden_labels": list(forbidden),
    }
    (output / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved: {output}")
    print(f"Meshes: {len(converted)} | points: {total_points}")
    print("Training shapes: one-segment straight edge cracks only")
    print("Runtime analytic size field: none")


if __name__ == "__main__":
    main()
