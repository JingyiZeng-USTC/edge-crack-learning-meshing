"""Export learned-size/strict-GNN single-crack meshes to downstream FCNN-SM."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# This adapter lives under ``gnn`` while the established FCNN/FEM modules live
# one directory above it.  Keep those modules unchanged and reuse their public
# mesh format and crack-face opening implementation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arbitrary_crack_mesh import ArbitraryCrackMesh
from crack_graph_geometry import CrackGraphGeometry
from infer_arbitrary_crack import save_mesh_npz, save_mesh_vtu
from module9_crack_graph_nodes import duplicate_crack_graph_nodes

from common.mesh_validation import triangle_quality


def _single_polyline_geometry(item: np.lib.npyio.NpzFile) -> CrackGraphGeometry:
    width = float(np.asarray(item["width"]).reshape(()))
    height = float(np.asarray(item["height"]).reshape(()))
    if "crack_polyline_points" in item.files:
        points = np.asarray(item["crack_polyline_points"], dtype=np.float64).reshape(-1, 2)
        offsets = np.asarray(
            item.get("crack_polyline_offsets", np.asarray([0, len(points)])),
            dtype=np.int64,
        ).reshape(-1)
        if len(offsets) != 2 or int(offsets[0]) != 0 or int(offsets[1]) != len(points):
            raise ValueError("FCNN adapter requires exactly one crack polyline")
        polyline = points
    else:
        segments = np.asarray(item["crack_segments"], dtype=np.float64).reshape(-1, 2, 2)
        if len(segments) != 1:
            raise ValueError(
                "A multi-segment candidate file must include crack_polyline_points"
            )
        polyline = segments[0]
    geometry = CrackGraphGeometry(width, height, [polyline])
    if geometry.has_branches or len(geometry.crack_entries) != 1 or len(geometry.crack_tips) != 1:
        raise ValueError("FCNN adapter supports one non-branching single-edge crack only")
    return geometry


def _bind_recovered_crack_chain(
    geometry: CrackGraphGeometry,
    points: np.ndarray,
    *,
    tolerance: float = 2.0e-6,
) -> None:
    """Use actual recovered crack nodes instead of re-discretizing the segment."""
    polyline = np.asarray(geometry.crack_polylines[0], dtype=np.float64)
    start, end = polyline[0], polyline[-1]
    values = np.asarray(points, dtype=np.float64)
    chain_parts: list[np.ndarray] = []
    for segment_start, segment_end in zip(polyline[:-1], polyline[1:]):
        direction = segment_end - segment_start
        length2 = float(direction @ direction)
        if length2 <= tolerance * tolerance:
            raise ValueError("Crack polyline contains a zero-length segment")
        relative = values - segment_start
        parameter = (relative @ direction) / length2
        distance = np.abs(
            direction[0] * relative[:, 1] - direction[1] * relative[:, 0]
        ) / np.sqrt(length2)
        identifiers = np.flatnonzero(
            (distance <= tolerance)
            & (parameter >= -tolerance)
            & (parameter <= 1.0 + tolerance)
        )
        identifiers = identifiers[np.argsort(parameter[identifiers])]
        if len(identifiers) < 2:
            raise ValueError("Recovered mesh is missing an atomic crack-polyline segment")
        part = values[identifiers].copy()
        if chain_parts and np.linalg.norm(chain_parts[-1][-1] - part[0]) <= tolerance * 2.0:
            part = part[1:]
        chain_parts.append(part)
    chain = np.vstack(chain_parts)
    if np.linalg.norm(chain[0] - start) > tolerance * 2.0:
        raise ValueError("Recovered crack chain is missing its boundary entry")
    if np.linalg.norm(chain[-1] - end) > tolerance * 2.0:
        raise ValueError("Recovered crack chain is missing its tip")
    # CrackGraphGeometry normally creates 30 evenly-spaced chain points for a
    # newly sampled mesh.  GNN recovery already has its own atomic crack nodes,
    # including any Triangle split points, so the adapter binds that exact chain
    # before calling the established crack-face duplication routine.
    geometry._discretized_chains = (chain,)


def export_one(
    recovered_path: Path,
    candidate_path: Path,
    output_dir: Path,
    *,
    write_vtu: bool,
) -> dict[str, object]:
    recovered = np.load(recovered_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)
    geometry = _single_polyline_geometry(candidate)
    shared_points = np.asarray(recovered["nodes"], dtype=np.float64)
    shared_triangles = np.asarray(recovered["triangles"], dtype=np.int64)
    if len(shared_points) < 3 or len(shared_triangles) < 1:
        raise ValueError(f"Recovered mesh is empty: {recovered_path}")
    candidate_points = np.asarray(candidate["nodes"], dtype=np.float64)
    if candidate_points.shape != shared_points.shape:
        raise ValueError(
            "Candidate and recovered meshes do not use the same node-index set; "
            f"refusing to couple different cases: {candidate_path.name}"
        )
    displacement = np.linalg.norm(shared_points - candidate_points, axis=1)
    maximum_displacement = float(displacement.max(initial=0.0))
    if "learned_h" in candidate.files:
        candidate_h = np.asarray(candidate["learned_h"], dtype=np.float64).reshape(-1)
        if len(candidate_h) != len(candidate_points) or np.any(candidate_h <= 0.0):
            raise ValueError(f"Candidate learned_h is invalid: {candidate_path}")
        displacement_ratio = displacement / candidate_h
        maximum_displacement_over_h = float(displacement_ratio.max(initial=0.0))
        if maximum_displacement_over_h > 1.0 + 1.0e-8:
            raise ValueError(
                "Recovered coordinates moved by more than one local learned size; "
                f"max(|dx|/h)={maximum_displacement_over_h:.6g}"
            )
    elif not np.allclose(candidate_points, shared_points, rtol=0.0, atol=2.0e-6):
        raise ValueError(
            "Smoothed recovered coordinates require learned_h in the candidate file"
        )
    else:
        maximum_displacement_over_h = 0.0

    tolerance = 2.0e-6 * max(float(geometry.W), float(geometry.H), 1.0)
    original_boundary = (
        np.isclose(candidate_points[:, 0], 0.0, atol=tolerance)
        | np.isclose(candidate_points[:, 0], geometry.W, atol=tolerance)
        | np.isclose(candidate_points[:, 1], 0.0, atol=tolerance)
        | np.isclose(candidate_points[:, 1], geometry.H, atol=tolerance)
    )
    recovered_boundary = (
        np.isclose(shared_points[:, 0], 0.0, atol=tolerance)
        | np.isclose(shared_points[:, 0], geometry.W, atol=tolerance)
        | np.isclose(shared_points[:, 1], 0.0, atol=tolerance)
        | np.isclose(shared_points[:, 1], geometry.H, atol=tolerance)
    )
    if not np.all(recovered_boundary[original_boundary]):
        raise ValueError("Quality smoothing moved an exterior-boundary node off the boundary")
    if not np.all(np.isfinite(shared_points)):
        raise ValueError(f"Recovered mesh contains non-finite coordinates: {recovered_path}")
    if int(shared_triangles.min()) < 0 or int(shared_triangles.max()) >= len(shared_points):
        raise ValueError(f"Recovered mesh contains an invalid triangle index: {recovered_path}")

    _bind_recovered_crack_chain(geometry, shared_points)
    open_triangles, open_points, crack_info = duplicate_crack_graph_nodes(
        shared_triangles,
        shared_points,
        geometry,
    )
    quality_values = triangle_quality(open_points, open_triangles)
    source_node_count = len(candidate_points)
    recovery_steiner_points = len(shared_points) - source_node_count
    quality = {
        "minimum_angle_deg": float(quality_values["min_angle_deg"]),
        "angle_p05_deg": float(quality_values["angle_p05_deg"]),
        "mean_minimum_angle_deg": float(quality_values["angle_median_deg"]),
        "degenerate_elements": int(quality_values["degenerate_triangles"]),
        "steiner_points": int(recovery_steiner_points),
        "requested_minimum_angle_deg": 0.0,
        "quality_target_met": int(quality_values["degenerate_triangles"]) == 0,
    }
    constraint_values = np.asarray(
        recovered.get("recovery_constraints", np.empty((0, 2), dtype=np.int64)),
        dtype=np.int64,
    ).reshape(-1, 2)
    mesh = ArbitraryCrackMesh(
        points=np.asarray(open_points, dtype=np.float64),
        triangles=np.asarray(open_triangles, dtype=np.int32),
        crack_info=crack_info,
        constraints={tuple(sorted(map(int, edge))) for edge in constraint_values},
        quality=quality,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{recovered_path.stem}.npz"
    save_mesh_npz(npz_path, mesh, geometry)
    vtu_path = output_dir / f"{recovered_path.stem}.vtu"
    if write_vtu:
        save_mesh_vtu(vtu_path, mesh, geometry)

    positive = np.asarray(sorted(crack_info.get("crack_upper_nodes", ())), dtype=np.int64)
    negative = np.asarray(sorted(crack_info.get("crack_lower_nodes", ())), dtype=np.int64)
    tips = np.asarray(sorted(crack_info.get("crack_tip_nodes", ())), dtype=np.int64)
    return {
        "name": recovered_path.stem,
        "source_nodes": source_node_count,
        "shared_recovered_nodes": len(shared_points),
        "fcnn_nodes": len(open_points),
        "triangles": len(open_triangles),
        "recovery_steiner_points": recovery_steiner_points,
        "duplicated_crack_face_nodes": len(open_points) - len(shared_points),
        "maximum_quality_smoothing_displacement": maximum_displacement,
        "maximum_quality_smoothing_displacement_over_h": maximum_displacement_over_h,
        "crack_positive_nodes": len(positive),
        "crack_negative_nodes": len(negative),
        "crack_tip_nodes": len(tips),
        "minimum_angle_deg": quality["minimum_angle_deg"],
        "angle_p05_deg": quality["angle_p05_deg"],
        "degenerate_elements": quality["degenerate_elements"],
        "npz": str(npz_path.resolve()),
        "vtu": str(vtu_path.resolve()) if write_vtu else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert planar GNN recovery output to the existing downstream FCNN-SM pure-mesh format."
    )
    parser.add_argument(
        "--recovery",
        type=Path,
        default=Path(
            "artifacts/learned_size_gnn_unseen_k20_prioritymerge05/meshes"
        ),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "artifacts/learned_size_gnn_unseen_k20_prioritymerge05/candidates"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/learned_size_gnn_unseen_k20_prioritymerge05/fcnn_meshes"
        ),
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Mesh stem to export; repeat for multiple meshes. Default: all recovered meshes.",
    )
    parser.add_argument("--vtu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.name:
        recovered_paths = [args.recovery / f"{name}.npz" for name in args.name]
    else:
        recovered_paths = sorted(args.recovery.glob("*.npz"))
    if not recovered_paths:
        raise FileNotFoundError(f"No recovered NPZ meshes found under {args.recovery}")

    rows: list[dict[str, object]] = []
    for index, recovered_path in enumerate(recovered_paths, start=1):
        if not recovered_path.is_file():
            raise FileNotFoundError(recovered_path)
        candidate_path = args.candidates / recovered_path.name
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        row = export_one(recovered_path, candidate_path, args.output, write_vtu=args.vtu)
        rows.append(row)
        print(
            f"[{index:02d}/{len(recovered_paths)}] {row['name']} -> "
            f"nodes={row['fcnn_nodes']} triangles={row['triangles']} "
            f"crack copies={row['duplicated_crack_face_nodes']}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "fcnn_mesh_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format": "learned-size-strict-gnn-pure-mesh-for-module44-fcnn-sm-v2",
        "scope": "single-nonbranching-polyline-crack-only",
        "source_recovery": str(args.recovery.resolve()),
        "source_candidates": str(args.candidates.resolve()),
        "crack_faces_opened_by_node_duplication": True,
        "fcnn_loading_contract": "top-uniform-line-load-no-pcg",
        "meshes": len(rows),
        "total_nodes": int(sum(int(row["fcnn_nodes"]) for row in rows)),
        "total_triangles": int(sum(int(row["triangles"]) for row in rows)),
        "degenerate_elements": int(sum(int(row["degenerate_elements"]) for row in rows)),
        "files": [str(row["npz"]) for row in rows],
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
