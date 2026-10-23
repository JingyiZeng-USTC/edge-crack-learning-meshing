"""Compare GNN and Gmsh connectivity on exactly the same nodes.

For each held-out straight-crack geometry, Gmsh first creates a locally refined
triangulation.  Its final (Relocate2D-optimized) nodes are frozen; the trained
GraphSAGE and strict structured decoder then reconnect those very nodes.
No learned size field, node smoothing, new nodes, or CDT fallback is used in
the GNN arm.  Gmsh's native triangles are the comparison arm, not ground truth.

Important: both arms use the single-node embedded crack-line representation
of the existing GNN test meshes.  This experiment measures planar connectivity;
it does not validate a mechanically opened, two-sided slit representation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import gmsh
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from common.mesh_validation import (  # noqa: E402
    recover_strict_gnn_project_mesh,
    topology_metrics,
    triangle_quality,
)
from common.meshing import edge_features, line_graph_edge_index, triangles_to_edges  # noqa: E402
from common.model import EdgeGraphSAGE  # noqa: E402
from common.project_data import project_candidate_edges  # noqa: E402
from gmsh_local_refinement_comparison.common_pipeline import generate_gmsh_mesh  # noqa: E402


DEFAULT_DATASET = PROJECT / "datasets" / "module30_single_straight_v1" / "meshes"
DEFAULT_TEST_CSV = ROOT / "artifacts" / "single_straight_200_structured_decoder_recovery_k20" / "per_mesh_metrics.csv"
DEFAULT_MODEL = ROOT / "artifacts" / "single_straight_200_strict_model.pt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "test40_fixed_nodes_gnn_vs_gmsh"


class GmshLocalField:
    """Match the Distance/Threshold/Min recipe used in experiment 19."""

    def __init__(self, scale: float):
        self.scale = float(scale)

    def install_gmsh(self, crack_line: int, crack_tip: int) -> None:
        field = gmsh.model.mesh.field
        thresholds = []
        for entity_list, entity, dist_min, dist_max, size_min in (
            ("PointsList", crack_tip, 0.008, 0.08, 0.004),
            ("CurvesList", crack_line, 0.004, 0.05, 0.008),
        ):
            distance = field.add("Distance")
            field.setNumbers(distance, entity_list, [entity])
            if entity_list == "CurvesList":
                field.setNumber(distance, "Sampling", 1000)
            threshold = field.add("Threshold")
            for key, value in (
                ("InField", distance),
                ("SizeMin", size_min * self.scale),
                ("SizeMax", 0.03 * self.scale),
                ("DistMin", dist_min),
                ("DistMax", dist_max),
                ("Sigmoid", 0),
            ):
                field.setNumber(threshold, key, value)
            thresholds.append(threshold)
        minimum = field.add("Min")
        field.setNumbers(minimum, "FieldsList", thresholds)
        field.setAsBackgroundMesh(minimum)


def canonical_coordinates(points: np.ndarray, edge: str, width: float, height: float):
    x, y = points[:, 0], points[:, 1]
    if edge == "left":
        return points.copy(), width, height
    if edge == "right":
        return np.column_stack((width - x, y)), width, height
    if edge == "bottom":
        return np.column_stack((y, x)), height, width
    if edge == "top":
        return np.column_stack((height - y, x)), height, width
    raise ValueError(f"Unknown edge side: {edge}")


def physical_coordinates(points: np.ndarray, edge: str, width: float, height: float):
    x, y = points[:, 0], points[:, 1]
    if edge == "left":
        return points.copy()
    if edge == "right":
        return np.column_stack((width - x, y))
    if edge == "bottom":
        return np.column_stack((y, x))
    if edge == "top":
        return np.column_stack((y, height - x))
    raise ValueError(f"Unknown edge side: {edge}")


def generate_shared_nodes(
    name: str,
    crack: np.ndarray,
    edge: str,
    width: float,
    height: float,
    target_nodes: int,
    attempts: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    canonical_crack, canonical_width, canonical_height = canonical_coordinates(
        crack, edge, width, height
    )
    scale = 1.0
    best = None
    trials = []
    for attempt in range(attempts):
        points, triangles, _ = generate_gmsh_mesh(
            GmshLocalField(scale),
            width=canonical_width,
            height=canonical_height,
            start=canonical_crack[0],
            tip=canonical_crack[1],
            algorithm=6,
            optimize=True,
            model_name=f"fixed_nodes_{name}_{attempt}",
            open_crack=False,
        )
        difference = abs(len(points) - target_nodes) / target_nodes
        trials.append({"size_scale": scale, "nodes": len(points), "relative_difference": difference})
        if best is None or difference < best[0]:
            best = (difference, points.copy(), triangles.copy(), scale)
        if difference <= 0.03:
            break
        scale = float(np.clip(scale * math.sqrt(len(points) / target_nodes), 0.4, 3.0))
    assert best is not None
    difference, canonical_points, triangles, chosen_scale = best
    points = physical_coordinates(canonical_points, edge, width, height).astype(np.float32)
    if len(np.unique(points, axis=0)) != len(points):
        raise RuntimeError("Gmsh node coordinates are not unique after float32 conversion")
    return points, triangles, {
        "gmsh_size_scale": chosen_scale,
        "reference_node_count": target_nodes,
        "gmsh_reference_node_count_difference": difference,
        "gmsh_size_trials": trials,
    }


def mesh_check(points, triangles, width, height, crack_segments):
    quality = triangle_quality(points, triangles)
    topology = topology_metrics(points, triangles, width, height, crack_segments)
    vertices = points[triangles].astype(np.float64)
    twice_area = np.abs(
        (vertices[:, 1, 0] - vertices[:, 0, 0]) * (vertices[:, 2, 1] - vertices[:, 0, 1])
        - (vertices[:, 1, 1] - vertices[:, 0, 1]) * (vertices[:, 2, 0] - vertices[:, 0, 0])
    )
    area_error = abs(0.5 * float(twice_area.sum()) - width * height) / (width * height)
    valid = (
        area_error <= 5e-6
        and topology["components"] == 1
        and quality["degenerate_triangles"] == 0
        and all(topology[key] == 0 for key in (
            "isolated_nodes", "nonmanifold_edges", "missing_constraint_edges", "duplicate_triangles"
        ))
    )
    return {
        "valid": bool(valid),
        "relative_area_error": float(area_error),
        "quality": quality,
        "topology": topology,
    }


def edge_overlap(reference_edges: np.ndarray, recovered_edges: np.ndarray) -> dict:
    original = {tuple(map(int, edge)) for edge in reference_edges}
    predicted = {tuple(map(int, edge)) for edge in recovered_edges}
    intersection = len(original & predicted)
    precision = intersection / len(predicted) if predicted else 0.0
    recall = intersection / len(original) if original else 0.0
    return {
        "edge_overlap_precision": precision,
        "edge_overlap_recall": recall,
        "edge_overlap_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], requested: int) -> dict:
    result = {
        "requested_cases": requested,
        "attempted_cases": len(rows),
        "completed_cases": sum(bool(row.get("gmsh_valid") and row.get("gnn_valid")) for row in rows),
    }
    for method in ("gmsh", "gnn"):
        valid = [row for row in rows if row.get(f"{method}_valid") is True]
        minima = np.asarray([row[f"{method}_min_angle_deg"] for row in valid], dtype=float)
        result[method] = {
            "valid_cases": len(valid),
            "mean_case_min_angle_deg": float(minima.mean()) if len(minima) else None,
            "worst_case_min_angle_deg": float(minima.min()) if len(minima) else None,
            "cases_min_angle_ge_20deg": int(np.count_nonzero(minima >= 20.0)),
            "mean_fraction_triangles_below_20deg": float(np.mean(
                [row[f"{method}_fraction_below_20deg"] for row in valid]
            )) if valid else None,
        }
    paired = [row for row in rows if row.get("gmsh_valid") and row.get("gnn_valid")]
    deltas = np.asarray([
        row["gnn_min_angle_deg"] - row["gmsh_min_angle_deg"] for row in paired
    ], dtype=float)
    coverage_rows = [row for row in rows if "gmsh_edge_candidate_coverage" in row]
    result["paired"] = {
        "cases": len(paired),
        "mean_gnn_minus_gmsh_min_angle_deg": float(deltas.mean()) if len(deltas) else None,
        "gnn_higher_min_angle_cases": int(np.count_nonzero(deltas > 0)),
        "mean_gmsh_edge_candidate_coverage": float(np.mean([
            row["gmsh_edge_candidate_coverage"] for row in coverage_rows
        ])) if coverage_rows else None,
        "mean_edge_overlap_f1_descriptive_only": float(np.mean([
            row["edge_overlap_f1"] for row in paired
        ])) if paired else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--case", type=str, default=None, help="Run one held-out case by name")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--gmsh-attempts", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-meshes", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or args.candidate_k < 2 or args.gmsh_attempts < 1:
        parser.error("--limit, --gmsh-attempts must be positive and --candidate-k >= 2")

    with args.test_csv.open(newline="", encoding="utf-8-sig") as handle:
        test_names = [row["name"] for row in csv.DictReader(handle)]
    if len(test_names) != 40 or len(set(test_names)) != 40:
        raise RuntimeError("The existing test CSV must list exactly 40 unique held-out cases")
    if args.case:
        if args.case not in test_names:
            parser.error(f"--case {args.case!r} is not one of the 40 held-out cases")
        names = [args.case]
    else:
        names = test_names[:args.limit]

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model = EdgeGraphSAGE(
        in_channels=int(checkpoint["in_channels"]),
        hidden_channels=int(checkpoint["hidden"]),
        layers=int(checkpoint["layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    threshold = float(checkpoint["threshold"])

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mesh_dir = output / "meshes"
    if args.save_meshes:
        mesh_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    details = []
    failures = []
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 0)
    try:
        for index, name in enumerate(names, start=1):
            started = time.perf_counter()
            row = {"name": name, "gmsh_valid": False, "gnn_valid": False}
            detail = {"name": name}
            try:
                with np.load(args.dataset / f"{name}.npz", allow_pickle=False) as source:
                    width, height = float(source["W"]), float(source["H"])
                    edge = str(source["edge_side"])
                    crack = np.asarray(source["crack_polyline_points"], dtype=np.float64)
                    target_nodes = len(source["points"])
                if crack.shape != (2, 2):
                    raise ValueError("Only one-segment straight cracks are supported")
                crack_segments = crack[None].astype(np.float32)
                gmsh_start = time.perf_counter()
                points, gmsh_triangles, generation = generate_shared_nodes(
                    name, crack, edge, width, height, target_nodes, args.gmsh_attempts
                )
                gmsh_seconds = time.perf_counter() - gmsh_start
                gmsh_check = mesh_check(points, gmsh_triangles, width, height, crack_segments)
                if not gmsh_check["valid"]:
                    raise RuntimeError(f"Gmsh mesh failed structural checks: {gmsh_check}")
                gt_edges = triangles_to_edges(gmsh_triangles)
                row.update({
                    "edge_side": edge,
                    "nodes": len(points),
                    "gmsh_valid": True,
                    "gmsh_triangles": len(gmsh_triangles),
                    "gmsh_min_angle_deg": gmsh_check["quality"]["min_angle_deg"],
                    "gmsh_angle_p05_deg": gmsh_check["quality"]["angle_p05_deg"],
                    "gmsh_fraction_below_20deg": gmsh_check["quality"]["fraction_below_20deg"],
                    "gmsh_area_error": gmsh_check["relative_area_error"],
                    "gmsh_seconds": gmsh_seconds,
                    "node_count_vs_reference_relative_difference": generation["gmsh_reference_node_count_difference"],
                })
                detail.update({"generation": generation, "gmsh_check": gmsh_check})

                candidates = project_candidate_edges(
                    points, width, height, crack_segments, k=args.candidate_k
                )
                candidate_set = {tuple(map(int, edge)) for edge in candidates}
                gt_set = {tuple(map(int, edge)) for edge in gt_edges}
                coverage = len(candidate_set & gt_set) / len(gt_set)
                row.update({
                    "candidates": len(candidates),
                    "gmsh_edges": len(gt_edges),
                    "gmsh_edge_candidate_coverage": coverage,
                })
                graph_start = time.perf_counter()
                features = edge_features(points, candidates, width, height)
                line_edges = line_graph_edge_index(candidates)
                graph_seconds = time.perf_counter() - graph_start
                inference_start = time.perf_counter()
                with torch.no_grad():
                    probability = model(
                        torch.from_numpy(features).to(device),
                        torch.from_numpy(line_edges).to(device),
                    ).softmax(dim=1)[:, 1].cpu().numpy()
                inference_seconds = time.perf_counter() - inference_start
                del features, line_edges
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                decode_start = time.perf_counter()
                recovered_points, gnn_triangles, selected_edges, decoder = (
                    recover_strict_gnn_project_mesh(
                        points, candidates, probability, width, height,
                        crack_segments, threshold=threshold, quality_passes=0,
                    )
                )
                decode_seconds = time.perf_counter() - decode_start
                if not np.array_equal(points, recovered_points):
                    raise RuntimeError("GNN decoder moved or inserted nodes")
                gnn_check = mesh_check(points, gnn_triangles, width, height, crack_segments)
                if not gnn_check["valid"]:
                    raise RuntimeError(f"GNN mesh failed structural checks: {gnn_check}")
                overlap = edge_overlap(gt_edges, triangles_to_edges(gnn_triangles))
                row.update({
                    "gnn_valid": True,
                    "gnn_triangles": len(gnn_triangles),
                    "gnn_min_angle_deg": gnn_check["quality"]["min_angle_deg"],
                    "gnn_angle_p05_deg": gnn_check["quality"]["angle_p05_deg"],
                    "gnn_fraction_below_20deg": gnn_check["quality"]["fraction_below_20deg"],
                    "gnn_area_error": gnn_check["relative_area_error"],
                    "graph_seconds": graph_seconds,
                    "inference_seconds": inference_seconds,
                    "decode_seconds": decode_seconds,
                    "gnn_selected_edges": len(selected_edges),
                    **overlap,
                })
                detail.update({"gnn_check": gnn_check, "decoder": decoder})
                if args.save_meshes:
                    np.savez_compressed(
                        mesh_dir / f"{name}.npz", nodes=points,
                        gmsh_triangles=gmsh_triangles, gnn_triangles=gnn_triangles,
                        crack_segments=crack_segments, candidates=candidates,
                        probability=probability,
                    )
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({"name": name, "error": row["error"]})
            row["total_seconds"] = time.perf_counter() - started
            rows.append(row)
            details.append(detail)
            print(
                f"[{index:02d}/{len(names)}] {name}: "
                f"Gmsh={row.get('gmsh_min_angle_deg', float('nan')):.3f}°, "
                f"GNN={row.get('gnn_min_angle_deg', float('nan')):.3f}°, "
                f"coverage={row.get('gmsh_edge_candidate_coverage', float('nan')):.4f} "
                f"{'OK' if row['gnn_valid'] else row.get('error', 'FAIL')}",
                flush=True,
            )
    finally:
        gmsh.finalize()

    summary = {
        "experiment": "fixed-nodes-gmsh-versus-gnn-connectivity",
        "node_source": "Gmsh final Relocate2D-optimized nodes; exact same float32 coordinate array for both arms",
        "geometry": "40 held-out one-segment straight edge cracks",
        "crack_representation": "single-node embedded crack line; not a mechanically opened slit",
        "gmsh_connectivity": "Distance/Threshold/Min field; 2D Frontal-Delaunay algorithm 6; Relocate2D before node freeze",
        "gnn_connectivity": "trained GraphSAGE, kNN candidates, strict structured decoder; no node moves, no flips, no CDT fallback",
        "gmsh_is_not_ground_truth": True,
        "edge_overlap_f1_is_descriptive_only": True,
        "candidate_k": args.candidate_k,
        "model": str(args.model.resolve()),
        "gmsh_version": gmsh.__version__,
        "statistics": summarize(rows, len(names)),
    }
    write_csv(output / "per_case_metrics.csv", rows)
    (output / "case_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(f"Saved results to {output}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)}/{len(names)} cases failed; see failures.json")


if __name__ == "__main__":
    main()
