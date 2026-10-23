"""Paired mesh-angle comparison on the existing 40 straight-crack test cases.

The first two arms use the *same* GNN-scored candidate edges and original
reference-node locations.  The Gmsh arm uses the same crack geometry, Gmsh's
native Distance/Threshold refinement, and a calibrated geometric-node budget.
No reference triangles are used to construct either compared mesh.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import gmsh
import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PROJECT))

from crack_graph_geometry import CrackGraphGeometry
from module30_learned_size_field.single_crack_local_field import SingleCrackLocalSizeField
from common.mesh_validation import (
    recover_strict_gnn_project_mesh,
    topology_metrics,
    triangle_quality,
)
from common.quality_smoothing import constrained_quality_smoothing
from gmsh_local_refinement_comparison.common_pipeline import generate_gmsh_mesh


DATASET = PROJECT / "datasets" / "module30_single_straight_v1" / "meshes"
GNN_RESULTS = ROOT / "artifacts" / "single_straight_200_structured_decoder_recovery_k20"
SIZE_MODEL = PROJECT / "module30_checkpoints" / "single_straight_crack_local_v4.pt"
DEFAULT_OUTPUT = ROOT / "artifacts" / "test40_three_way_mesh_quality_v1"


def _cross2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _init_gmsh() -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.RandomFactor", 0)
    atexit.register(gmsh.finalize)


def canonical_coordinates(
    points: np.ndarray, edge: str, width: float, height: float,
) -> tuple[np.ndarray, float, float]:
    """Map any edge crack to the left edge by an isometry."""
    x, y = points[:, 0], points[:, 1]
    if edge == "left":
        return points.copy(), width, height
    if edge == "right":
        return np.column_stack((width - x, y)), width, height
    if edge == "bottom":
        return np.column_stack((y, x)), height, width
    if edge == "top":
        return np.column_stack((height - y, x)), height, width
    raise ValueError(f"Unsupported edge side: {edge}")


def physical_coordinates(
    points: np.ndarray, edge: str, width: float, height: float,
) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    if edge == "left":
        return points.copy()
    if edge == "right":
        return np.column_stack((width - x, y))
    if edge == "bottom":
        return np.column_stack((y, x))
    if edge == "top":
        return np.column_stack((y, height - x))
    raise ValueError(f"Unsupported edge side: {edge}")


class GmshLocalField:
    """Same Gmsh Distance/Threshold/Min baseline as run_comparison.py."""

    def __init__(self, scale: float) -> None:
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


def gmsh_at_matched_budget(
    start: np.ndarray,
    tip: np.ndarray,
    edge: str,
    width: float,
    height: float,
    target_nodes: int,
    attempts: int,
    case_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    transformed, canonical_width, canonical_height = canonical_coordinates(
        np.stack((start, tip)), edge, width, height,
    )
    scale = 1.0
    best = None
    history = []
    for attempt in range(attempts):
        nodes, triangles, raw = generate_gmsh_mesh(
            GmshLocalField(scale),
            width=canonical_width,
            height=canonical_height,
            start=transformed[0],
            tip=transformed[1],
            algorithm=6,
            optimize=True,
            model_name=f"{case_name}_{attempt}",
        )
        geometric_nodes = len(nodes) - int(raw["duplicated_crack_nodes"])
        error = abs(geometric_nodes - target_nodes) / target_nodes
        history.append({
            "scale": scale,
            "geometric_nodes": geometric_nodes,
            "relative_node_count_difference": error,
        })
        if best is None or error < best[0]:
            best = (error, nodes.copy(), triangles.copy(), geometric_nodes, scale)
        if error <= 0.03:
            break
        # Mesh count is approximately proportional to 1 / h^2.
        scale = float(np.clip(scale * math.sqrt(geometric_nodes / target_nodes), 0.4, 3.0))
    assert best is not None
    error, canonical_nodes, triangles, geometric_nodes, chosen_scale = best
    nodes = physical_coordinates(canonical_nodes, edge, width, height)
    quality = triangle_quality(nodes, triangles)
    vertex_ids = np.unique(triangles)
    if len(vertex_ids) != len(nodes):
        raise RuntimeError(f"Gmsh produced {len(nodes)-len(vertex_ids)} isolated nodes")
    element = nodes[triangles]
    area = 0.5 * abs(_cross2d(element[:, 1] - element[:, 0], element[:, 2] - element[:, 0])).sum()
    relative_area_error = abs(float(area) - width * height) / (width * height)
    if relative_area_error > 1e-8 or quality["degenerate_triangles"]:
        raise RuntimeError(
            f"Invalid Gmsh mesh: area error={relative_area_error}, "
            f"degenerate={quality['degenerate_triangles']}"
        )
    info = {
        "quality": quality,
        "nodes": len(nodes),
        "geometric_nodes": geometric_nodes,
        "node_count_relative_difference": float(error),
        "size_scale": float(chosen_scale),
        "relative_area_error": relative_area_error,
        "attempts": history,
    }
    return nodes, triangles, info


def checked_gnn_mesh(
    nodes: np.ndarray,
    triangles: np.ndarray,
    width: float,
    height: float,
    crack_segments: np.ndarray,
) -> dict:
    quality = triangle_quality(nodes, triangles)
    topology = topology_metrics(nodes, triangles, width, height, crack_segments)
    element = nodes[triangles].astype(float)
    area = 0.5 * abs(_cross2d(element[:, 1] - element[:, 0], element[:, 2] - element[:, 0])).sum()
    relative_area_error = abs(float(area) - width * height) / (width * height)
    valid = (
        relative_area_error <= 1e-8
        and topology["components"] == 1
        and all(topology[key] == 0 for key in (
            "isolated_nodes", "nonmanifold_edges", "missing_constraint_edges",
            "duplicate_triangles",
        ))
        and quality["degenerate_triangles"] == 0
    )
    if not valid:
        raise RuntimeError(f"Invalid GNN mesh: topology={topology}, area={relative_area_error}")
    return {
        "quality": quality,
        "nodes": len(nodes),
        "geometric_nodes": len(nodes),
        "relative_area_error": relative_area_error,
        "topology": topology,
    }


def run_case(name: str, args: argparse.Namespace, mesh_dir: Path) -> dict:
    source_path = DATASET / f"{name}.npz"
    gnn_path = GNN_RESULTS / "meshes" / f"{name}.npz"
    if not source_path.is_file() or not gnn_path.is_file():
        raise FileNotFoundError(f"Missing source or recovered mesh for {name}")
    with np.load(source_path, allow_pickle=False) as source:
        width, height = float(source["W"]), float(source["H"])
        edge = str(source["edge_side"])
        crack = np.asarray(source["crack_polyline_points"], dtype=float)
        reference_nodes = len(source["points"])
    if crack.shape != (2, 2):
        raise ValueError(f"{name} is not a one-segment straight crack")
    with np.load(gnn_path, allow_pickle=False) as saved:
        nodes = np.asarray(saved["nodes"], dtype=float)
        baseline_triangles = np.asarray(saved["triangles"], dtype=np.int64)
        candidates = np.asarray(saved["candidates"], dtype=np.int64)
        probability = np.asarray(saved["probability"], dtype=float)
        crack_segments = np.asarray(saved["crack_segments"], dtype=float)
    if len(nodes) != reference_nodes:
        raise RuntimeError(f"{name}: reference-node count differs from GNN node count")
    baseline = checked_gnn_mesh(nodes, baseline_triangles, width, height, crack_segments)

    started = time.perf_counter()
    post_nodes, post_triangles, _, redecode = recover_strict_gnn_project_mesh(
        nodes, candidates, probability, width, height, crack_segments,
        quality_target_deg=20.0,
        quality_weight=12.0,
        quality_passes=12,
    )
    geometry = CrackGraphGeometry(width, height, [crack])
    size_field = SingleCrackLocalSizeField(
        geometry,
        checkpoint=str(args.size_model),
        grid_nx=161,
        grid_ny=321,
        smooth_sigma=1.0,
        complexity_ratio=1.0,
        device=args.device,
    )
    post_nodes, smoothing = constrained_quality_smoothing(
        post_nodes, post_triangles, size_field.evaluate_bulk(post_nodes),
        width, height, crack_segments,
        target_angle_deg=20.0,
        max_passes=10,
        move_factor=0.30,
    )
    postprocess_seconds = time.perf_counter() - started
    post = checked_gnn_mesh(post_nodes, post_triangles, width, height, crack_segments)
    post["quality_replacements"] = int(redecode["structured_quality_replacements"])
    post["smoothing_moves"] = int(smoothing["accepted_moves"])
    post["seconds"] = postprocess_seconds
    post["all_edges_gnn_scored"] = True

    started = time.perf_counter()
    gmsh_nodes, gmsh_triangles, gmsh_info = gmsh_at_matched_budget(
        crack[0], crack[1], edge, width, height, len(nodes),
        args.gmsh_attempts, name,
    )
    gmsh_info["seconds"] = time.perf_counter() - started
    if args.save_meshes:
        np.savez_compressed(
            mesh_dir / f"{name}_post.npz",
            nodes=post_nodes, triangles=post_triangles, crack_segments=crack_segments,
        )
        np.savez_compressed(
            mesh_dir / f"{name}_gmsh.npz",
            nodes=gmsh_nodes, triangles=gmsh_triangles, crack_segments=crack_segments,
        )
    return {
        "name": name,
        "edge_side": edge,
        "geometry": {"width": width, "height": height,
                     "start": crack[0].tolist(), "tip": crack[1].tolist()},
        "gnn": baseline,
        "gnn_post": post,
        "gmsh": gmsh_info,
    }


def summarize(rows: list[dict]) -> dict:
    output = {}
    for method in ("gnn", "gnn_post", "gmsh"):
        subset = [row[method] for row in rows if method in row]
        if not subset:
            output[method] = {"cases": 0}
            continue
        minima = np.asarray([item["quality"]["min_angle_deg"] for item in subset])
        output[method] = {
            "cases": len(subset),
            "mean_of_case_min_angle_deg": float(minima.mean()),
            "minimum_case_min_angle_deg": float(minima.min()),
            "median_case_min_angle_deg": float(np.median(minima)),
            "cases_min_angle_ge_20deg": int(np.count_nonzero(minima >= 20.0)),
            "mean_nodes": float(np.mean([item["geometric_nodes"] for item in subset])),
            "mean_fraction_of_triangles_below_20deg": float(np.mean(
                [item["quality"]["fraction_below_20deg"] for item in subset]
            )),
        }
    if rows:
        output["gmsh_node_matching"] = {
            "maximum_relative_difference": float(max(
                row["gmsh"]["node_count_relative_difference"] for row in rows
                if "gmsh" in row
            )),
            "mean_relative_difference": float(np.mean([
                row["gmsh"]["node_count_relative_difference"] for row in rows
                if "gmsh" in row
            ])),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size-model", type=Path, default=SIZE_MODEL)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--gmsh-attempts", type=int, default=7)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=40, help="Use 1 for a smoke test")
    parser.add_argument("--save-meshes", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.limit <= 40 or args.gmsh_attempts < 1 or args.workers < 1:
        parser.error("--limit must be 1..40; --gmsh-attempts and --workers must be positive")
    if not args.size_model.is_file():
        raise FileNotFoundError(args.size_model)
    metrics_path = GNN_RESULTS / "per_mesh_metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        names = [row["name"] for row in csv.DictReader(handle)]
    if len(names) != 40 or len(set(names)) != 40:
        raise RuntimeError("Expected exactly 40 distinct test case names")
    output = args.output.resolve()
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = output / "meshes"
    if args.save_meshes:
        mesh_dir.mkdir(exist_ok=True)
    failures = []
    pending = [name for name in names[:args.limit] if not (cases_dir / f"{name}.json").exists()]
    print(f"Running {len(pending)} new cases with {args.workers} worker(s)", flush=True)

    def record(name: str, row: dict) -> None:
        (cases_dir / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"[{name}] "
            f"{row['gnn']['quality']['min_angle_deg']:.3f} / "
            f"{row['gnn_post']['quality']['min_angle_deg']:.3f} / "
            f"{row['gmsh']['quality']['min_angle_deg']:.3f} deg; "
            f"Gmsh node delta {100*row['gmsh']['node_count_relative_difference']:.2f}%",
            flush=True,
        )

    if args.workers == 1:
        _init_gmsh()
        try:
            for name in pending:
                try:
                    record(name, run_case(name, args, mesh_dir))
                except Exception as exc:
                    failures.append({"name": name, "error": str(exc)})
                    print(f"[{name}] FAILED: {exc}", flush=True)
        finally:
            gmsh.finalize()
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_gmsh) as pool:
            jobs = {pool.submit(run_case, name, args, mesh_dir): name for name in pending}
            for job in as_completed(jobs):
                name = jobs[job]
                try:
                    record(name, job.result())
                except Exception as exc:
                    failures.append({"name": name, "error": str(exc)})
                    print(f"[{name}] FAILED: {exc}", flush=True)
    rows = [
        json.loads((cases_dir / f"{name}.json").read_text(encoding="utf-8"))
        for name in names[:args.limit] if (cases_dir / f"{name}.json").exists()
    ]
    result = {
        "scope": "same 40 test-case geometries; GNN arms share original reference nodes",
        "gnn_baseline": "existing k=20 structured-decoder recovery, before 20-degree quality postprocessing",
        "gnn_post": "rerank scored candidate diagonals at 20 degrees, then learned-h constrained smoothing",
        "gmsh_baseline": "native Distance + Threshold + Min, algorithm 6, Relocate2D, geometric node budget matched per case",
        "requested_cases": args.limit,
        "completed_cases": len(rows),
        "summary": summarize(rows),
        "failures": failures,
        "gmsh_version": gmsh.__version__,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"Saved {len(rows)}/{args.limit} matched cases to {output}", flush=True)
    if len(rows) != args.limit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
