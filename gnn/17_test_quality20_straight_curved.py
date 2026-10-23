"""Run the final teacher-free mesh experiment on 12 straight and 4 curved cracks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def curved_polyline(case: dict[str, object]) -> np.ndarray:
    entry = np.asarray(case["entry"], dtype=float)
    length = float(case["chord_length"])
    angle = math.radians(float(case["chord_angle_deg"]))
    amplitude = float(case["curvature_amplitude"])
    segments = int(case["segments"])
    direction = np.asarray((math.cos(angle), math.sin(angle)), dtype=float)
    normal = np.asarray((-direction[1], direction[0]), dtype=float)
    parameter = np.linspace(0.0, 1.0, segments + 1)
    return (
        entry[None, :]
        + parameter[:, None] * length * direction[None, :]
        + amplitude * np.sin(np.pi * parameter)[:, None] * normal[None, :]
    )


def load_cases(straight_manifest: Path, curved_manifest: Path) -> list[dict[str, object]]:
    straight = json.loads(straight_manifest.read_text(encoding="utf-8"))
    curved = json.loads(curved_manifest.read_text(encoding="utf-8"))
    result: list[dict[str, object]] = []
    for index, case in enumerate(straight):
        result.append({"family": "straight", "seed": 74031 + index, **case})
    for case in curved:
        result.append({"family": "slight_curve", **case})
    names = [str(case["case_name"]) for case in result]
    if len(names) != 16 or len(set(names)) != 16:
        raise ValueError("the experiment must contain 12+4 uniquely named cases")
    return result


def report_row(report: dict[str, object], runtime: float) -> dict[str, object]:
    geometry = report["geometry"]
    connectivity = report["connectivity"]
    recovery = report["recovery"]
    quality = recovery["quality"]
    topology = recovery["topology"]
    smoothing = recovery.get("node_smoothing", {})
    return {
        "case_name": Path(str(report["files"]["recovered_mesh"])).stem,
        "family": geometry["family"],
        "edge": geometry["edge"],
        "top_clearance_m": geometry["top_clearance_m"],
        "required_top_clearance_m": geometry["required_top_clearance_m"],
        "path_length": geometry["path_length"],
        "chord_length": geometry["chord_length"],
        "nodes": recovery["nodes"],
        "triangles": recovery["triangles"],
        "candidates": connectivity["candidates"],
        "line_graph_edges": connectivity["line_graph_edges"],
        "decoder_min_angle_deg": connectivity["structured_min_angle_after_deg"],
        "final_min_angle_deg": quality["min_angle_deg"],
        "fraction_below_20deg": quality["fraction_below_20deg"],
        "quality_mean": quality["quality_mean"],
        "relative_area_error": connectivity["relative_area_error"],
        "inserted_nodes": recovery["inserted_nodes"],
        "components": topology["components"],
        "isolated_nodes": topology["isolated_nodes"],
        "nonmanifold_edges": topology["nonmanifold_edges"],
        "missing_constraint_edges": topology["missing_constraint_edges"],
        "duplicate_triangles": topology["duplicate_triangles"],
        "degenerate_triangles": quality["degenerate_triangles"],
        "orientation_preserved": smoothing.get("orientation_preserved", False),
        "smoothing_moves": smoothing.get("accepted_moves", 0),
        "teacher_used": report["teacher_used_at_inference"],
        "cdt_fallback": report["cdt_fallback_used"],
        "triangulation_fallback": report["geometric_triangulation_fallback_used"],
        "runtime_seconds": runtime,
    }


def accepted(row: dict[str, object]) -> bool:
    return (
        not bool(row["teacher_used"])
        and not bool(row["cdt_fallback"])
        and not bool(row["triangulation_fallback"])
        and int(row["inserted_nodes"]) == 0
        and int(row["components"]) == 1
        and int(row["isolated_nodes"]) == 0
        and int(row["nonmanifold_edges"]) == 0
        and int(row["missing_constraint_edges"]) == 0
        and int(row["duplicate_triangles"]) == 0
        and int(row["degenerate_triangles"]) == 0
        and bool(row["orientation_preserved"])
        and float(row["relative_area_error"]) <= 1.0e-10
        and float(row["final_min_angle_deg"]) >= 20.0
    )


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"cases": 0}
    minimum_row = min(rows, key=lambda item: float(item["final_min_angle_deg"]))
    clearance_row = min(rows, key=lambda item: float(item["top_clearance_m"]))
    return {
        "cases": len(rows),
        "all_accepted": all(accepted(row) for row in rows),
        "worst_min_angle_deg": float(minimum_row["final_min_angle_deg"]),
        "worst_min_angle_case": minimum_row["case_name"],
        "mean_case_min_angle_deg": float(np.mean([
            float(row["final_min_angle_deg"]) for row in rows
        ])),
        "mean_decoder_min_angle_deg": float(np.mean([
            float(row["decoder_min_angle_deg"]) for row in rows
        ])),
        "minimum_top_clearance_m": float(clearance_row["top_clearance_m"]),
        "minimum_top_clearance_case": clearance_row["case_name"],
        "maximum_relative_area_error": float(max(
            float(row["relative_area_error"]) for row in rows
        )),
        "mean_runtime_seconds": float(np.mean([
            float(row["runtime_seconds"]) for row in rows
        ])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--straight-manifest", type=Path,
        default=ROOT / "unseen_single_crack_cases_k20.json",
    )
    parser.add_argument(
        "--curved-manifest", type=Path,
        default=ROOT / "quality20_curved_cases.json",
    )
    parser.add_argument(
        "--pipeline", type=Path, default=ROOT / "10_learned_size_to_gnn.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts" / "quality20_12straight_4curve",
    )
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument(
        "--min-top-clearance", type=float, default=0.20,
        help="Minimum crack-to-loaded-top distance in metres",
    )
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--figure", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cases = load_cases(args.straight_manifest.resolve(), args.curved_manifest.resolve())
    output = args.output.resolve()
    geometry_dir = output / "geometries"
    output.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    start_all = time.perf_counter()

    for index, case in enumerate(cases, start=1):
        name = str(case["case_name"])
        family = str(case["family"])
        command = [
            sys.executable, "-B", "-u", str(args.pipeline.resolve()),
            "--output", str(output), "--case-name", name,
            "--candidate-k", str(args.candidate_k),
            "--min-top-clearance", str(args.min_top_clearance),
            "--seed", str(case["seed"]), "--device", args.device,
            "--crack-spacing-factor", "0.9",
            "--quality-target-deg", "20", "--quality-weight", "12",
            "--quality-passes", "12", "--node-smoothing",
            "--smoothing-passes", "10", "--smoothing-move-factor", "0.30",
            "--figure" if args.figure else "--no-figure",
        ]
        if family == "straight":
            command.extend([
                "--edge", str(case["edge"]), "--origin", str(case["origin"]),
                "--length", str(case["length"]),
                "--angle-deg", str(case["angle_deg"]),
            ])
        else:
            polyline = curved_polyline(case)
            if np.any(polyline[:, 0] < 0.0) or np.any(polyline[:, 0] > 1.0):
                raise ValueError(f"{name} leaves the domain in x")
            if np.any(polyline[:, 1] < 0.0) or np.any(polyline[:, 1] > 2.0):
                raise ValueError(f"{name} leaves the domain in y")
            geometry_path = geometry_dir / f"{name}.json"
            geometry_path.write_text(json.dumps({
                "format": "crack-graph-geometry-v1", "W": 1.0, "H": 2.0,
                "crack_polylines": [polyline.tolist()],
            }, indent=2), encoding="utf-8")
            command.extend(["--geometry-json", str(geometry_path)])

        print(f"[{index:02d}/16] START {name} ({family})", flush=True)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command, cwd=ROOT, text=True, encoding="utf-8", errors="replace",
                capture_output=True, timeout=args.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as error:
            failures.append({"case_name": name, "family": family, "error": str(error)})
            print(f"[{index:02d}/16] TIMEOUT {name}", flush=True)
            continue
        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            failures.append({
                "case_name": name, "family": family,
                "error": "pipeline returned non-zero",
                "stdout_tail": completed.stdout[-3000:],
                "stderr_tail": completed.stderr[-3000:],
            })
            print(f"[{index:02d}/16] FAILURE {name}", flush=True)
            continue
        report_path = output / f"{name}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        row = report_row(report, runtime)
        rows.append(row)
        status = "PASS" if accepted(row) else "REJECT"
        if status != "PASS":
            failures.append({"case_name": name, "family": family, "metrics": row})
        print(
            f"[{index:02d}/16] {status} {name}: "
            f"decoder={float(row['decoder_min_angle_deg']):.3f} deg "
            f"final={float(row['final_min_angle_deg']):.3f} deg "
            f"time={runtime:.1f}s",
            flush=True,
        )

    if rows:
        with (output / "per_case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    straight_rows = [row for row in rows if row["family"] == "straight"]
    curved_rows = [row for row in rows if row["family"] == "slight_curve"]
    summary = {
        "format": "quality20-12-straight-4-curved-v1",
        "requested_cases": 16,
        "completed_cases": len(rows),
        "accepted_cases": sum(accepted(row) for row in rows),
        "failed_cases": len(failures),
        "all_accepted": len(rows) == 16 and not failures and all(accepted(row) for row in rows),
        "straight": aggregate(straight_rows),
        "slight_curve": aggregate(curved_rows),
        "overall": aggregate(rows),
        "elapsed_seconds": time.perf_counter() - start_all,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    (output / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved 12+4 experiment to {output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
