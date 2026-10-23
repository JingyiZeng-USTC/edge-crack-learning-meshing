"""Batch bridge from generalized pure crack meshes to Module 10 physics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .assembly import assemble_stiffness
from .boundary import vertical_tension_boundary
from .material import plane_strain_matrix, plane_stress_matrix
from .mesh_io import load_mesh, save_vtu
from .recovery import recover_fields
from .solver import solve_linear_system
from .stage_io import save_fields, save_solution, save_system


def _scalar_bool(metadata: dict, name: str, default: bool) -> bool:
    if name not in metadata:
        return default
    return bool(np.asarray(metadata[name]).reshape(()).item())


def _constitutive(args):
    if args.behavior == "plane-stress":
        return plane_stress_matrix(args.E, args.nu)
    return plane_strain_matrix(args.E, args.nu)


def _write_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "case", "status", "reason", "nodes", "triangles", "dofs", "nnz",
        "system_id", "relative_residual", "strain_energy",
        "max_displacement", "max_von_mises", "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def process_mesh(mesh_path: Path, output_dir: Path, args) -> dict:
    started = perf_counter()
    case = mesh_path.stem
    record = {
        "case": case,
        "status": "failed",
        "reason": "",
        "nodes": "",
        "triangles": "",
        "dofs": "",
        "nnz": "",
        "system_id": "",
        "relative_residual": "",
        "strain_energy": "",
        "max_displacement": "",
        "max_von_mises": "",
        "elapsed_seconds": "",
    }
    try:
        points, triangles, metadata = load_mesh(mesh_path)
        record["nodes"] = len(points)
        record["triangles"] = len(triangles)
        if not _scalar_bool(metadata, "crack_faces_open", True):
            record["status"] = "skipped"
            record["reason"] = "crack_faces_not_open"
            return record

        system = assemble_stiffness(points, triangles, _constitutive(args), args.thickness)
        boundary = vertical_tension_boundary(
            system.points,
            top_displacement=args.top_uy,
            bottom_displacement=args.bottom_uy,
            pin_node=args.pin_node,
        )
        solution = solve_linear_system(system, boundary)
        fields = recover_fields(system, solution.displacement)

        prefix = output_dir / case
        identifier = save_system(
            str(prefix) + "_system.npz",
            system,
            source_mesh=str(mesh_path.resolve()),
        )
        save_solution(str(prefix) + "_solution.npz", system, boundary, solution)
        save_fields(str(prefix) + "_fields.npz", system, fields)
        if args.write_vtu:
            save_vtu(str(prefix) + ".vtu", system, solution, fields)

        rhs_norm = max(float(np.linalg.norm(solution.rhs_free)), 1.0e-30)
        record.update({
            "status": "ok",
            "dofs": system.n_dofs,
            "nnz": system.stiffness.nnz,
            "system_id": identifier,
            "relative_residual": float(np.linalg.norm(solution.residual_free)) / rhs_norm,
            "strain_energy": solution.strain_energy,
            "max_displacement": float(
                np.linalg.norm(solution.displacement_xy, axis=1).max()
            ),
            "max_von_mises": float(fields.element_von_mises.max()),
        })
    except Exception as error:  # keep the remaining independent cases running
        record["reason"] = f"{type(error).__name__}: {error}"
    finally:
        record["elapsed_seconds"] = perf_counter() - started
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch pure-mesh -> K -> U -> strain/stress bridge",
    )
    parser.add_argument("--mesh-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*.npz")
    parser.add_argument("--E", type=float, default=2.10e11)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument(
        "--behavior", choices=("plane-stress", "plane-strain"),
        default="plane-stress",
    )
    parser.add_argument("--top-uy", type=float, default=2.0e-4)
    parser.add_argument("--bottom-uy", type=float, default=0.0)
    parser.add_argument("--pin-node", type=int, default=None)
    parser.add_argument("--write-vtu", action="store_true")
    parser.add_argument(
        "--strict", action="store_true",
        help="Return an error code if any non-skipped case fails",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mesh_dir = Path(args.mesh_dir)
    output_dir = Path(args.output_dir)
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory does not exist: {mesh_dir}")
    mesh_paths = sorted(mesh_dir.glob(args.pattern))
    if not mesh_paths:
        raise FileNotFoundError(f"No meshes match {mesh_dir / args.pattern}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, mesh_path in enumerate(mesh_paths, start=1):
        record = process_mesh(mesh_path, output_dir, args)
        records.append(record)
        print(
            f"[{index:02d}/{len(mesh_paths):02d}] {record['case']}: "
            f"{record['status']} {record['reason']}",
        )

    counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("ok", "skipped", "failed")
    }
    valid = [record for record in records if record["status"] == "ok"]
    summary = {
        "dataset_family": "deterministic-module10-staged-physics-v1",
        "mesh_directory": str(mesh_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "material": {
            "E": args.E,
            "nu": args.nu,
            "thickness": args.thickness,
            "behavior": args.behavior,
        },
        "loading": {
            "top_uy": args.top_uy,
            "bottom_uy": args.bottom_uy,
            "pin_node": args.pin_node,
        },
        "counts": counts,
        "maximum_relative_residual": max(
            (record["relative_residual"] for record in valid),
            default=None,
        ),
        "maximum_von_mises": max(
            (record["max_von_mises"] for record in valid),
            default=None,
        ),
        "records": records,
    }
    csv_path = output_dir / "physics_cases.csv"
    json_path = output_dir / "physics_summary.json"
    _write_csv(csv_path, records)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: ok={counts['ok']}, skipped={counts['skipped']}, failed={counts['failed']}")
    print(f"Saved: {csv_path}, {json_path}")
    if args.strict and counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
