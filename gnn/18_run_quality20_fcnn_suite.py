"""Export the final 12+4 GNN meshes and run no-PCG FCNN-SM on every mesh."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean


METRICS = (
    "equilibrium_relative_l2",
    "displacement_relative_l2",
    "element_strain_relative_l2",
    "element_stress_relative_l2",
    "element_von_mises_relative_l2",
    "strain_energy_relative_error",
)


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def compatible_existing(path: Path, line_load: float, samples: int) -> bool:
    if not path.is_file():
        return False
    try:
        report = load_json(path)
        return bool(
            report.get("loading_control") == "top-line-load"
            and report.get("training_uses_displacement_or_stress_labels") is False
            and int(report.get("random_load_samples", -1)) == samples
            and abs(float(report.get("reference_top_line_load_y_n_per_m", 0.0)) - line_load)
            <= max(1.0, abs(line_load)) * 1.0e-12
            and isinstance(report.get("raw_paper_fcnn"), dict)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def export_fcnn_meshes(
    python: Path,
    adapter: Path,
    suite_root: Path,
    write_vtu: bool,
    case_names: list[str],
) -> Path:
    if not case_names or len(set(case_names)) != len(case_names):
        raise ValueError("FCNN export requires a non-empty list of unique case names")
    output = suite_root / "fcnn_meshes"
    command = [
        str(python), "-B", "-u", str(adapter),
        "--recovery", str(suite_root / "meshes"),
        "--candidates", str(suite_root / "candidates"),
        "--output", str(output),
        "--vtu" if write_vtu else "--no-vtu",
    ]
    for name in case_names:
        command.extend(("--name", name))
    completed = subprocess.run(command, cwd=adapter.parent, check=False)
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)
    summary = load_json(output / "summary.json")
    expected = len(case_names)
    if int(summary.get("meshes", -1)) != expected:
        raise RuntimeError(
            f"Expected {expected} exported FCNN meshes, got {summary.get('meshes')}"
        )
    if int(summary.get("degenerate_elements", -1)) != 0:
        raise RuntimeError("Exported FCNN meshes contain degenerate elements")
    return output


def run_case(
    python: Path,
    project_root: Path,
    mesh: Path,
    output: Path,
    log: Path,
    *,
    line_load: float,
    epochs: int,
    samples: int,
    seed: int,
    device: str,
) -> None:
    command = [
        str(python), "-B", "-u", "-m", "module44_load_surrogate_fcnn_sm.train_solve",
        "--mesh", str(mesh), "--output", str(output),
        "--E", "2.10e11", "--nu", "0.30", "--thickness", "0.01",
        "--behavior", "plane-stress",
        "--loading-control", "top-line-load",
        "--top-line-load", f"{line_load:.17g}", "--bottom-uy", "0",
        "--load-family", "uniform-amplitude",
        "--samples", str(samples), "--batch-size", "64",
        "--epochs", str(epochs), "--learning-rate", "1e-3",
        "--loss", "ritz-hybrid", "--ritz-residual-weight", "1e-2",
        "--global-balance-weight", "1.0", "--width-multiplier", "1",
        "--low-modes", "16", "--activation", "silu", "--patience", "800",
        "--dtype", "float64", "--device", device, "--seed", str(seed),
    ]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=project_root, stdout=stream, stderr=subprocess.STDOUT,
            text=True, check=False,
        )
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def result_row(
    case: dict[str, str],
    fcnn: dict[str, object],
    mesh_report: dict[str, object],
) -> dict[str, object]:
    raw = fcnn.get("raw_paper_fcnn")
    if not isinstance(raw, dict):
        raise ValueError("raw_paper_fcnn is missing")
    geometry = mesh_report["geometry"]
    recovery = mesh_report["recovery"]
    row: dict[str, object] = {
        "case_name": case["case_name"],
        "family": case["family"],
        "edge": geometry["edge"],
        "mesh_nodes": recovery["nodes"],
        "mesh_triangles": recovery["triangles"],
        "mesh_min_angle_deg": recovery["quality"]["min_angle_deg"],
        "top_line_load_n_per_m": fcnn["reference_top_line_load_y_n_per_m"],
        "top_resultant_n": fcnn["reference_top_resultant_y_n"],
        "best_epoch": fcnn["best_epoch"],
        "low_mode_count": fcnn["low_mode_count"],
        "training_seconds": fcnn["training_seconds"],
        "inference_seconds": fcnn["inference_seconds"],
        "direct_reference_seconds": fcnn["direct_reference_seconds"],
        "speedup_vs_direct": float(fcnn["direct_reference_seconds"])
        / max(float(fcnn["inference_seconds"]), 1.0e-30),
    }
    for metric in METRICS:
        row[metric] = float(raw[metric])
    return row


def aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"cases": 0}
    averages = {
        metric: mean(float(row[metric]) for row in rows)
        for metric in (*METRICS, "training_seconds", "inference_seconds", "speedup_vs_direct")
    }
    maxima: dict[str, object] = {}
    for metric in METRICS:
        worst = max(rows, key=lambda row: float(row[metric]))
        maxima[metric] = {
            "value": float(worst[metric]), "case_name": str(worst["case_name"]),
        }
    return {
        "cases": len(rows), "mean": averages, "maximum": maxima,
        "all_displacement_errors_below_1_percent": all(
            float(row["displacement_relative_l2"]) < 0.01 for row in rows
        ),
        "all_stress_errors_below_1_percent": all(
            float(row["element_stress_relative_l2"]) < 0.01 for row in rows
        ),
        "all_von_mises_errors_below_1_percent": all(
            float(row["element_von_mises_relative_l2"]) < 0.01 for row in rows
        ),
        "all_primary_field_errors_below_2_percent": all(
            max(
                float(row["displacement_relative_l2"]),
                float(row["element_stress_relative_l2"]),
                float(row["element_von_mises_relative_l2"]),
            ) < 0.02 for row in rows
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    project_root = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-root", type=Path,
        default=root / "artifacts" / "quality20_12straight_4curve",
    )
    parser.add_argument("--adapter", type=Path, default=root / "09_export_gnn_mesh_for_fcnn.py")
    parser.add_argument("--python", type=Path, default=Path(r"D:\itl\eg\pinn_gpu\python.exe"))
    parser.add_argument("--top-line-load", type=float, default=5.0e5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=2560)
    parser.add_argument("--seed", type=int, default=78001)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vtu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    if args.epochs < 1 or args.samples < 3:
        raise ValueError("epochs must be positive and samples must be at least 3")
    if not args.python.is_file() or not args.adapter.is_file():
        raise FileNotFoundError(f"Missing Python or adapter: {args.python}, {args.adapter}")
    metric_path = suite_root / "per_case_metrics.csv"
    with metric_path.open("r", encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 16:
        raise ValueError(f"Expected 16 mesh cases, got {len(cases)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        cases = cases[: args.limit]

    print("[export] Opening crack faces and writing FCNN mesh files...", flush=True)
    mesh_dir = export_fcnn_meshes(
        args.python.resolve(), args.adapter.resolve(), suite_root, args.vtu,
        [case["case_name"] for case in cases],
    )
    if args.export_only:
        print(f"Export complete: {mesh_dir}", flush=True)
        return

    result_dir = suite_root / "fcnn_line_load_results"
    log_dir = result_dir / "logs"
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    for index, case in enumerate(cases, start=1):
        name = case["case_name"]
        report_path = result_dir / f"{name}.json"
        try:
            if args.skip_existing and compatible_existing(
                report_path, args.top_line_load, args.samples,
            ):
                print(f"[{index:02d}/{len(cases):02d}] REUSE {name}", flush=True)
            else:
                print(f"[{index:02d}/{len(cases):02d}] TRAIN {name}", flush=True)
                run_case(
                    args.python.resolve(), project_root, mesh_dir / f"{name}.npz",
                    result_dir / name, log_dir / f"{name}.log",
                    line_load=args.top_line_load, epochs=args.epochs,
                    samples=args.samples, seed=args.seed + index - 1, device=args.device,
                )
            fcnn = load_json(report_path)
            mesh_report = load_json(suite_root / f"{name}.json")
            row = result_row(case, fcnn, mesh_report)
            rows.append(row)
            print(
                f"[{index:02d}/{len(cases):02d}] PASS {name}: "
                f"u={100.0 * float(row['displacement_relative_l2']):.3f}% "
                f"stress={100.0 * float(row['element_stress_relative_l2']):.3f}% "
                f"vm={100.0 * float(row['element_von_mises_relative_l2']):.3f}%",
                flush=True,
            )
        except Exception as error:
            failures.append({"case_name": name, "family": case["family"], "error": f"{type(error).__name__}: {error}"})
            print(f"[{index:02d}/{len(cases):02d}] FAIL {name}: {error}", flush=True)
            if not args.continue_on_error:
                break

    if rows:
        with (result_dir / "fcnn_suite_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    straight = [row for row in rows if row["family"] == "straight"]
    curved = [row for row in rows if row["family"] == "slight_curve"]
    summary = {
        "format": "quality20-12-straight-4-curved-no-pcg-fcnn-sm-v1",
        "loading_control": "top-line-load",
        "top_line_load_n_per_m": args.top_line_load,
        "training_uses_displacement_or_stress_labels": False,
        "pcg_used_after_fcnn": False,
        "epochs": args.epochs, "unlabeled_load_samples": args.samples,
        "requested_cases": len(cases), "successful_cases": len(rows),
        "failed_cases": len(failures), "straight": aggregate(straight),
        "slight_curve": aggregate(curved), "overall": aggregate(rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (result_dir / "fcnn_suite_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    (result_dir / "fcnn_suite_failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved FCNN suite: {result_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
