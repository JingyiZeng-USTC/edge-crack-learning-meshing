"""Paper-style fixed-mesh FCNN-SM trained from random loads without labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from module10_fem.assembly import assemble_stiffness
from module10_fem.boundary import (
    top_uniform_line_load_boundary,
    vertical_tension_boundary,
)
from module10_fem.material import plane_strain_matrix, plane_stress_matrix
from module10_fem.mesh_io import load_mesh
from module10_fem.recovery import recover_fields
from module10_fem.solver import solve_linear_system
from module43_coordinate_fcnn_sm.physics import scipy_csr_to_torch
from module43_coordinate_fcnn_sm.train_solve import (
    _comparison,
    _save_products,
    _solution_from_displacement,
)

from .model import LoadToDisplacementFCNNSM


MODEL_FAMILY = "fixed-mesh-multiload-fcnn-sm-zhang-yu-2025-v1"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_samples(
    count: int,
    top_x: np.ndarray,
    seed: int,
    family: str = "mixed",
) -> np.ndarray:
    """Generate unlabeled load parameters for one or several load patterns."""
    if count < 3:
        raise ValueError("At least three load samples are required")
    generator = np.random.default_rng(seed)
    if family == "uniform-amplitude":
        samples = generator.uniform(0.10, 1.0, size=(count, 1))
        samples[0, 0] = 1.0
        samples[1, 0] = 0.25
        samples[2, 0] = 0.5
        return samples
    if family != "mixed":
        raise ValueError(f"Unsupported load family: {family}")
    n_top = len(top_x)
    samples = np.empty((count, n_top), dtype=np.float64)
    normalized_x = (top_x - top_x.min()) / max(float(np.ptp(top_x)), 1.0e-30)
    for index in range(count):
        family = index % 3
        if family == 0:
            samples[index] = generator.uniform(-1.0, 1.0, size=n_top)
        elif family == 1:
            samples[index] = generator.uniform(-1.0, 1.0)
        else:
            intercept = generator.uniform(-1.0, 1.0)
            slope = generator.uniform(-1.0, 1.0)
            samples[index] = np.clip(
                intercept + slope * (2.0 * normalized_x - 1.0), -1.0, 1.0,
            )
    # Anchor representative amplitudes for deterministic validation.
    samples[0] = 1.0
    samples[1] = -1.0
    samples[2] = 0.5
    return samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-mesh multi-load FCNN-SM following Zhang and Yu (2025)",
    )
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--E", type=float, default=2.10e11)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument(
        "--behavior", choices=("plane-stress", "plane-strain"), default="plane-stress",
    )
    parser.add_argument(
        "--loading-control",
        choices=("top-line-load", "top-displacement"),
        default="top-displacement",
        help=(
            "top-line-load keeps every top displacement DOF free and applies a "
            "uniform line load in N/m; top-displacement retains the legacy path"
        ),
    )
    parser.add_argument(
        "--top-line-load",
        type=float,
        default=5.0e5,
        help="signed reference vertical line load q_y on the top edge [N/m]",
    )
    parser.add_argument("--top-uy", type=float, default=2.0e-4)
    parser.add_argument("--bottom-uy", type=float, default=0.0)
    parser.add_argument(
        "--load-family",
        choices=("uniform-amplitude", "mixed"),
        default="uniform-amplitude",
        help=(
            "uniform-amplitude keeps one engineering load pattern and varies only "
            "its scalar magnitude; mixed reproduces the broader paper-style loads"
        ),
    )
    parser.add_argument("--samples", type=int, default=2560)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--loss", choices=("residual", "ritz-hybrid"), default="ritz-hybrid",
    )
    parser.add_argument("--ritz-residual-weight", type=float, default=1.0e-2)
    parser.add_argument(
        "--global-balance-weight",
        type=float,
        default=1.0,
        help="penalty on the summed free-node force imbalance in x and y",
    )
    parser.add_argument("--width-multiplier", type=int, default=1)
    parser.add_argument(
        "--low-modes", type=int, default=32,
        help="K-orthogonalized low-frequency stiffness modes used by the spectral head",
    )
    parser.add_argument(
        "--activation", choices=("silu", "tanh", "relu", "gelu"), default="silu",
    )
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=44044)
    parser.add_argument(
        "--warm-start",
        type=Path,
        default=None,
        help=(
            "Optional compatible module44 checkpoint. Model weights are resumed "
            "but optimizer state is intentionally restarted."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]

    mesh_path = Path(args.mesh)
    points, triangles, _ = load_mesh(mesh_path)
    constitutive = (
        plane_stress_matrix(args.E, args.nu)
        if args.behavior == "plane-stress"
        else plane_strain_matrix(args.E, args.nu)
    )
    system = assemble_stiffness(points, triangles, constitutive, args.thickness)
    if args.loading_control == "top-line-load":
        if args.load_family != "uniform-amplitude":
            raise ValueError(
                "Top uniform line-load control requires --load-family uniform-amplitude"
            )
        if args.bottom_uy != 0.0:
            raise ValueError(
                "Force-controlled training currently requires --bottom-uy 0 so the "
                "load-to-displacement map has no unmodelled affine displacement term"
            )
        if not np.isfinite(args.top_line_load) or args.top_line_load == 0.0:
            raise ValueError("--top-line-load must be a finite nonzero reference load")
        boundary = top_uniform_line_load_boundary(
            points,
            line_load_y=args.top_line_load,
            bottom_displacement=args.bottom_uy,
        )
    else:
        boundary = vertical_tension_boundary(
            points,
            top_displacement=args.top_uy,
            bottom_displacement=args.bottom_uy,
        )
    free = boundary.free_dofs(system.n_dofs)
    constrained = np.asarray(boundary.prescribed_dofs, dtype=np.int64)
    top_nodes = np.asarray(boundary.named_nodes["TOP"], dtype=np.int64)
    top_dofs = 2 * top_nodes + 1
    Kff = system.stiffness[free][:, free].tocsr()
    Kfc = system.stiffness[free][:, constrained].tocsr()

    diagonal = np.maximum(np.abs(Kff.diagonal()), 1.0e-30)
    sqrt_diagonal = np.sqrt(diagonal)
    inverse_sqrt = 1.0 / sqrt_diagonal
    median_sqrt = max(float(np.median(sqrt_diagonal)), 1.0e-30)
    scaling = diags(inverse_sqrt)
    stiffness_scaled = (scaling @ Kff @ scaling).tocsr()
    if args.loading_control == "top-line-load":
        # U_f = displacement_scale * median_sqrt * D^(-1/2) z and
        # A z = D^(-1/2) F_f / (displacement_scale * median_sqrt).
        # The physical estimate keeps z near unit scale without using a FEM
        # displacement label or any post-FCNN correction.
        height = max(float(np.ptp(np.asarray(points)[:, 1])), 1.0e-30)
        if args.E <= 0.0 or args.thickness <= 0.0:
            raise ValueError("E and thickness must be positive")
        displacement_scale = max(
            abs(args.top_line_load) * height / (args.E * args.thickness),
            1.0e-12,
        )
        free_force = np.asarray(boundary.force[free], dtype=np.float64)
        rhs_map_all_top = (
            inverse_sqrt[:, None] * free_force[:, None]
        ) / (displacement_scale * median_sqrt)
        # Both physical residual and target are divided by displacement_scale;
        # their ratio, and therefore the balance loss, is unchanged.
        rhs_map_physical = free_force[:, None] / displacement_scale
    else:
        displacement_scale = max(abs(args.top_uy), abs(args.bottom_uy), 1.0e-12)
        constrained_lookup = {
            int(dof): index for index, dof in enumerate(constrained)
        }
        if any(int(dof) not in constrained_lookup for dof in top_dofs):
            raise RuntimeError("Every top vertical DOF must be prescribed")
        active_columns = np.asarray(
            [constrained_lookup[int(dof)] for dof in top_dofs], dtype=np.int64,
        )
        # A z = R p, where p is the normalized prescribed top displacement.
        rhs_map_all_top = -(
            inverse_sqrt[:, None] * Kfc[:, active_columns].toarray()
        ) / median_sqrt
        rhs_map_physical = -Kfc[:, active_columns].toarray()
    rhs_map = rhs_map_all_top
    if args.load_family == "uniform-amplitude":
        rhs_map = np.sum(rhs_map_all_top, axis=1, keepdims=True)
        rhs_map_physical = np.sum(rhs_map_physical, axis=1, keepdims=True)
    stiffness_tensor = scipy_csr_to_torch(stiffness_scaled, device, dtype)
    rhs_map_tensor = torch.as_tensor(rhs_map, dtype=dtype, device=device)
    rhs_map_physical_tensor = torch.as_tensor(
        rhs_map_physical, dtype=dtype, device=device,
    )
    physical_residual_scale = torch.as_tensor(
        sqrt_diagonal * median_sqrt, dtype=dtype, device=device,
    )
    free_tensor = torch.as_tensor(free, dtype=torch.int64, device=device)
    free_x = (free_tensor % 2) == 0
    free_y = ~free_x

    low_mode_basis = None
    low_mode_eigenvalues = np.empty(0, dtype=float)
    if args.low_modes > 0:
        mode_count = min(int(args.low_modes), stiffness_scaled.shape[0] - 2)
        low_mode_eigenvalues, low_mode_vectors = eigsh(
            stiffness_scaled,
            k=mode_count,
            which="SM",
            tol=1.0e-7,
            maxiter=max(20000, 20 * stiffness_scaled.shape[0]),
        )
        order = np.argsort(low_mode_eigenvalues)
        low_mode_eigenvalues = low_mode_eigenvalues[order]
        low_mode_vectors = low_mode_vectors[:, order]
        eigenvalue_floor = max(
            float(low_mode_eigenvalues[-1]) * 1.0e-12, 1.0e-30,
        )
        # Dividing by sqrt(lambda) makes the Ritz Hessian of these modal
        # coefficients approximately the identity.
        low_mode_basis = torch.as_tensor(
            low_mode_vectors
            / np.sqrt(np.maximum(low_mode_eigenvalues, eigenvalue_floor))[None, :],
            dtype=dtype,
            device=device,
        )

    samples = _load_samples(
        args.samples,
        np.asarray(points)[top_nodes, 0],
        args.seed,
        family=args.load_family,
    )
    generator = np.random.default_rng(args.seed + 1)
    permutation = generator.permutation(len(samples))
    n_test = max(1, int(round(args.test_fraction * len(samples))))
    n_validation = max(1, int(round(args.validation_fraction * len(samples))))
    n_train = len(samples) - n_validation - n_test
    if n_train < 1:
        raise ValueError("The requested data fractions leave no training samples")
    train_samples = torch.as_tensor(
        samples[permutation[:n_train]], dtype=dtype, device=device,
    )
    validation_samples = torch.as_tensor(
        samples[permutation[n_train:n_train + n_validation]], dtype=dtype, device=device,
    )
    test_samples = torch.as_tensor(
        samples[permutation[n_train + n_validation:]], dtype=dtype, device=device,
    )
    # Always include the final uniform load in validation and test physics checks.
    uniform = torch.ones((1, samples.shape[1]), dtype=dtype, device=device)
    validation_samples = torch.cat((uniform, validation_samples), dim=0)
    test_samples = torch.cat((uniform, test_samples), dim=0)

    model = LoadToDisplacementFCNNSM(
        samples.shape[1],
        len(free),
        width_multiplier=args.width_multiplier,
        activation=args.activation,
        low_mode_basis=low_mode_basis,
    ).to(device=device, dtype=dtype)
    warm_start_path = None
    if args.warm_start is not None:
        warm_start_path = Path(args.warm_start).resolve()
        if not warm_start_path.is_file():
            raise FileNotFoundError(warm_start_path)
        checkpoint = torch.load(warm_start_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_family") != MODEL_FAMILY:
            raise ValueError(f"Incompatible warm-start model family: {warm_start_path}")
        if checkpoint.get("model_config") != model.config():
            raise ValueError(
                "Warm-start model configuration differs from the requested model: "
                f"checkpoint={checkpoint.get('model_config')}, requested={model.config()}"
            )
        checkpoint_free = torch.as_tensor(checkpoint.get("free_dofs"), dtype=torch.int64)
        if not torch.equal(checkpoint_free.cpu(), torch.as_tensor(free, dtype=torch.int64)):
            raise ValueError("Warm-start checkpoint uses a different free-DOF map")
        checkpoint_mesh = Path(str(checkpoint.get("mesh", ""))).resolve()
        if checkpoint_mesh != mesh_path.resolve():
            raise ValueError(
                f"Warm-start checkpoint mesh differs: {checkpoint_mesh} != {mesh_path.resolve()}"
            )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"Warm-started model weights from {warm_start_path}")
    optimizer = torch.optim.Adamax(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 8, 40),
    )

    def physics_loss(load_batch: torch.Tensor):
        predicted = model(load_batch)
        internal = torch.sparse.mm(stiffness_tensor, predicted.T).T
        target = load_batch @ rhs_map_tensor.T
        residual = internal - target
        denominator = torch.sum(target * target).clamp_min(torch.finfo(dtype).tiny)
        residual_loss = torch.sum(residual * residual) / denominator
        physical_residual = residual * physical_residual_scale[None, :]
        physical_target = load_batch @ rhs_map_physical_tensor.T
        balance_denominator = torch.sum(
            physical_target * physical_target
        ).clamp_min(torch.finfo(dtype).tiny)
        global_balance = torch.stack(
            (
                torch.sum(physical_residual[:, free_x], dim=1),
                torch.sum(physical_residual[:, free_y], dim=1),
            ),
            dim=1,
        )
        balance_loss = torch.sum(global_balance * global_balance) / balance_denominator
        potential = 0.5 * torch.sum(predicted * internal) - torch.sum(predicted * target)
        ritz = potential / denominator
        objective = residual_loss + args.global_balance_weight * balance_loss
        if args.loss == "ritz-hybrid":
            objective = (
                ritz
                + args.ritz_residual_weight * residual_loss
                + args.global_balance_weight * balance_loss
            )
        return objective, ritz, residual_loss, balance_loss, torch.sqrt(residual_loss)

    def validation_value():
        model.eval()
        with torch.no_grad():
            _, _, residual_loss, balance_loss, _ = physics_loss(validation_samples)
            score = torch.sqrt(
                residual_loss + args.global_balance_weight * balance_loss
            )
        return float(score.cpu())

    history = []
    if warm_start_path is None:
        best_validation = float("inf")
        best_epoch = 0
        best_state = None
    else:
        best_validation = validation_value()
        best_epoch = 0
        best_state = deepcopy({
            key: value.detach().cpu() for key, value in model.state_dict().items()
        })
        print(f"Warm-start validation baseline={best_validation:.6e}")
    without_improvement = 0
    started = time.perf_counter()
    print(
        f"Paper-style FCNN-SM | device={device} | inputs={samples.shape[1]} | "
        f"outputs={len(free)} | hidden={model.hidden_widths}"
    )
    print(
        f"Unlabeled load samples: train={len(train_samples)}, "
        f"validation={len(validation_samples)}, test={len(test_samples)}"
    )
    for epoch in range(1, args.epochs + 1):
        indices = torch.randint(
            0, len(train_samples), (args.batch_size,), device=device,
        )
        batch = train_samples[indices]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, ritz, residual_loss, balance_loss, relative = physics_loss(batch)
        loss.backward()
        if args.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        validation = validation_value()
        scheduler.step(validation)
        history.append({
            "epoch": epoch,
            "training_objective": float(loss.detach().cpu()),
            "training_ritz": float(ritz.detach().cpu()),
            "training_scaled_residual": float(relative.detach().cpu()),
            "training_global_balance_loss": float(balance_loss.detach().cpu()),
            "validation_scaled_residual": validation,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        })
        if validation < best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = deepcopy({
                key: value.detach().cpu() for key, value in model.state_dict().items()
            })
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Epoch {epoch:5d} | train={history[-1]['training_scaled_residual']:.3e} | "
                f"validation={validation:.3e} | lr={optimizer.param_groups[0]['lr']:.2e}"
            )
        if args.patience > 0 and without_improvement >= args.patience:
            print(f"Early stop at epoch {epoch}")
            break
    if best_state is None:
        raise RuntimeError("FCNN-SM training did not produce a checkpoint")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        _, _, _, test_balance_tensor, test_relative_tensor = physics_loss(test_samples)
        inference_started = time.perf_counter()
        final_z = model(uniform).reshape(-1)
        inference_seconds = time.perf_counter() - inference_started
    final_z_np = final_z.cpu().numpy().astype(np.float64, copy=False)
    final_U = np.zeros(system.n_dofs, dtype=np.float64)
    final_U[free] = (
        displacement_scale * median_sqrt * inverse_sqrt * final_z_np
    )
    final_U[constrained] = boundary.prescribed_values
    solution = _solution_from_displacement(system, boundary, final_U)
    fields = recover_fields(system, final_U)
    reference_started = time.perf_counter()
    reference_solution = solve_linear_system(system, boundary)
    reference_fields = recover_fields(system, reference_solution.displacement)
    reference_seconds = time.perf_counter() - reference_started
    metrics = _comparison(solution, fields, reference_solution, reference_fields)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_products(Path(str(output) + "_paper_fcnn_raw"), system, boundary, solution, fields)
    _save_products(
        Path(str(output) + "_direct_reference"),
        system,
        boundary,
        reference_solution,
        reference_fields,
    )
    checkpoint = {
        "model_family": MODEL_FAMILY,
        "model_config": model.config(),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "mesh": str(mesh_path.resolve()),
        "free_dofs": torch.as_tensor(free, dtype=torch.int64),
        "prescribed_dofs": torch.as_tensor(constrained, dtype=torch.int64),
        "top_nodes": torch.as_tensor(top_nodes, dtype=torch.int64),
        "inverse_sqrt_diagonal": torch.as_tensor(inverse_sqrt, dtype=torch.float64),
        "median_sqrt_diagonal": median_sqrt,
        "displacement_scale": displacement_scale,
        "loading_control": args.loading_control,
        "reference_top_line_load_y_n_per_m": (
            float(args.top_line_load)
            if args.loading_control == "top-line-load"
            else None
        ),
        "training_args": vars(args),
        "low_mode_eigenvalues": torch.as_tensor(
            low_mode_eigenvalues, dtype=torch.float64,
        ),
    }
    torch.save(checkpoint, output.with_suffix(".pt"))
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "format": "fixed-mesh-multiload-fcnn-sm-report-v1",
        "reference_paper": {
            "title": (
                "FEM-constrained neural network-based surrogate model (FCNN-SM) "
                "for rapid structural response prediction"
            ),
            "doi": "10.1016/j.ijsolstr.2025.113556",
            "adapted_elements": [
                "fixed mesh and fixed output DOFs",
                "known load vector as network input",
                "five dense layers",
                "random unlabeled load samples",
                "nodal force equilibrium residual loss",
                "Adamax optimizer",
            ],
        },
        "training_uses_displacement_or_stress_labels": False,
        "random_load_samples": int(args.samples),
        "load_family": args.load_family,
        "input_load_parameters": int(samples.shape[1]),
        "loading_control": args.loading_control,
        "reference_top_line_load_y_n_per_m": (
            float(args.top_line_load)
            if args.loading_control == "top-line-load"
            else None
        ),
        "loaded_top_length_m": (
            float(np.sum(np.linalg.norm(np.diff(points[top_nodes], axis=0), axis=1)))
            if args.loading_control == "top-line-load"
            else None
        ),
        "reference_top_resultant_y_n": (
            float(np.sum(boundary.force[top_dofs]))
            if args.loading_control == "top-line-load"
            else None
        ),
        "top_loaded_nodes": int(np.count_nonzero(boundary.force[top_dofs])),
        "top_prescribed_dofs": int(np.intersect1d(top_dofs, constrained).size),
        "output_free_dofs": int(len(free)),
        "low_mode_count": int(len(low_mode_eigenvalues)),
        "warm_start_checkpoint": (
            str(warm_start_path) if warm_start_path is not None else None
        ),
        "best_epoch": int(best_epoch),
        "best_validation_scaled_residual": float(best_validation),
        "test_scaled_residual": float(test_relative_tensor.cpu()),
        "test_global_balance_loss": float(test_balance_tensor.cpu()),
        "training_seconds": float(training_seconds),
        "inference_seconds": float(inference_seconds),
        "direct_reference_seconds": float(reference_seconds),
        "raw_paper_fcnn": metrics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print("Paper-style FCNN-SM:", metrics)
    print(
        f"Saved: {output.with_suffix('.pt')}, {output.with_suffix('.csv')}, "
        f"{output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
