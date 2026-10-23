"""Train one coordinate FCNN-SM from K and b, then recover physical fields."""

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
from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import cg

from module10_fem.assembly import assemble_stiffness
from module10_fem.boundary import vertical_tension_boundary
from module10_fem.material import plane_strain_matrix, plane_stress_matrix
from module10_fem.mesh_io import load_mesh, save_npz, save_vtu
from module10_fem.recovery import recover_fields
from module10_fem.solver import Solution, solve_linear_system

from .features import FEATURE_NAMES, build_node_features
from .model import CoordinateFCNNSM
from .physics import ScaledRitzEquilibrium


MODEL_FAMILY = "coordinate-fcnn-sm-neural-galerkin-v3"


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _relative(values: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(np.asarray(values) - np.asarray(reference)))
    denominator = max(float(np.linalg.norm(reference)), 1.0e-30)
    return numerator / denominator


def _solution_from_displacement(system, boundary, displacement: np.ndarray) -> Solution:
    U = np.asarray(displacement, dtype=float).reshape(-1)
    free = boundary.free_dofs(system.n_dofs)
    constrained = np.asarray(boundary.prescribed_dofs, dtype=np.int64)
    rhs = boundary.force[free] - system.stiffness[free][:, constrained] @ boundary.prescribed_values
    internal = np.asarray(system.stiffness @ U).reshape(-1)
    residual = internal - boundary.force
    return Solution(
        displacement=U,
        displacement_xy=U.reshape(-1, 2),
        reactions=residual[constrained],
        free_dofs=free,
        prescribed_dofs=constrained,
        rhs_free=np.asarray(rhs).reshape(-1),
        residual_free=residual[free],
        strain_energy=0.5 * float(U @ internal),
    )


def _model_displacement(
    model,
    feature_tensor,
    *,
    equilibrium: ScaledRitzEquilibrium,
    free_dofs: np.ndarray,
    prescribed_dofs: np.ndarray,
    prescribed_values: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        dimensionless = model(feature_tensor).reshape(-1)
        full = torch.zeros_like(dimensionless)
        free_tensor = torch.as_tensor(free_dofs, dtype=torch.int64, device=dimensionless.device)
        constrained_tensor = torch.as_tensor(
            prescribed_dofs, dtype=torch.int64, device=dimensionless.device,
        )
        prescribed_tensor = torch.as_tensor(
            prescribed_values, dtype=dimensionless.dtype, device=dimensionless.device,
        )
        full[free_tensor] = equilibrium.free_displacement(dimensionless[free_tensor])
        full[constrained_tensor] = prescribed_tensor
    return full.cpu().numpy().astype(np.float64, copy=False)


def _save_products(prefix: Path, system, boundary, solution, fields) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_npz(prefix.with_suffix(".npz"), system, boundary, solution, fields)
    save_vtu(prefix.with_suffix(".vtu"), system, solution, fields)


def _comparison(solution, fields, reference_solution, reference_fields) -> dict:
    result = _self_metrics(solution, fields)
    result.update({
        "displacement_relative_l2": _relative(
            solution.displacement, reference_solution.displacement,
        ),
        "element_strain_relative_l2": _relative(
            fields.element_strain, reference_fields.element_strain,
        ),
        "element_stress_relative_l2": _relative(
            fields.element_stress, reference_fields.element_stress,
        ),
        "element_von_mises_relative_l2": _relative(
            fields.element_von_mises, reference_fields.element_von_mises,
        ),
        "strain_energy_relative_error": abs(
            solution.strain_energy - reference_solution.strain_energy
        ) / max(abs(reference_solution.strain_energy), 1.0e-30),
    })
    return result


def _self_metrics(solution, fields) -> dict:
    rhs_norm = max(float(np.linalg.norm(solution.rhs_free)), 1.0e-30)
    return {
        "equilibrium_relative_l2": float(np.linalg.norm(solution.residual_free)) / rhs_norm,
        "strain_energy": float(solution.strain_energy),
        "maximum_displacement": float(np.linalg.norm(solution.displacement_xy, axis=1).max()),
        "maximum_element_von_mises": float(fields.element_von_mises.max()),
    }


def _galerkin_initialize_output(
    model: CoordinateFCNNSM,
    feature_tensor: torch.Tensor,
    free_dofs: np.ndarray,
    stiffness_ff,
    equilibrium: ScaledRitzEquilibrium,
    *,
    rcond: float,
) -> dict:
    """Solve the reduced Galerkin system for the network's final linear layer.

    Hidden activations are treated as basis functions.  Only Kff, the corrected
    right-hand side and the boundary lifting enter this solve; no reference
    displacement, strain or stress is used.
    """
    if not (0.0 < rcond < 1.0):
        raise ValueError("galerkin_rcond must lie between zero and one")
    model.eval()
    with torch.no_grad():
        hidden = model.basis_features(feature_tensor).detach().cpu().numpy()
    hidden = np.asarray(hidden, dtype=np.float64)
    augmented = np.column_stack((hidden, np.ones(len(hidden), dtype=np.float64)))
    basis_per_component = augmented.shape[1]
    free_array = np.asarray(free_dofs, dtype=np.int64)
    nodes = free_array // 2
    components = free_array % 2
    basis = np.zeros(
        (len(free_array), 2 * basis_per_component), dtype=np.float64,
    )
    for component in (0, 1):
        rows = np.flatnonzero(components == component)
        column_start = component * basis_per_component
        basis[rows, column_start:column_start + basis_per_component] = augmented[nodes[rows]]

    diagonal = np.maximum(np.abs(stiffness_ff.diagonal()), 1.0e-30)
    inverse_sqrt = 1.0 / np.sqrt(diagonal)
    scaling = diags(inverse_sqrt)
    scaled_matrix = (scaling @ stiffness_ff @ scaling).tocsr()
    variable_scale = float(equilibrium.scaled_variable_scale.detach().cpu())
    scaled_basis = variable_scale * basis
    rhs_scaled = equilibrium.rhs_scaled.detach().cpu().numpy().astype(np.float64, copy=False)
    matrix_times_basis = np.asarray(scaled_matrix @ scaled_basis)
    reduced_matrix = scaled_basis.T @ matrix_times_basis
    reduced_matrix = 0.5 * (reduced_matrix + reduced_matrix.T)
    reduced_rhs = scaled_basis.T @ rhs_scaled

    eigenvalues, eigenvectors = eigh(
        reduced_matrix,
        overwrite_a=True,
        check_finite=False,
        driver="evr",
    )
    maximum = max(float(eigenvalues[-1]), 1.0e-30)
    threshold = float(rcond) * maximum
    active = eigenvalues > threshold
    if not np.any(active):
        raise RuntimeError("The neural Galerkin basis has no numerically independent mode")
    active_vectors = eigenvectors[:, active]
    coefficients = active_vectors @ (
        (active_vectors.T @ reduced_rhs) / eigenvalues[active]
    )
    coefficient_blocks = coefficients.reshape(2, basis_per_component)
    with torch.no_grad():
        model.output_layer.weight.copy_(torch.as_tensor(
            coefficient_blocks[:, :-1],
            dtype=model.output_layer.weight.dtype,
            device=model.output_layer.weight.device,
        ))
        model.output_layer.bias.copy_(torch.as_tensor(
            coefficient_blocks[:, -1],
            dtype=model.output_layer.bias.dtype,
            device=model.output_layer.bias.device,
        ))

    reduced_residual = reduced_matrix @ coefficients - reduced_rhs
    return {
        "basis_functions_per_component": int(basis_per_component),
        "reduced_dofs": int(2 * basis_per_component),
        "retained_modes": int(np.count_nonzero(active)),
        "discarded_modes": int(len(active) - np.count_nonzero(active)),
        "eigenvalue_threshold": threshold,
        "minimum_retained_eigenvalue": float(eigenvalues[active][0]),
        "maximum_eigenvalue": maximum,
        "reduced_condition_number": float(maximum / eigenvalues[active][0]),
        "reduced_equation_relative_l2": float(np.linalg.norm(reduced_residual))
        / max(float(np.linalg.norm(reduced_rhs)), 1.0e-30),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-mesh zero-label coordinate FCNN-SM displacement solver",
    )
    parser.add_argument("--mesh", required=True, help="Pure crack mesh NPZ")
    parser.add_argument("--output", default="output/coordinate_fcnn_sm/result")
    parser.add_argument("--warm-start", default=None, help="Previous mesh FCNN-SM checkpoint")
    parser.add_argument("--E", type=float, default=2.10e11)
    parser.add_argument("--nu", type=float, default=0.30)
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument(
        "--behavior", choices=("plane-stress", "plane-strain"), default="plane-stress",
    )
    parser.add_argument("--top-uy", type=float, default=2.0e-4)
    parser.add_argument("--bottom-uy", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--activation", choices=("silu", "tanh", "gelu", "relu"), default="silu")
    parser.add_argument(
        "--readout-features",
        choices=("final", "all", "all_with_input"),
        default="all_with_input",
        help="Use the final hidden state or concatenate all hidden levels as neural bases",
    )
    parser.add_argument(
        "--galerkin-initialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the linear readout by a reduced Galerkin solve",
    )
    parser.add_argument(
        "--galerkin-refresh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute the Galerkin readout after each hidden-basis training stage",
    )
    parser.add_argument("--galerkin-rcond", type=float, default=1.0e-9)
    parser.add_argument(
        "--fine-tune-scope", choices=("readout", "all"), default="readout",
        help="Fine-tune only Galerkin coefficients or the complete neural basis",
    )
    parser.add_argument(
        "--basis-epochs",
        type=int,
        default=0,
        help="Alternating hidden-basis updates followed by exact reduced Galerkin solves",
    )
    parser.add_argument("--basis-lr", type=float, default=1.0e-5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--residual-epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-8)
    parser.add_argument("--patience", type=int, default=500)
    parser.add_argument("--target-residual", type=float, default=1.0e-6)
    parser.add_argument(
        "--target-scaled-residual",
        type=float,
        default=None,
        help="Stopping tolerance for the Jacobi-scaled residual; defaults to --target-residual",
    )
    parser.add_argument(
        "--ritz-residual-weight", type=float, default=1.0e-2,
        help="Small scaled-residual term used during the Ritz stage",
    )
    parser.add_argument("--local-feature-h-min", type=float, default=0.004)
    parser.add_argument("--lbfgs-steps", type=int, default=80)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=43043)
    parser.add_argument("--correction", choices=("none", "pcg"), default="pcg")
    parser.add_argument("--correction-rtol", type=float, default=1.0e-10)
    parser.add_argument("--correction-maxiter", type=int, default=2000)
    parser.add_argument(
        "--residual-enrichment-steps",
        type=int,
        default=0,
        help=(
            "Add a fixed number of preconditioned Krylov residual directions to "
            "the neural Galerkin solution"
        ),
    )
    parser.add_argument("--residual-enrichment-rtol", type=float, default=1.0e-12)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Do not compute the sparse direct solution after neural training",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.epochs < 1
        or args.residual_epochs < 0
        or args.lbfgs_steps < 0
        or args.basis_epochs < 0
        or args.residual_enrichment_steps < 0
    ):
        raise ValueError("training epoch counts must be nonnegative and Ritz epochs positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    dtype = _dtype(args.dtype)
    target_scaled_residual = (
        float(args.target_residual)
        if args.target_scaled_residual is None
        else float(args.target_scaled_residual)
    )

    mesh_path = Path(args.mesh)
    points, triangles, metadata = load_mesh(mesh_path)
    if mesh_path.suffix.lower() != ".npz":
        raise ValueError("Use the pure-mesh NPZ so crack-face identity is available")
    if args.behavior == "plane-stress":
        constitutive = plane_stress_matrix(args.E, args.nu)
    else:
        constitutive = plane_strain_matrix(args.E, args.nu)
    system = assemble_stiffness(points, triangles, constitutive, args.thickness)
    boundary = vertical_tension_boundary(
        points,
        top_displacement=args.top_uy,
        bottom_displacement=args.bottom_uy,
    )
    free = boundary.free_dofs(system.n_dofs)
    constrained = np.asarray(boundary.prescribed_dofs, dtype=np.int64)
    Kff = system.stiffness[free][:, free].tocsr()
    rhs = np.asarray(
        boundary.force[free] - system.stiffness[free][:, constrained] @ boundary.prescribed_values,
        dtype=float,
    ).reshape(-1)
    displacement_scale = max(
        abs(float(args.top_uy)), abs(float(args.bottom_uy)),
        abs(float(args.top_uy) - float(args.bottom_uy)), 1.0e-12,
    )
    features, geometry = build_node_features(
        points,
        metadata,
        boundary,
        young_modulus=args.E,
        poisson_ratio=args.nu,
        thickness=args.thickness,
        displacement_scale=displacement_scale,
        local_feature_h_min=args.local_feature_h_min,
    )
    feature_tensor = torch.as_tensor(features, dtype=dtype, device=device)
    free_tensor = torch.as_tensor(free, dtype=torch.int64, device=device)
    # An affine lifting satisfies the prescribed vertical displacement exactly.
    # The network therefore learns only the crack-induced correction.  This is
    # constructed from geometry and boundary values, not from an FEM solution.
    point_coordinates = np.asarray(points, dtype=float)[:, :2]
    x = point_coordinates[:, 0]
    y = point_coordinates[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    height = max(y_max - y_min, 1.0e-30)
    axial_strain = (float(args.top_uy) - float(args.bottom_uy)) / height
    if args.behavior == "plane-stress":
        lateral_ratio = float(args.nu)
    else:
        lateral_ratio = float(args.nu) / max(1.0 - float(args.nu), 1.0e-30)
    pin_node = int(np.asarray(boundary.named_nodes["PIN"], dtype=np.int64)[0])
    lifting_U = np.zeros(system.n_dofs, dtype=float)
    lifting_U[0::2] = -lateral_ratio * axial_strain * (x - x[pin_node])
    lifting_U[1::2] = (
        float(args.bottom_uy)
        + (float(args.top_uy) - float(args.bottom_uy)) * (y - y_min) / height
    )
    lifting_U[constrained] = boundary.prescribed_values
    equilibrium = ScaledRitzEquilibrium.from_scipy(
        Kff,
        rhs,
        displacement_scale=displacement_scale,
        base_free_displacement=lifting_U[free],
        device=device,
        dtype=dtype,
    )

    warm_checkpoint = None
    if args.warm_start:
        warm_checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        if warm_checkpoint.get("model_family") != MODEL_FAMILY:
            raise ValueError("Warm-start file is not a coordinate FCNN-SM checkpoint")
        if list(warm_checkpoint.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("Warm-start feature schema is incompatible")
        model_config = dict(warm_checkpoint["model_config"])
        print(f"Warm start: {args.warm_start}")
    else:
        model_config = {
            "input_dim": len(FEATURE_NAMES),
            "hidden_dim": args.hidden,
            "residual_blocks": args.blocks,
            "activation": args.activation,
            "readout_features": args.readout_features,
        }
    model = CoordinateFCNNSM(**model_config).to(device=device, dtype=dtype)
    if warm_checkpoint is not None:
        model.load_state_dict(warm_checkpoint["model_state_dict"], strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def evaluate_network():
        dimensionless_free = model(feature_tensor).reshape(-1)[free_tensor]
        return equilibrium.losses_and_metrics(dimensionless_free)

    started = time.perf_counter()

    def evaluated_values() -> tuple[float, float, float, float]:
        model.eval()
        with torch.no_grad():
            ritz, residual_loss, relative, scaled_relative, _ = evaluate_network()
        return (
            float(ritz.cpu()),
            float(residual_loss.cpu()),
            float(relative.cpu()),
            float(scaled_relative.cpu()),
        )

    def history_row(stage: str, iteration: int, values, learning_rate: float) -> dict:
        ritz_value, residual_loss_value, relative_value, scaled_relative_value = values
        loss_value = (
            residual_loss_value
            if stage.startswith("residual")
            else ritz_value + args.ritz_residual_weight * residual_loss_value
        )
        return {
            "stage": stage,
            "iteration": iteration,
            "loss": loss_value,
            "ritz_loss": ritz_value,
            "residual_loss": residual_loss_value,
            "equilibrium_relative_l2": relative_value,
            "scaled_equilibrium_relative_l2": scaled_relative_value,
            "learning_rate": learning_rate,
        }

    lifting_values = evaluated_values()
    history: list[dict] = [history_row("lifting_initial", 0, lifting_values, args.lr)]
    galerkin_events: list[dict] = []

    def apply_galerkin(stage: str, iteration: int):
        previous_state = deepcopy(
            {k: v.detach().cpu() for k, v in model.state_dict().items()}
        )
        before = evaluated_values()
        event = _galerkin_initialize_output(
            model,
            feature_tensor,
            free,
            Kff,
            equilibrium,
            rcond=args.galerkin_rcond,
        )
        after = evaluated_values()
        candidate = after
        # Galerkin projection minimizes the discrete potential over the current
        # neural subspace.  The algebraic residual need not decrease monotonically
        # for a reduced trial space, so acceptance is based on Ritz energy.
        accepted = after[0] <= before[0]
        if not accepted:
            model.load_state_dict(previous_state)
            after = before
        event.update({
            "stage": stage,
            "accepted": bool(accepted),
            "scaled_residual_before": before[3],
            "scaled_residual_after": after[3],
            "scaled_residual_candidate": candidate[3],
            "physical_residual_before": before[2],
            "physical_residual_after": after[2],
            "physical_residual_candidate": candidate[2],
            "ritz_before": before[0],
            "ritz_after": after[0],
            "ritz_candidate": candidate[0],
        })
        galerkin_events.append(event)
        history.append(history_row(stage, iteration, after, 0.0))
        print(
            f"Galerkin {stage}: modes={event['retained_modes']}/"
            f"{event['reduced_dofs']} | physical={after[2]:.6e} | "
            f"scaled={after[3]:.6e} | accepted={accepted}"
        )
        return after

    if args.galerkin_initialization:
        initial_values = apply_galerkin("galerkin_initialization", 0)
    else:
        initial_values = lifting_values
    initial_ritz, initial_residual_loss, initial_relative, initial_scaled_relative = (
        initial_values
    )
    best_relative = initial_relative
    best_scaled_relative = initial_scaled_relative
    best_ritz = initial_ritz
    best_epoch = len(history) - 1
    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    without_improvement = 0

    print(f"Device: {device} | dtype: {dtype}")
    print(
        f"Mesh: nodes={system.n_nodes}, triangles={len(system.triangles)}, "
        f"DOFs={system.n_dofs}, free={len(free)}"
    )
    print(
        f"Network: {len(FEATURE_NAMES)} -> {model.hidden_dim} x "
        f"{model.residual_blocks} residual blocks -> {model.readout_dim} neural bases -> 2"
    )
    print(
        "Supervision: none; neural-basis Galerkin initialization, followed by "
        "scaled Ritz energy and equilibrium-residual fine-tuning"
    )
    print(
        f"Initial residual: physical={initial_relative:.6e}, "
        f"scaled={initial_scaled_relative:.6e}"
    )

    basis_adaptation = None
    if args.basis_epochs > 0:
        hidden_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if not name.startswith("output_layer.")
        ]
        basis_optimizer = torch.optim.AdamW(
            hidden_parameters,
            lr=args.basis_lr,
            weight_decay=args.weight_decay,
        )
        basis_best_ritz = initial_ritz
        basis_best_values = initial_values
        basis_best_state = deepcopy(
            {key: value.detach().cpu() for key, value in model.state_dict().items()}
        )
        last_information = None
        for basis_epoch in range(1, args.basis_epochs + 1):
            model.train()
            basis_optimizer.zero_grad(set_to_none=True)
            ritz, _, _, _, _ = evaluate_network()
            ritz.backward()
            if args.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    hidden_parameters, args.gradient_clip,
                )
            basis_optimizer.step()

            # With the updated hidden functions fixed, recompute the energy-
            # optimal output coefficients exactly in the reduced subspace.
            last_information = _galerkin_initialize_output(
                model,
                feature_tensor,
                free,
                Kff,
                equilibrium,
                rcond=args.galerkin_rcond,
            )
            adapted_values = evaluated_values()
            history.append(history_row(
                "alternating_neural_galerkin",
                basis_epoch,
                adapted_values,
                args.basis_lr,
            ))
            if adapted_values[0] < basis_best_ritz:
                basis_best_ritz = adapted_values[0]
                basis_best_values = adapted_values
                basis_best_state = deepcopy({
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                })
            if basis_epoch == 1 or basis_epoch % args.print_every == 0:
                print(
                    f"Basis {basis_epoch:5d} | ritz={adapted_values[0]:.6e} | "
                    f"physical={adapted_values[2]:.6e} | "
                    f"scaled={adapted_values[3]:.6e}"
                )
        model.load_state_dict(basis_best_state)
        best_ritz = basis_best_values[0]
        best_relative = basis_best_values[2]
        best_scaled_relative = basis_best_values[3]
        best_epoch = len(history) - 1
        best_state = deepcopy(basis_best_state)
        basis_adaptation = {
            "epochs": int(args.basis_epochs),
            "learning_rate": float(args.basis_lr),
            "initial_ritz": float(initial_ritz),
            "best_ritz": float(basis_best_values[0]),
            "best_physical_residual": float(basis_best_values[2]),
            "best_scaled_residual": float(basis_best_values[3]),
            "last_reduced_system": last_information,
        }
        print(
            f"Basis adaptation best | ritz={best_ritz:.6e} | "
            f"physical={best_relative:.6e} | scaled={best_scaled_relative:.6e}"
        )

    if args.fine_tune_scope == "readout":
        trainable_parameters = list(model.output_layer.parameters())
    else:
        trainable_parameters = list(model.parameters())

    # Stage 1 minimizes the convex finite-element potential energy.  In contrast
    # to a squared residual, its Hessian is A rather than A^T A, so the stiffness
    # condition number is not squared by the training objective.
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(args.patience // 5, 20),
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        ritz, residual_loss, _, _, _ = evaluate_network()
        loss = ritz + args.ritz_residual_weight * residual_loss
        loss.backward()
        if args.gradient_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
        optimizer.step()
        # Record and checkpoint the state after the optimizer update.
        model.eval()
        with torch.no_grad():
            updated_ritz, updated_residual_loss, updated_relative, updated_scaled, _ = (
                evaluate_network()
            )
            updated_loss = (
                updated_ritz + args.ritz_residual_weight * updated_residual_loss
            )
        ritz_value = float(updated_ritz.cpu())
        residual_loss_value = float(updated_residual_loss.cpu())
        relative_value = float(updated_relative.cpu())
        scaled_relative_value = float(updated_scaled.cpu())
        loss_value = float(updated_loss.cpu())
        model.train()
        scheduler.step(scaled_relative_value)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append({
            "stage": "ritz_adamw",
            "iteration": epoch,
            "loss": loss_value,
            "ritz_loss": ritz_value,
            "residual_loss": residual_loss_value,
            "equilibrium_relative_l2": relative_value,
            "scaled_equilibrium_relative_l2": scaled_relative_value,
            "learning_rate": learning_rate,
        })
        if scaled_relative_value < best_scaled_relative:
            best_ritz = ritz_value
            best_scaled_relative = scaled_relative_value
            best_relative = relative_value
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            without_improvement = 0
        else:
            without_improvement += 1
        if epoch == 1 or epoch % args.print_every == 0:
            print(
                f"Ritz {epoch:5d} | objective={loss_value:.6e} | "
                f"physical={relative_value:.6e} | scaled={scaled_relative_value:.6e} | "
                f"lr={learning_rate:.3e}"
            )
        if scaled_relative_value <= target_scaled_residual:
            print(f"Target scaled residual reached at epoch {epoch}")
            break
        if args.patience > 0 and without_improvement >= args.patience:
            print(f"Ritz-stage early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    if args.galerkin_initialization and args.galerkin_refresh:
        refreshed = apply_galerkin("galerkin_after_ritz", len(history))
        if refreshed[0] < best_ritz:
            best_ritz = refreshed[0]
            best_scaled_relative = refreshed[3]
            best_relative = refreshed[2]
            best_epoch = len(history) - 1
            best_state = deepcopy(
                {k: v.detach().cpu() for k, v in model.state_dict().items()}
            )
        else:
            model.load_state_dict(best_state)

    # Stage 2 directly reduces the Jacobi-scaled equilibrium residual.  Starting
    # from the Ritz solution avoids asking this more ill-conditioned objective to
    # discover the complete displacement field from scratch.
    if args.residual_epochs > 0 and best_scaled_relative > target_scaled_residual:
        residual_optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=max(args.lr * 0.2, 1.0e-6),
            weight_decay=args.weight_decay,
        )
        residual_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            residual_optimizer,
            mode="min",
            factor=0.5,
            patience=max(args.patience // 8, 15),
        )
        residual_without_improvement = 0
        for epoch in range(1, args.residual_epochs + 1):
            model.train()
            residual_optimizer.zero_grad(set_to_none=True)
            _, residual_loss, _, _, _ = evaluate_network()
            residual_loss.backward()
            if args.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
            residual_optimizer.step()

            model.eval()
            with torch.no_grad():
                ritz, residual_loss, relative, scaled_relative, _ = evaluate_network()
            ritz_value = float(ritz.cpu())
            residual_loss_value = float(residual_loss.cpu())
            relative_value = float(relative.cpu())
            scaled_relative_value = float(scaled_relative.cpu())
            residual_scheduler.step(scaled_relative_value)
            learning_rate = float(residual_optimizer.param_groups[0]["lr"])
            history.append({
                "stage": "residual_adamw",
                "iteration": epoch,
                "loss": residual_loss_value,
                "ritz_loss": ritz_value,
                "residual_loss": residual_loss_value,
                "equilibrium_relative_l2": relative_value,
                "scaled_equilibrium_relative_l2": scaled_relative_value,
                "learning_rate": learning_rate,
            })
            if scaled_relative_value < best_scaled_relative:
                best_ritz = ritz_value
                best_scaled_relative = scaled_relative_value
                best_relative = relative_value
                best_epoch = len(history) - 1
                best_state = deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
                residual_without_improvement = 0
            else:
                residual_without_improvement += 1
            if epoch == 1 or epoch % args.print_every == 0:
                print(
                    f"Residual {epoch:5d} | loss={residual_loss_value:.6e} | "
                    f"physical={relative_value:.6e} | scaled={scaled_relative_value:.6e} | "
                    f"lr={learning_rate:.3e}"
                )
            if scaled_relative_value <= target_scaled_residual:
                print(f"Target scaled residual reached during residual stage at epoch {epoch}")
                break
            if args.patience > 0 and residual_without_improvement >= args.patience:
                print(f"Residual-stage early stop at epoch {epoch}")
                break

        model.load_state_dict(best_state)

        if args.galerkin_initialization and args.galerkin_refresh:
            refreshed = apply_galerkin("galerkin_after_residual", len(history))
            if refreshed[0] < best_ritz:
                best_ritz = refreshed[0]
                best_scaled_relative = refreshed[3]
                best_relative = refreshed[2]
                best_epoch = len(history) - 1
                best_state = deepcopy(
                    {k: v.detach().cpu() for k, v in model.state_dict().items()}
                )
            else:
                model.load_state_dict(best_state)

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            trainable_parameters,
            lr=0.8,
            max_iter=args.lbfgs_steps,
            max_eval=max(args.lbfgs_steps * 2, args.lbfgs_steps + 1),
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            history_size=min(50, args.lbfgs_steps),
            line_search_fn="strong_wolfe",
        )
        calls = 0

        def closure():
            nonlocal calls
            lbfgs.zero_grad(set_to_none=True)
            _, residual_loss, _, _, _ = evaluate_network()
            residual_loss.backward()
            calls += 1
            return residual_loss

        model.train()
        lbfgs.step(closure)
        model.eval()
        with torch.no_grad():
            ritz, residual_loss, relative, scaled_relative, _ = evaluate_network()
        relative_value = float(relative.cpu())
        scaled_relative_value = float(scaled_relative.cpu())
        ritz_value = float(ritz.cpu())
        loss_value = float(residual_loss.cpu())
        history.append({
            "stage": "residual_lbfgs",
            "iteration": calls,
            "loss": loss_value,
            "ritz_loss": ritz_value,
            "residual_loss": loss_value,
            "equilibrium_relative_l2": relative_value,
            "scaled_equilibrium_relative_l2": scaled_relative_value,
            "learning_rate": 0.8,
        })
        print(
            f"L-BFGS calls={calls} | loss={loss_value:.6e} | "
            f"physical={relative_value:.6e} | scaled={scaled_relative_value:.6e}"
        )
        if scaled_relative_value < best_scaled_relative:
            best_ritz = ritz_value
            best_scaled_relative = scaled_relative_value
            best_relative = relative_value
            best_epoch = len(history) - 1
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            model.load_state_dict(best_state)

    model.load_state_dict(best_state)

    training_seconds = time.perf_counter() - started
    inference_started = time.perf_counter()
    raw_U = _model_displacement(
        model,
        feature_tensor,
        equilibrium=equilibrium,
        free_dofs=free,
        prescribed_dofs=constrained,
        prescribed_values=boundary.prescribed_values,
    )
    inference_seconds = time.perf_counter() - inference_started
    raw_solution = _solution_from_displacement(system, boundary, raw_U)
    raw_fields = recover_fields(system, raw_U)

    _save_products(Path(str(output) + "_fcnn_raw"), system, boundary, raw_solution, raw_fields)
    reference_solution = None
    reference_fields = None
    reference_seconds = None
    if args.skip_reference:
        raw_metrics = _self_metrics(raw_solution, raw_fields)
    else:
        # Reference is computed only after training; it never enters the loss.
        reference_started = time.perf_counter()
        reference_solution = solve_linear_system(system, boundary)
        reference_fields = recover_fields(system, reference_solution.displacement)
        reference_seconds = time.perf_counter() - reference_started
        raw_metrics = _comparison(
            raw_solution, raw_fields, reference_solution, reference_fields,
        )
        _save_products(
            Path(str(output) + "_direct_reference"),
            system, boundary, reference_solution, reference_fields,
        )

    enriched_metrics = None
    enrichment_info = None
    enriched_free_correction = None
    if args.residual_enrichment_steps > 0:
        enrichment_iterations = 0

        def enrichment_callback(_):
            nonlocal enrichment_iterations
            enrichment_iterations += 1

        initial_enrichment_residual = (
            float(np.linalg.norm(rhs - Kff @ raw_U[free]))
            / max(float(np.linalg.norm(rhs)), 1.0e-30)
        )
        enrichment_started = time.perf_counter()
        enriched_uf, enrichment_status = cg(
            Kff,
            rhs,
            x0=raw_U[free],
            M=diags(1.0 / np.maximum(np.abs(Kff.diagonal()), 1.0e-30)),
            rtol=args.residual_enrichment_rtol,
            atol=0.0,
            maxiter=args.residual_enrichment_steps,
            callback=enrichment_callback,
        )
        enrichment_seconds = time.perf_counter() - enrichment_started
        enriched_U = raw_U.copy()
        enriched_U[free] = enriched_uf
        enriched_free_correction = enriched_U[free] - raw_U[free]
        enriched_solution = _solution_from_displacement(
            system, boundary, enriched_U,
        )
        enriched_fields = recover_fields(system, enriched_U)
        if reference_solution is None:
            enriched_metrics = _self_metrics(enriched_solution, enriched_fields)
        else:
            enriched_metrics = _comparison(
                enriched_solution,
                enriched_fields,
                reference_solution,
                reference_fields,
            )
        final_enrichment_residual = (
            float(np.linalg.norm(enriched_solution.residual_free))
            / max(float(np.linalg.norm(enriched_solution.rhs_free)), 1.0e-30)
        )
        enrichment_info = {
            "method": "adaptive_preconditioned_residual_galerkin_enrichment",
            "mathematical_relation": (
                "The enrichment directions form the same Krylov subspace used by PCG; "
                "this is a hybrid neural-Krylov solver, not a pure FCNN output."
            ),
            "maximum_steps": int(args.residual_enrichment_steps),
            "iterations": int(enrichment_iterations),
            "solver_info": int(enrichment_status),
            "rtol": float(args.residual_enrichment_rtol),
            "seconds": float(enrichment_seconds),
            "initial_equilibrium_relative_l2": initial_enrichment_residual,
            "final_equilibrium_relative_l2": final_enrichment_residual,
        }
        _save_products(
            Path(str(output) + "_fcnn_enriched"),
            system,
            boundary,
            enriched_solution,
            enriched_fields,
        )

    corrected_metrics = None
    correction_info = None
    if args.correction == "pcg":
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        correction_started = time.perf_counter()
        corrected_uf, info = cg(
            Kff,
            rhs,
            x0=raw_U[free],
            M=diags(1.0 / np.maximum(np.abs(Kff.diagonal()), 1.0e-30)),
            rtol=args.correction_rtol,
            atol=0.0,
            maxiter=args.correction_maxiter,
            callback=callback,
        )
        correction_seconds = time.perf_counter() - correction_started
        corrected_U = raw_U.copy()
        corrected_U[free] = corrected_uf
        corrected_solution = _solution_from_displacement(system, boundary, corrected_U)
        corrected_fields = recover_fields(system, corrected_U)
        if reference_solution is None:
            corrected_metrics = _self_metrics(corrected_solution, corrected_fields)
        else:
            corrected_metrics = _comparison(
                corrected_solution, corrected_fields, reference_solution, reference_fields,
            )
        correction_info = {
            "solver_info": int(info),
            "iterations": int(iterations),
            "seconds": float(correction_seconds),
        }
        _save_products(
            Path(str(output) + "_pcg_corrected"),
            system, boundary, corrected_solution, corrected_fields,
        )

    checkpoint = {
        "model_family": MODEL_FAMILY,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "model_config": model.config(),
        "feature_names": FEATURE_NAMES,
        "displacement_scale": displacement_scale,
        "solution_parameterization": (
            "Uf=U_lift+D^(-1/2)*Phi_theta*c+Psi_m*a"
            if enriched_free_correction is not None
            else "Uf=U_lift+D^(-1/2)*Phi_theta*c"
        ),
        "lifting": "homogeneous_tension_with_poisson_contraction",
        "scaled_variable_scale": float(equilibrium.scaled_variable_scale.cpu()),
        "mesh": str(mesh_path.resolve()),
        "mesh_nodes": system.n_nodes,
        "mesh_triangles": len(system.triangles),
        "material": {
            "E": args.E,
            "nu": args.nu,
            "thickness": args.thickness,
            "behavior": args.behavior,
        },
        "loading": {"top_uy": args.top_uy, "bottom_uy": args.bottom_uy},
        "geometry": geometry.to_dict(),
        "warm_start": str(Path(args.warm_start).resolve()) if args.warm_start else None,
        "lifting_physical_residual": lifting_values[2],
        "lifting_scaled_residual": lifting_values[3],
        "initial_training_residual": initial_relative,
        "initial_scaled_training_residual": initial_scaled_relative,
        "best_training_residual": best_relative,
        "best_scaled_training_residual": best_scaled_relative,
        "best_ritz_potential": best_ritz,
        "best_history_index": best_epoch,
        "selection_metric": "scaled_equilibrium_relative_l2_after_galerkin_initialization",
        "galerkin_events": galerkin_events,
        "basis_adaptation": basis_adaptation,
        "residual_enrichment": enrichment_info,
        "residual_enrichment_free_correction": (
            torch.as_tensor(enriched_free_correction, dtype=torch.float64)
            if enriched_free_correction is not None
            else None
        ),
        "training_strategy": {
            "stage_0": "neural_basis_galerkin_readout_initialization",
            "stage_1": "alternating_hidden_basis_and_reduced_galerkin_optimization",
            "stage_2": "jacobi_scaled_ritz_fine_tuning",
            "stage_1_residual_weight": args.ritz_residual_weight,
            "stage_3": "jacobi_scaled_residual_fine_tuning",
            "stage_4": "lbfgs_scaled_residual_fine_tuning",
            "galerkin_refresh": bool(args.galerkin_refresh),
        },
        "training_seconds": training_seconds,
        "training_args": vars(args),
    }
    torch.save(checkpoint, output.with_suffix(".pt"))
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    report = {
        "format": "coordinate-fcnn-sm-neural-galerkin-report-v3",
        "model_family": MODEL_FAMILY,
        "mesh": str(mesh_path.resolve()),
        "nodes": system.n_nodes,
        "triangles": len(system.triangles),
        "free_dofs": len(free),
        "training_uses_displacement_or_stress_labels": False,
        "reference_computed_after_training": not args.skip_reference,
        "warm_start": checkpoint["warm_start"],
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "direct_reference_seconds": reference_seconds,
        "lifting_physical_residual": lifting_values[2],
        "lifting_scaled_residual": lifting_values[3],
        "initial_training_residual": initial_relative,
        "initial_scaled_training_residual": initial_scaled_relative,
        "best_training_residual": best_relative,
        "best_scaled_training_residual": best_scaled_relative,
        "best_ritz_potential": best_ritz,
        "selection_metric": "scaled_equilibrium_relative_l2_after_galerkin_initialization",
        "galerkin_events": galerkin_events,
        "basis_adaptation": basis_adaptation,
        "raw_fcnn": raw_metrics,
        "residual_enrichment": enrichment_info,
        "enriched_fcnn": enriched_metrics,
        "pcg_correction": correction_info,
        "corrected": corrected_metrics,
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print("\nRaw FCNN-SM:", raw_metrics)
    if enriched_metrics is not None:
        print("Residual-enriched FCNN-SM:", enriched_metrics)
        print("Residual enrichment:", enrichment_info)
    if corrected_metrics is not None:
        print("PCG corrected:", corrected_metrics)
        print("PCG info:", correction_info)
    print(f"Training seconds: {training_seconds:.3f}")
    if reference_seconds is not None:
        print(f"Direct reference seconds: {reference_seconds:.3f}")
    print(
        f"Saved: {output.with_suffix('.pt')}, {output.with_suffix('.csv')}, "
        f"{output.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
