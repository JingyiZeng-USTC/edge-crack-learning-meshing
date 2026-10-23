"""Versioned, mismatch-safe files for the three deterministic FEM stages."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .assembly import FEMSystem
from .boundary import BoundaryConditions
from .recovery import FieldResults
from .solver import Solution


SYSTEM_FORMAT = "module10-fem-system-v1"
SOLUTION_FORMAT = "module10-fem-solution-v1"
FIELDS_FORMAT = "module10-fem-fields-v1"


def _text_scalar(data, key: str) -> str:
    value = np.asarray(data[key])
    if value.size != 1:
        raise ValueError(f"{key} must be a scalar string")
    return str(value.reshape(()).item())


def system_identifier(system: FEMSystem) -> str:
    """Hash the complete assembled system so stage files cannot be mixed."""
    digest = hashlib.sha256()
    for array, dtype in (
        (system.points, np.dtype("<f8")),
        (system.triangles, np.dtype("<i8")),
        (system.stiffness.data, np.dtype("<f8")),
        (system.stiffness.indices, np.dtype("<i8")),
        (system.stiffness.indptr, np.dtype("<i8")),
        (np.asarray(system.stiffness.shape), np.dtype("<i8")),
        (system.constitutive, np.dtype("<f8")),
        (np.asarray([system.thickness]), np.dtype("<f8")),
        (system.element_areas, np.dtype("<f8")),
        (system.element_B, np.dtype("<f8")),
        (system.element_dof_map, np.dtype("<i8")),
    ):
        values = np.ascontiguousarray(np.asarray(array, dtype=dtype))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_system(system: FEMSystem) -> None:
    n_elements = len(system.triangles)
    if system.stiffness.shape != (system.n_dofs, system.n_dofs):
        raise ValueError("Stiffness shape does not match the mesh DOF count")
    if system.element_areas.shape != (n_elements,):
        raise ValueError("element_areas has the wrong shape")
    if system.element_B.shape != (n_elements, 3, 6):
        raise ValueError("element_B has the wrong shape")
    if system.element_dof_map.shape != (n_elements, 6):
        raise ValueError("element_dof_map has the wrong shape")
    if system.constitutive.shape != (3, 3):
        raise ValueError("constitutive must have shape (3, 3)")
    if system.triangles.size and (
        system.triangles.min() < 0 or system.triangles.max() >= system.n_nodes
    ):
        raise ValueError("Triangle connectivity contains an invalid node index")
    if system.element_dof_map.size and (
        system.element_dof_map.min() < 0
        or system.element_dof_map.max() >= system.n_dofs
    ):
        raise ValueError("Element DOF map contains an invalid index")
    if np.any(system.element_areas <= 0.0):
        raise ValueError("The system contains a non-positive element area")
    arrays = (
        system.points,
        system.constitutive,
        system.element_areas,
        system.element_B,
        system.stiffness.data,
    )
    if not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("The FEM system contains NaN or infinity")


def save_system(path: str | Path, system: FEMSystem, *, source_mesh: str = "") -> str:
    validate_system(system)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    identifier = system_identifier(system)
    np.savez_compressed(
        destination,
        format=np.asarray(SYSTEM_FORMAT),
        system_id=np.asarray(identifier),
        source_mesh=np.asarray(str(source_mesh)),
        points=np.asarray(system.points, dtype=np.float64),
        triangles=np.asarray(system.triangles, dtype=np.int64),
        stiffness_data=np.asarray(system.stiffness.data, dtype=np.float64),
        stiffness_indices=np.asarray(system.stiffness.indices, dtype=np.int64),
        stiffness_indptr=np.asarray(system.stiffness.indptr, dtype=np.int64),
        stiffness_shape=np.asarray(system.stiffness.shape, dtype=np.int64),
        constitutive=np.asarray(system.constitutive, dtype=np.float64),
        thickness=np.asarray(system.thickness, dtype=np.float64),
        element_areas=np.asarray(system.element_areas, dtype=np.float64),
        element_B=np.asarray(system.element_B, dtype=np.float64),
        element_dof_map=np.asarray(system.element_dof_map, dtype=np.int64),
    )
    return identifier


def load_system(path: str | Path) -> tuple[FEMSystem, str, dict]:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if _text_scalar(data, "format") != SYSTEM_FORMAT:
            raise ValueError(f"{source} is not a {SYSTEM_FORMAT} file")
        saved_identifier = _text_scalar(data, "system_id")
        shape = tuple(np.asarray(data["stiffness_shape"], dtype=np.int64).tolist())
        stiffness = csr_matrix(
            (
                np.asarray(data["stiffness_data"], dtype=np.float64),
                np.asarray(data["stiffness_indices"], dtype=np.int64),
                np.asarray(data["stiffness_indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
        system = FEMSystem(
            points=np.asarray(data["points"], dtype=np.float64),
            triangles=np.asarray(data["triangles"], dtype=np.int64),
            stiffness=stiffness,
            constitutive=np.asarray(data["constitutive"], dtype=np.float64),
            thickness=float(np.asarray(data["thickness"]).item()),
            element_areas=np.asarray(data["element_areas"], dtype=np.float64),
            element_B=np.asarray(data["element_B"], dtype=np.float64),
            element_dof_map=np.asarray(data["element_dof_map"], dtype=np.int64),
        )
        metadata = {
            "source_mesh": _text_scalar(data, "source_mesh"),
            "path": str(source.resolve()),
        }
    validate_system(system)
    computed = system_identifier(system)
    if computed != saved_identifier:
        raise ValueError("FEM system hash mismatch: the file is corrupted or was edited")
    return system, saved_identifier, metadata


def _pack_named_nodes(named_nodes: dict[str, np.ndarray]):
    names = sorted(str(name) for name in named_nodes)
    offsets = [0]
    values = []
    for name in names:
        nodes = np.asarray(named_nodes[name], dtype=np.int64).reshape(-1)
        values.append(nodes)
        offsets.append(offsets[-1] + len(nodes))
    packed = np.concatenate(values) if values else np.empty(0, dtype=np.int64)
    return np.asarray(names, dtype="U64"), np.asarray(offsets, dtype=np.int64), packed


def _unpack_named_nodes(names, offsets, values) -> dict[str, np.ndarray]:
    names = np.asarray(names).astype(str)
    offsets = np.asarray(offsets, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    if len(offsets) != len(names) + 1 or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("Invalid named-node packed arrays")
    return {
        str(name): values[offsets[index]:offsets[index + 1]].copy()
        for index, name in enumerate(names)
    }


def save_solution(
    path: str | Path,
    system: FEMSystem,
    boundary: BoundaryConditions,
    solution: Solution,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    names, offsets, nodes = _pack_named_nodes(boundary.named_nodes)
    np.savez_compressed(
        destination,
        format=np.asarray(SOLUTION_FORMAT),
        system_id=np.asarray(system_identifier(system)),
        prescribed_dofs=np.asarray(boundary.prescribed_dofs, dtype=np.int64),
        prescribed_values=np.asarray(boundary.prescribed_values, dtype=np.float64),
        force=np.asarray(boundary.force, dtype=np.float64),
        named_node_names=names,
        named_node_offsets=offsets,
        named_node_values=nodes,
        displacement=np.asarray(solution.displacement, dtype=np.float64),
        displacement_xy=np.asarray(solution.displacement_xy, dtype=np.float64),
        reactions=np.asarray(solution.reactions, dtype=np.float64),
        free_dofs=np.asarray(solution.free_dofs, dtype=np.int64),
        solution_prescribed_dofs=np.asarray(solution.prescribed_dofs, dtype=np.int64),
        rhs_free=np.asarray(solution.rhs_free, dtype=np.float64),
        residual_free=np.asarray(solution.residual_free, dtype=np.float64),
        strain_energy=np.asarray(solution.strain_energy, dtype=np.float64),
    )


def load_solution(
    path: str | Path,
    system: FEMSystem,
) -> tuple[BoundaryConditions, Solution]:
    source = Path(path)
    expected_id = system_identifier(system)
    with np.load(source, allow_pickle=False) as data:
        if _text_scalar(data, "format") != SOLUTION_FORMAT:
            raise ValueError(f"{source} is not a {SOLUTION_FORMAT} file")
        if _text_scalar(data, "system_id") != expected_id:
            raise ValueError("Solution belongs to a different FEM system")
        boundary = BoundaryConditions(
            prescribed_dofs=np.asarray(data["prescribed_dofs"], dtype=np.int64),
            prescribed_values=np.asarray(data["prescribed_values"], dtype=np.float64),
            force=np.asarray(data["force"], dtype=np.float64),
            named_nodes=_unpack_named_nodes(
                data["named_node_names"],
                data["named_node_offsets"],
                data["named_node_values"],
            ),
        )
        solution = Solution(
            displacement=np.asarray(data["displacement"], dtype=np.float64),
            displacement_xy=np.asarray(data["displacement_xy"], dtype=np.float64),
            reactions=np.asarray(data["reactions"], dtype=np.float64),
            free_dofs=np.asarray(data["free_dofs"], dtype=np.int64),
            prescribed_dofs=np.asarray(data["solution_prescribed_dofs"], dtype=np.int64),
            rhs_free=np.asarray(data["rhs_free"], dtype=np.float64),
            residual_free=np.asarray(data["residual_free"], dtype=np.float64),
            strain_energy=float(np.asarray(data["strain_energy"]).item()),
        )
    if boundary.force.shape != (system.n_dofs,):
        raise ValueError("Solution force vector has the wrong size")
    if boundary.prescribed_dofs.shape != boundary.prescribed_values.shape:
        raise ValueError("Prescribed DOFs and values have different sizes")
    if boundary.prescribed_dofs.size and (
        boundary.prescribed_dofs.min() < 0
        or boundary.prescribed_dofs.max() >= system.n_dofs
    ):
        raise ValueError("Solution contains an invalid prescribed DOF")
    if solution.displacement.shape != (system.n_dofs,):
        raise ValueError("Solution displacement has the wrong size")
    if solution.displacement_xy.shape != (system.n_nodes, 2):
        raise ValueError("Solution displacement_xy has the wrong shape")
    if not np.array_equal(solution.displacement_xy, solution.displacement.reshape(-1, 2)):
        raise ValueError("Flat and nodal displacement arrays are inconsistent")
    numeric_arrays = (
        boundary.prescribed_values,
        boundary.force,
        solution.displacement,
        solution.reactions,
        solution.rhs_free,
        solution.residual_free,
    )
    if not all(np.all(np.isfinite(values)) for values in numeric_arrays):
        raise ValueError("Solution file contains NaN or infinity")
    return boundary, solution


def save_fields(
    path: str | Path,
    system: FEMSystem,
    fields: FieldResults,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        format=np.asarray(FIELDS_FORMAT),
        system_id=np.asarray(system_identifier(system)),
        element_strain=np.asarray(fields.element_strain, dtype=np.float64),
        element_stress=np.asarray(fields.element_stress, dtype=np.float64),
        element_von_mises=np.asarray(fields.element_von_mises, dtype=np.float64),
        nodal_strain=np.asarray(fields.nodal_strain, dtype=np.float64),
        nodal_stress=np.asarray(fields.nodal_stress, dtype=np.float64),
        nodal_von_mises=np.asarray(fields.nodal_von_mises, dtype=np.float64),
    )


def load_fields(path: str | Path, system: FEMSystem) -> FieldResults:
    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if _text_scalar(data, "format") != FIELDS_FORMAT:
            raise ValueError(f"{source} is not a {FIELDS_FORMAT} file")
        if _text_scalar(data, "system_id") != system_identifier(system):
            raise ValueError("Field results belong to a different FEM system")
        fields = FieldResults(
            element_strain=np.asarray(data["element_strain"], dtype=np.float64),
            element_stress=np.asarray(data["element_stress"], dtype=np.float64),
            element_von_mises=np.asarray(data["element_von_mises"], dtype=np.float64),
            nodal_strain=np.asarray(data["nodal_strain"], dtype=np.float64),
            nodal_stress=np.asarray(data["nodal_stress"], dtype=np.float64),
            nodal_von_mises=np.asarray(data["nodal_von_mises"], dtype=np.float64),
        )
    expected_shapes = {
        "element_strain": (len(system.triangles), 3),
        "element_stress": (len(system.triangles), 3),
        "element_von_mises": (len(system.triangles),),
        "nodal_strain": (system.n_nodes, 3),
        "nodal_stress": (system.n_nodes, 3),
        "nodal_von_mises": (system.n_nodes,),
    }
    for name, expected in expected_shapes.items():
        values = getattr(fields, name)
        if values.shape != expected:
            raise ValueError(f"{name} has shape {values.shape}, expected {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains NaN or infinity")
    return fields


__all__ = [
    "FIELDS_FORMAT",
    "SOLUTION_FORMAT",
    "SYSTEM_FORMAT",
    "load_fields",
    "load_solution",
    "load_system",
    "save_fields",
    "save_solution",
    "save_system",
    "system_identifier",
    "validate_system",
]
