"""Open arbitrary crack graphs by duplicating constrained-chain nodes."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _edge(first: int, second: int) -> tuple[int, int]:
    return (min(int(first), int(second)), max(int(first), int(second)))


def _coordinate_lookup(points: np.ndarray, tolerance: float) -> dict[tuple[int, int], int]:
    quantum = max(float(tolerance), 1.0e-14)
    lookup: dict[tuple[int, int], int] = {}
    for index, point in enumerate(points):
        key = tuple(np.rint(point / quantum).astype(np.int64))
        if key in lookup and np.linalg.norm(points[lookup[key]] - point) > tolerance:
            raise ValueError("Mesh coordinate lookup is ambiguous at the requested tolerance")
        lookup[key] = index
    return lookup


def _chain_node_ids(
    points: np.ndarray,
    chains: tuple[np.ndarray, ...],
    tolerance: float,
) -> list[list[int]]:
    lookup = _coordinate_lookup(points, tolerance)
    quantum = max(float(tolerance), 1.0e-14)
    result = []
    for chain_index, chain in enumerate(chains):
        base_identifiers = []
        coordinates = np.asarray(chain, dtype=float)
        for coordinate in coordinates:
            key = tuple(np.rint(coordinate / quantum).astype(np.int64))
            identifier = lookup.get(key)
            if identifier is None or np.linalg.norm(points[identifier] - coordinate) > tolerance * 2.0:
                raise ValueError(
                    f"Constraint node {coordinate.tolist()} from crack chain "
                    f"{chain_index} is absent from the triangulation"
                )
            base_identifiers.append(int(identifier))
        # Other deterministic samples (for example a branch ring or an offset
        # layer from a neighboring segment) may land exactly on this chain.
        # Include them so the crack constraint and both free faces are atomic.
        identifiers = []
        for first, second in zip(base_identifiers[:-1], base_identifiers[1:]):
            start = points[first]
            vector = points[second] - start
            length2 = float(vector @ vector)
            parameter = ((points - start) @ vector) / length2
            closest = start + np.clip(parameter, 0.0, 1.0)[:, None] * vector
            on_segment = (
                (np.linalg.norm(points - closest, axis=1) <= tolerance)
                & (parameter >= -tolerance)
                & (parameter <= 1.0 + tolerance)
            )
            atomic = np.flatnonzero(on_segment)
            atomic = atomic[np.argsort(parameter[atomic])].tolist()
            if identifiers and atomic and identifiers[-1] == atomic[0]:
                atomic = atomic[1:]
            identifiers.extend(int(value) for value in atomic)
        result.append(identifiers)
    return result


def duplicate_crack_graph_nodes(
    triangles: np.ndarray,
    points: np.ndarray,
    geometry,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create two free faces for curved or multiple non-branching cracks.

    Physical crack tips remain shared and boundary mouths are duplicated.
    Branched graphs are dispatched to the local material-sector splitter.
    """
    if geometry.has_branches:
        return duplicate_branch_crack_graph_nodes(
            triangles,
            points,
            geometry,
            tolerance=tolerance,
        )
    tri = np.asarray(triangles, dtype=np.int64).copy()
    pts = np.asarray(points, dtype=float).copy()
    if len(tri) == 0:
        return tri.astype(np.int32), pts, {}
    chains = geometry.crack_chains
    chain_ids = _chain_node_ids(pts, chains, tolerance)

    tip_ids = set()
    for tip in np.asarray(geometry.crack_tips, dtype=float):
        distances = np.linalg.norm(pts - tip, axis=1)
        selected = int(np.argmin(distances))
        if distances[selected] > tolerance * 2.0:
            raise ValueError(f"No mesh node coincides with crack tip {tip.tolist()}")
        tip_ids.add(selected)

    duplicate_candidates = sorted(
        {identifier for chain in chain_ids for identifier in chain if identifier not in tip_ids}
    )
    if not duplicate_candidates:
        raise ValueError("No crack-face nodes are available for duplication")

    # An oriented local normal is accumulated at every crack node.  For a
    # corner this is the normalized bisector of adjacent segment normals.
    normal_sum: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=float))
    for chain in chain_ids:
        for first, second in zip(chain[:-1], chain[1:]):
            vector = pts[second] - pts[first]
            length = float(np.linalg.norm(vector))
            if length <= tolerance:
                raise ValueError("A discretized crack chain contains a zero-length edge")
            normal = np.asarray([-vector[1], vector[0]], dtype=float) / length
            normal_sum[first] += normal
            normal_sum[second] += normal
    node_normal = {}
    for identifier in duplicate_candidates:
        normal = normal_sum[identifier]
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-12:
            raise ValueError(
                f"Crack normal is ambiguous at node {identifier}; split the polyline "
                "at a reversal or add an explicit branch definition"
            )
        node_normal[identifier] = normal / norm

    old_to_duplicate: dict[int, int] = {}
    for old in duplicate_candidates:
        old_to_duplicate[old] = len(pts)
        pts = np.vstack((pts, pts[old]))

    # Crack-edge adjacency is the authoritative face classification.  It
    # guarantees that each constrained edge gets exactly one triangle on each
    # side even at a curved corner where an averaged nodal normal is ambiguous.
    edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, identifiers in enumerate(tri):
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_triangles[_edge(identifiers[first], identifiers[second])].append(triangle_index)
    forced_negative: set[int] = set()
    forced_positive: set[int] = set()
    for chain in chain_ids:
        for first, second in zip(chain[:-1], chain[1:]):
            start = pts[first]
            vector = pts[second] - start
            for triangle_index in edge_triangles.get(_edge(first, second), ()): 
                centroid = np.mean(pts[tri[triangle_index]], axis=0)
                relative = centroid - start
                signed = float(vector[0] * relative[1] - vector[1] * relative[0])
                if signed < -tolerance:
                    forced_negative.add(triangle_index)
                elif signed > tolerance:
                    forced_positive.add(triangle_index)
    conflict = forced_negative & forced_positive
    if conflict:
        raise ValueError(
            "A triangle is adjacent to opposing crack-face sectors at a sharp "
            f"polyline corner: {sorted(conflict)[:10]}"
        )

    # Each incident triangle is assigned using its centroid relative to the
    # local oriented normal at the crack node.  The negative face receives the
    # coincident duplicate node.
    for triangle_index, identifiers in enumerate(tri):
        centroid = np.mean(pts[identifiers], axis=0)
        updated = identifiers.copy()
        for local_index, old in enumerate(identifiers):
            old = int(old)
            if old not in old_to_duplicate:
                continue
            should_duplicate = triangle_index in forced_negative
            if triangle_index not in forced_negative and triangle_index not in forced_positive:
                signed = float((centroid - pts[old]) @ node_normal[old])
                should_duplicate = signed < -tolerance
            if should_duplicate:
                updated[local_index] = old_to_duplicate[old]
        tri[triangle_index] = updated

    crack_records = []
    all_positive_edges = set()
    all_negative_edges = set()
    positive_nodes = set()
    negative_nodes = set()
    for chain in chain_ids:
        positive_chain = [int(value) for value in chain]
        negative_chain = [old_to_duplicate.get(int(value), int(value)) for value in chain]
        positive_edges = {_edge(a, b) for a, b in zip(positive_chain[:-1], positive_chain[1:])}
        negative_edges = {_edge(a, b) for a, b in zip(negative_chain[:-1], negative_chain[1:])}
        all_positive_edges.update(positive_edges)
        all_negative_edges.update(negative_edges)
        positive_nodes.update(positive_chain)
        negative_nodes.update(negative_chain)
        crack_records.append(
            {
                "positive_chain": positive_chain,
                "negative_chain": negative_chain,
                "positive_edges": positive_edges,
                "negative_edges": negative_edges,
            }
        )

    edge_incidence: dict[tuple[int, int], int] = defaultdict(int)
    for identifiers in tri:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_incidence[_edge(identifiers[first], identifiers[second])] += 1
    bad_positive = [edge for edge in all_positive_edges if edge_incidence[edge] != 1]
    bad_negative = [edge for edge in all_negative_edges if edge_incidence[edge] != 1]
    if bad_positive or bad_negative:
        raise ValueError(
            "Polyline crack faces are not free boundaries after node duplication: "
            f"positive={bad_positive[:5]}, negative={bad_negative[:5]}"
        )

    info = {
        "cracks": crack_records,
        "crack_upper": all_positive_edges,
        "crack_lower": all_negative_edges,
        "crack_upper_nodes": positive_nodes - tip_ids,
        "crack_lower_nodes": negative_nodes - tip_ids,
        "crack_tip_nodes": tip_ids,
        "old_to_dup": old_to_duplicate,
        "has_branches": False,
    }
    return tri.astype(np.int32), pts, info


def _matching_node_ids(
    points: np.ndarray,
    coordinates: np.ndarray,
    tolerance: float,
) -> set[int]:
    identifiers = set()
    for coordinate in np.asarray(coordinates, dtype=float):
        distances = np.linalg.norm(points - coordinate, axis=1)
        selected = int(np.argmin(distances))
        if distances[selected] > tolerance * 2.0:
            raise ValueError(f"No mesh node coincides with crack point {coordinate.tolist()}")
        identifiers.add(selected)
    return identifiers


def _connected_components(vertices: list[int], adjacency: dict[int, set[int]]):
    remaining = set(vertices)
    components = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.remove(seed)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            neighbors = adjacency[current] & remaining
            stack.extend(sorted(neighbors, reverse=True))
            remaining.difference_update(neighbors)
        components.append(sorted(component))
    return components


def duplicate_branch_crack_graph_nodes(
    triangles: np.ndarray,
    points: np.ndarray,
    geometry,
    *,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Open a branched crack graph using local material-sector components.

    Around a crack node, incident triangles are connected only across radial
    mesh edges that are not crack constraints.  The resulting local connected
    components are the material sectors.  One coincident node copy is assigned
    to each sector; therefore the method does not depend on a global
    positive/negative face convention and naturally handles branch vertices.
    """
    tri_original = np.asarray(triangles, dtype=np.int64)
    tri = tri_original.copy()
    pts = np.asarray(points, dtype=float).copy()
    if len(tri) == 0:
        return tri.astype(np.int32), pts, {}

    chain_ids = _chain_node_ids(pts, geometry.crack_chains, tolerance)
    crack_edges = {
        _edge(first, second)
        for chain in chain_ids
        for first, second in zip(chain[:-1], chain[1:])
    }
    crack_nodes = sorted({node for edge in crack_edges for node in edge})
    tip_ids = _matching_node_ids(pts, geometry.crack_tips, tolerance)
    branch_ids = _matching_node_ids(pts, geometry.branch_points, tolerance)

    edge_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)
    node_triangles: dict[int, list[int]] = defaultdict(list)
    crack_neighbors: dict[int, set[int]] = defaultdict(set)
    for triangle_index, identifiers in enumerate(tri_original):
        for node in identifiers:
            node_triangles[int(node)].append(triangle_index)
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_triangles[_edge(identifiers[first], identifiers[second])].append(
                triangle_index,
            )
    for first, second in crack_edges:
        crack_neighbors[first].add(second)
        crack_neighbors[second].add(first)
        incidence = len(edge_triangles.get((first, second), ()))
        if incidence != 2:
            raise ValueError(
                f"Crack constraint {(first, second)} has {incidence} adjacent "
                "triangles; two are required before face separation"
            )

    node_components: dict[int, list[list[int]]] = {}
    triangle_node_copy: dict[tuple[int, int], int] = {}
    node_copies: dict[int, list[int]] = {}
    for node in crack_nodes:
        incident = sorted(set(node_triangles[node]))
        adjacency = {triangle_index: set() for triangle_index in incident}
        for triangle_index in incident:
            identifiers = tri_original[triangle_index]
            for neighbor in identifiers:
                neighbor = int(neighbor)
                if neighbor == node or _edge(node, neighbor) in crack_edges:
                    continue
                for adjacent_triangle in edge_triangles[_edge(node, neighbor)]:
                    if adjacent_triangle != triangle_index:
                        adjacency[triangle_index].add(adjacent_triangle)
                        adjacency[adjacent_triangle].add(triangle_index)
        components = _connected_components(incident, adjacency)
        node_components[node] = components

        ray_count = len(crack_neighbors[node])
        if node in tip_ids:
            expected_sectors = 1
        elif node in branch_ids:
            expected_sectors = ray_count
        else:
            # Includes ordinary degree-two chain nodes and a boundary mouth.
            expected_sectors = 2
        if len(components) != expected_sectors:
            role = "tip" if node in tip_ids else "branch" if node in branch_ids else "face"
            raise ValueError(
                f"Crack {role} node {node} has {len(components)} material "
                f"sectors; expected {expected_sectors} (crack rays={ray_count})"
            )

        copies = [node]
        for _ in components[1:]:
            copies.append(len(pts))
            pts = np.vstack((pts, pts[node]))
        node_copies[node] = copies
        for component_index, component in enumerate(components):
            copy = copies[component_index]
            for triangle_index in component:
                triangle_node_copy[(triangle_index, node)] = copy
                local = np.flatnonzero(tri[triangle_index] == node)
                if len(local) != 1:
                    raise ValueError(
                        f"Triangle {triangle_index} does not contain crack node {node} once"
                    )
                tri[triangle_index, local[0]] = copy

    face_edges = set()
    face_edge_pairs = []
    for crack_edge in sorted(crack_edges):
        first, second = crack_edge
        pair = []
        for triangle_index in edge_triangles[crack_edge]:
            opened_edge = _edge(
                triangle_node_copy[(triangle_index, first)],
                triangle_node_copy[(triangle_index, second)],
            )
            pair.append(opened_edge)
            face_edges.add(opened_edge)
        if len(set(pair)) != 2:
            raise ValueError(
                f"Crack constraint {crack_edge} did not open into two distinct faces"
            )
        face_edge_pairs.append((pair[0], pair[1]))

    edge_incidence: dict[tuple[int, int], int] = defaultdict(int)
    for identifiers in tri:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge_incidence[_edge(identifiers[first], identifiers[second])] += 1
    bad_faces = [edge for edge in face_edges if edge_incidence[edge] != 1]
    if bad_faces:
        raise ValueError(
            "Branched crack faces are not free boundaries after sector splitting: "
            f"{bad_faces[:10]}"
        )

    # Coincident copies are retained explicitly for downstream inspection.
    copy_offsets = [0]
    copy_values = []
    copy_source_nodes = []
    for source in crack_nodes:
        copies = node_copies[source]
        copy_source_nodes.append(source)
        copy_values.extend(copies)
        copy_offsets.append(len(copy_values))
    crack_face_nodes = {
        node
        for source, copies in node_copies.items()
        if source not in tip_ids
        for node in copies
    }
    branch_sector_nodes = {
        node for source in branch_ids for node in node_copies[source]
    }
    info = {
        "cracks": [],
        "crack_upper": set(),
        "crack_lower": set(),
        "crack_upper_nodes": set(),
        "crack_lower_nodes": set(),
        "crack_face_edges": face_edges,
        "crack_face_edge_pairs": face_edge_pairs,
        "crack_face_nodes": crack_face_nodes,
        "crack_tip_nodes": tip_ids,
        "branch_sector_nodes": branch_sector_nodes,
        "node_copy_sources": np.asarray(copy_source_nodes, dtype=np.int64),
        "node_copy_offsets": np.asarray(copy_offsets, dtype=np.int64),
        "node_copy_values": np.asarray(copy_values, dtype=np.int64),
        "old_to_dup": {
            source: copies[1:]
            for source, copies in node_copies.items()
            if len(copies) > 1
        },
        "has_branches": True,
        "shared_crack_nodes": False,
        "topology_mode": "open_branch_sectors",
    }
    return tri.astype(np.int32), pts, info


__all__ = ["duplicate_branch_crack_graph_nodes", "duplicate_crack_graph_nodes"]
