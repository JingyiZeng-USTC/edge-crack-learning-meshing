from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


DEFAULT_MIN_TOP_CLEARANCE_RATIO = 0.10


@dataclass(frozen=True)
class EdgeCrack:
    """Rectangular plate with one straight crack starting on the left edge."""

    width: float = 1.0
    height: float = 2.0
    y0: float = 1.0
    length: float = 0.25
    angle_deg: float = 0.0

    @property
    def start(self) -> np.ndarray:
        return np.array([0.0, self.y0], dtype=np.float64)

    @property
    def tip(self) -> np.ndarray:
        theta = np.deg2rad(self.angle_deg)
        return self.start + self.length * np.array([np.cos(theta), np.sin(theta)])

    def validate(self, min_top_clearance: float | None = None) -> None:
        tip = self.tip
        clearance = (
            DEFAULT_MIN_TOP_CLEARANCE_RATIO * self.height
            if min_top_clearance is None
            else float(min_top_clearance)
        )
        if not (0.0 <= clearance < self.height):
            raise ValueError("minimum top clearance must lie in [0, plate height)")
        if not (0.0 < self.y0 < self.height):
            raise ValueError("crack start must lie inside the left boundary")
        if self.length <= 0.0:
            raise ValueError("crack length must be positive")
        if not (0.0 < tip[0] < self.width and 0.0 < tip[1] < self.height):
            raise ValueError("crack tip must lie inside the plate")
        highest_y = max(float(self.y0), float(tip[1]))
        actual_clearance = self.height - highest_y
        if actual_clearance < clearance - 1.0e-12:
            raise ValueError(
                "crack is too close to the loaded top edge: "
                f"clearance={actual_clearance:.6g}, required>={clearance:.6g}"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        min_top_clearance: float | None = None,
    ) -> "EdgeCrack":
        for _ in range(100):
            crack = cls(
                y0=float(rng.uniform(0.35, 1.65)),
                length=float(rng.uniform(0.15, 0.38)),
                angle_deg=float(rng.uniform(-25.0, 25.0)),
            )
            try:
                crack.validate(min_top_clearance=min_top_clearance)
                return crack
            except ValueError:
                pass
        raise RuntimeError("failed to sample a valid edge crack")


def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float(np.cross(b - a, c - a))


def proper_segment_intersection(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12
) -> bool:
    """True only for an interior/interior segment intersection."""
    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    # Orientation has units of length squared.  Comparing the product o1*o2
    # with one fixed epsilon has units of length to the fourth and misses real
    # crossings when adaptive sizing creates short edges.  Test the signs of
    # each orientation separately with a domain-scaled area tolerance instead.
    coordinate_scale = max(
        float(np.max(np.abs(np.concatenate((a, b, c, d))))), 1.0,
    )
    tolerance = float(eps) * coordinate_scale * coordinate_scale
    first_opposite = (o1 > tolerance and o2 < -tolerance) or (
        o1 < -tolerance and o2 > tolerance
    )
    second_opposite = (o3 > tolerance and o4 < -tolerance) or (
        o3 < -tolerance and o4 > tolerance
    )
    return bool(first_opposite and second_opposite)


def collinear_interior_overlap(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-10
) -> bool:
    """True when collinear segments overlap by more than a shared endpoint."""
    if abs(orientation(a, b, c)) > eps or abs(orientation(a, b, d)) > eps:
        return False
    axis = int(np.argmax(np.abs(b - a)))
    left = max(min(a[axis], b[axis]), min(c[axis], d[axis]))
    right = min(max(a[axis], b[axis]), max(c[axis], d[axis]))
    return bool(right - left > eps)


def edge_crosses_crack(a: np.ndarray, b: np.ndarray, crack: EdgeCrack) -> bool:
    # Crack nodes are stored as float32 while parameters are float64. Treat an
    # endpoint lying on the crack within meshing tolerance as an allowed touch,
    # not as an interior crossing caused by round-off.
    direction = crack.tip - crack.start
    length2 = float(direction @ direction)
    for point in (a, b):
        projection = float((point - crack.start) @ direction / length2)
        distance = abs(orientation(crack.start, crack.tip, point)) / np.sqrt(length2)
        if distance <= 1e-6 and -1e-6 <= projection <= 1.0 + 1e-6:
            return False
    return proper_segment_intersection(a, b, crack.start, crack.tip)


def point_in_plate(p: np.ndarray, crack: EdgeCrack, eps: float = 1e-9) -> bool:
    return -eps <= p[0] <= crack.width + eps and -eps <= p[1] <= crack.height + eps
