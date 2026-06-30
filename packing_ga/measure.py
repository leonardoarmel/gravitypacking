"""
Void-fraction measurement.

Your "packing ratio" = empty / total within a measurement window. That is what
the literature calls void fraction / porosity; its complement is the packing
fraction / density. We compute the packing fraction here and the caller does
1 - that to get your ratio (problem.py).

Why a *window* rather than the whole container: a naturalistic (gravity-settled)
pile has two regions that bias the count -- the uneven free surface at the top
and the ordered/gappy layers against the walls. We measure inside an inset
window that excludes both, so the number reflects the bulk packing.

2D: exact. Shapes don't overlap in a rigid settle, so covered area is just the
    sum of (shape polygon  ∩  window) areas (shapely).
3D: Monte-Carlo with the analytic superquadric inside-test -- no mesh needed.
    Sample points uniformly in the window; packing fraction = fraction of points
    inside any body. Same signature, so the GA/problem code is dimension-blind.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

from .shapes import Superellipse, Superquadric


@dataclass
class Window2D:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def polygon(self) -> Polygon:
        return Polygon([(self.xmin, self.ymin), (self.xmax, self.ymin),
                        (self.xmax, self.ymax), (self.xmin, self.ymax)])


@dataclass
class Window3D:
    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)

    @property
    def volume(self) -> float:
        return float(np.prod(self.hi - self.lo))


def packing_fraction_2d(shape: Superellipse,
                        placements: np.ndarray,
                        window: Window2D) -> float:
    """
    placements: (N, 3) array of (x, y, angle_radians) for each settled body.
    Returns packing fraction (covered / window area). Caller does 1 - this.
    """
    from shapely.ops import unary_union
    win = window.polygon()
    base = shape.polygon()
    pieces = []
    for x, y, ang in placements:
        poly = translate(rotate(base, ang, origin=(0, 0), use_radians=True),
                         xoff=x, yoff=y)
        if poly.intersects(win):
            pieces.append(poly.intersection(win))
    if not pieces:
        return 0.0
    # Union (not sum) so any solver interpenetration is not double-counted;
    # covered area can never exceed the window -> void fraction stays in [0,1].
    covered = unary_union(pieces).area
    return covered / window.area


def packing_fraction_3d(shape: Superquadric,
                        placements: np.ndarray,
                        window: Window3D,
                        n_samples: int = 200_000,
                        rng: np.random.Generator | None = None) -> float:
    """
    placements: (N, 7) array of (x,y,z, qx,qy,qz,qw) -- position + quaternion.
    Monte-Carlo packing fraction. Sampling noise ~ sqrt(p(1-p)/n_samples);
    200k samples -> ~0.001 std, which is below typical run-to-run physics noise.
    """
    rng = rng or np.random.default_rng()
    pts = rng.uniform(window.lo, window.hi, size=(n_samples, 3))
    inside_any = np.zeros(n_samples, dtype=bool)
    br = shape.bounding_radius()
    for x, y, z, qx, qy, qz, qw in placements:
        centre = np.array([x, y, z])
        # Cheap reject: only test points within the bounding sphere.
        near = np.linalg.norm(pts - centre, axis=1) <= br
        if not near.any():
            continue
        R = _quat_to_matrix(qx, qy, qz, qw)
        local = (pts[near] - centre) @ R  # world->local = R^T applied as p@R
        hit = shape.inside(local)
        idx = np.where(near)[0][hit]
        inside_any[idx] = True
    return inside_any.mean()


def _quat_to_matrix(qx, qy, qz, qw) -> np.ndarray:
    """Rotation matrix (columns = local axes in world). p_local = (p_world-c) @ R."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])
