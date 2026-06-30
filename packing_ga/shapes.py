"""
Shape genomes for the packing-ratio GA.

A "shape" is the thing the genetic algorithm evolves. The GA only ever sees a
flat vector of floats (the genome); each Shape subclass knows how to turn a
genome into actual geometry that the physics backend can drop and the measurer
can score.

Design choice (v1): the default 2D family is the SUPERELLIPSE and the default
3D family is the SUPERQUADRIC. Rationale:
  * Tiny genome (2D: 2 numbers, 3D: 4 numbers) -> the GA converges in few
    *physics evaluations*, which are the expensive part.
  * Direct 2D<->3D analogue, so the framework generalises cleanly.
  * Exponent >= 1 guarantees a CONVEX shape, which keeps the physics simple
    (no convex decomposition needed). Exponent < 1 gives concave/star shapes;
    the polygoniser handles them, but v1 keeps the GA bounds at >=1 (see
    problem.py). Concave shapes are the documented route to higher void
    fractions.

To add a richer family later (polar polygon in 2D, spherical-harmonic blob in
3D), implement the Shape interface and hand its `from_genome` + `bounds` to a
PackingProblem. Nothing else changes.

Size is normalised away: every 2D shape is rescaled to unit area, every 3D
shape to unit volume, then multiplied by a length scale L. The void fraction is
scale-invariant, so L exists only for numerical comfort in the physics engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from shapely.geometry import Polygon


# --------------------------------------------------------------------------- #
# Abstract interface
# --------------------------------------------------------------------------- #
class Shape:
    """Interface every evolvable shape family must implement."""

    dim: int  # 2 or 3

    @classmethod
    def from_genome(cls, genome: np.ndarray, scale: float = 1.0) -> "Shape":
        raise NotImplementedError

    @staticmethod
    def genome_length() -> int:
        raise NotImplementedError

    def bounding_radius(self) -> float:
        """Radius of the smallest origin-centred circle/sphere enclosing it."""
        raise NotImplementedError

    def pieces(self) -> list[np.ndarray]:
        """
        Convex decomposition for the physics engine, in the shape's local
        (centroid-centred) frame. A convex shape returns one piece (itself);
        a non-convex shape returns several convex polygons whose union is the
        shape. The physics backend builds one rigid body from these pieces.
        """
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# 2D: superellipse   |x/a|^n + |y/b|^n = 1
# --------------------------------------------------------------------------- #
@dataclass
class Superellipse(Shape):
    """
    Genome = [aspect, n]
        aspect : b/a   (>= 1 by convention; orientation is randomised at drop
                        time, so aspect < 1 is just a rotation of aspect > 1)
        n      : exponent. n=2 ellipse, n->inf rectangle, n=1 diamond,
                 n<1 concave 4-pointed star.

    Stored as a closed CCW vertex ring, already normalised to area = scale**2.
    """

    dim = 2
    aspect: float
    n: float
    scale: float
    verts: np.ndarray  # (M, 2), CCW, no repeated last point

    M_DEFAULT = 28  # polygon vertices; convex superellipse stays convex

    @staticmethod
    def genome_length() -> int:
        return 2

    @classmethod
    def from_genome(cls, genome: np.ndarray, scale: float = 1.0,
                    m: int = M_DEFAULT) -> "Superellipse":
        aspect, n = float(genome[0]), float(genome[1])
        a, b = 1.0, aspect
        t = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        # Superellipse parametric form (sign-preserving fractional powers).
        x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** (2.0 / n)
        y = b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** (2.0 / n)
        verts = np.column_stack([x, y])
        # Normalise to unit area, then apply length scale.
        area = Polygon(verts).area
        verts *= scale / np.sqrt(area)
        return cls(aspect=aspect, n=n, scale=scale, verts=verts)

    def polygon(self) -> Polygon:
        return Polygon(self.verts)

    def bounding_radius(self) -> float:
        return float(np.max(np.linalg.norm(self.verts, axis=1)))

    def is_convex(self) -> bool:
        return self.n >= 1.0

    def pieces(self) -> list[np.ndarray]:
        # Convex (n>=1): the whole polygon is a single convex piece.
        return [self.verts]


# --------------------------------------------------------------------------- #
# 2D: Fourier-boundary radial shape (the "complex" family)
#     r(theta) = 1 + sum_k [a_k cos(k theta) + b_k sin(k theta)]
# --------------------------------------------------------------------------- #
@dataclass
class FourierShape(Shape):
    """
    Genome = flat [a1, b1, a2, b2, ..., aK, bK]   (K = number of modes).

    The boundary radius is a truncated Fourier series in the polar angle. This
    spans far more shapes than the superellipse -- ellipse/peanut (mode 2),
    triangular (3), square-ish (4), gear/star (high modes), and asymmetric
    blobs (mixed modes). Complexity is set by K.

    Validity is automatic: a radial curve with r(theta) > 0 is always a simple,
    non-self-intersecting polygon that is STAR-SHAPED from its centre. If a
    genome would drive r(theta) <= 0, the AC part is scaled down so the minimum
    radius equals r_min (a "repair" step) -- so every genome maps to a valid
    shape and the GA never wastes evaluations on broken geometry.

    Because the shape is star-shaped from the radial centre, it decomposes into
    convex pieces for free: a fan of triangles (centre, v_i, v_{i+1}). No
    general convex-decomposition library is needed.

    Geometry is normalised to area = scale**2 and stored centred on its area
    centroid, so it rotates naturally and matches the measurement code.
    """

    dim = 2
    coeffs: np.ndarray   # (K, 2): rows are (a_k, b_k) for k = 1..K
    scale: float
    verts: np.ndarray    # (M, 2) CCW boundary, centred on the area centroid
    apex: np.ndarray     # (2,) radial centre in the centred frame (fan apex)

    M_DEFAULT = 24       # boundary samples (>> 2K for a faithful curve)

    @staticmethod
    def genome_length(n_modes: int) -> int:  # type: ignore[override]
        return 2 * n_modes

    @classmethod
    def from_genome(cls, genome: np.ndarray, scale: float = 1.0,
                    m: int = M_DEFAULT, r_min: float = 0.15) -> "FourierShape":
        g = np.asarray(genome, dtype=float)
        K = len(g) // 2
        coeffs = g[:2 * K].reshape(K, 2)
        theta = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
        ks = np.arange(1, K + 1)[:, None]                       # (K, 1)
        ac = (coeffs[:, 0:1] * np.cos(ks * theta) +
              coeffs[:, 1:2] * np.sin(ks * theta)).sum(axis=0)  # (M,)
        mn = (1.0 + ac).min()
        if mn < r_min:                                          # repair
            ac *= (1.0 - r_min) / (1.0 - mn)
        r = 1.0 + ac
        pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])

        area = Polygon(pts).area
        pts *= scale / np.sqrt(area)                            # unit-area * scale
        centroid = np.asarray(Polygon(pts).centroid.coords[0])
        verts = pts - centroid                                  # centre on centroid
        apex = -centroid                                        # radial centre, centred
        return cls(coeffs=coeffs, scale=scale, verts=verts, apex=apex)

    def polygon(self) -> Polygon:
        return Polygon(self.verts)

    def bounding_radius(self) -> float:
        return float(np.max(np.linalg.norm(self.verts, axis=1)))

    def is_convex(self) -> bool:
        p = self.polygon()
        return p.equals(p.convex_hull)

    def dominant_mode(self) -> int:
        mags = np.linalg.norm(self.coeffs, axis=1)
        return int(np.argmax(mags) + 1) if mags.size else 0

    def pieces(self) -> list[np.ndarray]:
        # Fan triangulation from the radial centre (valid because star-shaped).
        v, a, m = self.verts, self.apex, len(self.verts)
        out = []
        for i in range(m):
            tri = np.array([a, v[i], v[(i + 1) % m]])
            if _signed_area(tri) < 0:        # ensure CCW for the physics engine
                tri = tri[::-1]
            if abs(_signed_area(tri)) > 1e-9:
                out.append(tri)
        return out


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


# --------------------------------------------------------------------------- #
# 3D: superquadric (geometry + analytic inside-test; testable without a
# physics engine). Used by the 3D framework in physics.py / measure.py.
# --------------------------------------------------------------------------- #
@dataclass
class Superquadric(Shape):
    """
    Genome = [b_over_a, c_over_a, e1, e2]
        b_over_a, c_over_a : axis ratios (semi-axes b/a, c/a; a fixed = 1)
        e1, e2             : squareness exponents.
                             e1=e2=1 -> ellipsoid; e<1 -> boxy; e>1 -> pinched.

    Inside test (in the shape's own local frame, semi-axes a,b,c):
        F = (|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1)
        inside  <=>  F <= 1
    Semi-axes are scaled so the volume = scale**3.
    """

    dim = 3
    a: float
    b: float
    c: float
    e1: float
    e2: float

    @staticmethod
    def genome_length() -> int:
        return 4

    @classmethod
    def from_genome(cls, genome: np.ndarray, scale: float = 1.0,
                    res: int = 40) -> "Superquadric":
        b_over_a, c_over_a, e1, e2 = (float(g) for g in genome)
        a, b, c = 1.0, b_over_a, c_over_a
        # Normalise volume via the mesh, then rescale semi-axes.
        verts, _ = _superquadric_mesh(a, b, c, e1, e2, res)
        vol = _mesh_volume(*_superquadric_mesh(a, b, c, e1, e2, res))
        k = scale / (vol ** (1.0 / 3.0))
        return cls(a=a * k, b=b * k, c=c * k, e1=e1, e2=e2)

    def inside(self, pts_local: np.ndarray) -> np.ndarray:
        """Vectorised inside test. pts_local: (N,3) in the shape's local frame."""
        x = np.abs(pts_local[:, 0] / self.a)
        y = np.abs(pts_local[:, 1] / self.b)
        z = np.abs(pts_local[:, 2] / self.c)
        f = (x ** (2.0 / self.e2) + y ** (2.0 / self.e2)) ** (self.e2 / self.e1) \
            + z ** (2.0 / self.e1)
        return f <= 1.0

    def mesh(self, res: int = 40):
        """Triangulated surface (verts, faces) for a physics collision shape."""
        return _superquadric_mesh(self.a, self.b, self.c, self.e1, self.e2, res)

    def bounding_radius(self) -> float:
        # Convex superquadric (e<=1): vertices lie within max semi-axis * sqrt(3)
        # in the worst (box) case; use the mesh extent for accuracy.
        v, _ = self.mesh(res=24)
        return float(np.max(np.linalg.norm(v, axis=1)))


def _superquadric_mesh(a, b, c, e1, e2, res):
    """Parametric superquadric surface -> (verts (K,3), faces (T,3))."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, res)        # latitude
    omega = np.linspace(-np.pi, np.pi, res)              # longitude
    E, O = np.meshgrid(eta, omega, indexing="ij")

    def sgnpow(v, p):
        return np.sign(v) * np.abs(v) ** p

    x = a * sgnpow(np.cos(E), e1) * sgnpow(np.cos(O), e2)
    y = b * sgnpow(np.cos(E), e1) * sgnpow(np.sin(O), e2)
    z = c * sgnpow(np.sin(E), e1)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    faces = []
    R = res
    for i in range(R - 1):
        for j in range(R - 1):
            p0 = i * R + j
            p1 = p0 + 1
            p2 = p0 + R
            p3 = p2 + 1
            faces.append([p0, p2, p1])
            faces.append([p1, p2, p3])
    return verts, np.asarray(faces, dtype=np.int64)


def _mesh_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume via the divergence theorem (sum of tetrahedra)."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    vol = np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0
    return abs(float(vol))
