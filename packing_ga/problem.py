"""
PackingProblem ties the pieces together and is the ONLY object the GA needs:

    genome --(Shape.from_genome)--> shape
           --(Settler.settle)------> resting placements   [naturalistic packing]
           --(packing_fraction)----> packing fraction
           1 - fraction ----------> void fraction  ("packing ratio", your term)
           |void - target| -------> objective the GA minimises

Two ready configs:
  * make_problem_2d(...)  -> tested (pymunk + shapely, superellipse genome).
  * make_problem_3d(...)  -> wired to the untested Settler3D scaffold; the
    geometry (superquadric) and Monte-Carlo measurement are ready, only the
    pybullet settling step needs finishing (see physics.Settler3D docstring).

`probe_range()` settles a handful of corner genomes to estimate which void
fractions this configuration can actually reach -- because a naturalistic
packing has a bounded achievable band, and a target outside it cannot be hit by
any shape (the runner uses this to warn and clamp).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .shapes import Superellipse, Superquadric, FourierShape
from .physics import (Settler2D, Container2D, default_window_2d,
                      Settler3D, Container3D, default_window_3d)
from .measure import packing_fraction_2d, packing_fraction_3d


@dataclass
class PackingProblem:
    """Dimension-agnostic objective wrapper. Built via the factories below."""
    target_void: float
    bounds: list[tuple[float, float]]
    _eval_void: Callable[[np.ndarray, np.random.Generator], float]
    drops_per_eval: int = 3
    seed: int = 0
    _counter: int = 0

    def void_of(self, genome: np.ndarray, drops: int | None = None) -> float:
        """Mean void fraction over `drops` independent settles (the noisy signal)."""
        drops = drops or self.drops_per_eval
        vals = []
        for _ in range(drops):
            rng = np.random.default_rng(self.seed * 1_000_003 + self._counter)
            self._counter += 1
            vals.append(self._eval_void(genome, rng))
        return float(np.mean(vals))

    def objective(self, genome: np.ndarray) -> float:
        """Scalar the GA minimises: distance from the target void fraction."""
        return abs(self.void_of(genome) - self.target_void)

    def probe_range(self, corner_genomes: list[np.ndarray],
                    drops: int = 2) -> tuple[float, float]:
        vs = [self.void_of(g, drops=drops) for g in corner_genomes]
        return min(vs), max(vs)


# --------------------------------------------------------------------------- #
# 2D factory (tested)
# --------------------------------------------------------------------------- #
def make_problem_2d(target_void: float,
                    length_scale: float = 12.0,
                    container_width_L: float = 16.0,
                    n_shapes: int = 120,
                    aspect_bounds=(1.0, 6.0),
                    exponent_bounds=(1.0, 8.0),   # n>=1 keeps shapes convex
                    drops_per_eval: int = 3,
                    seed: int = 0) -> PackingProblem:
    container = Container2D(width=container_width_L * length_scale,
                            length_scale=length_scale, n_shapes=n_shapes)
    settler = Settler2D(container)

    def eval_void(genome: np.ndarray, rng: np.random.Generator) -> float:
        shape = Superellipse.from_genome(genome, scale=length_scale)
        res = settler.settle(shape, rng)
        win = default_window_2d(res)
        return 1.0 - packing_fraction_2d(shape, res.placements, win)

    return PackingProblem(target_void=target_void,
                          bounds=[aspect_bounds, exponent_bounds],
                          _eval_void=eval_void,
                          drops_per_eval=drops_per_eval, seed=seed)


def corner_genomes_2d() -> list[np.ndarray]:
    """Extremes of the 2D convex family, to bracket the achievable void band."""
    return [np.array([1.0, 2.0]),   # disk     (loosest convex packer here)
            np.array([1.0, 8.0]),   # square
            np.array([1.0, 1.0]),   # diamond  (near-tiling, densest)
            np.array([6.0, 2.0])]   # long ellipse


# --------------------------------------------------------------------------- #
# 2D factory -- Fourier "complex" family (tested)
# --------------------------------------------------------------------------- #
def make_problem_2d_fourier(target_void: float,
                            n_modes: int = 4,
                            amp: float = 0.6,
                            length_scale: float = 12.0,
                            container_width_L: float = 16.0,
                            n_shapes: int = 100,
                            drops_per_eval: int = 2,
                            seed: int = 0) -> PackingProblem:
    """
    Evolve Fourier-boundary shapes (genome = 2*n_modes amplitudes in [-amp, amp]).
    Richer than the superellipse family and reaches HIGHER void fractions,
    because lobed / spiky shapes interlock poorly and leave more empty space.
    """
    container = Container2D(width=container_width_L * length_scale,
                            length_scale=length_scale, n_shapes=n_shapes)
    settler = Settler2D(container)

    def eval_void(genome: np.ndarray, rng: np.random.Generator) -> float:
        shape = FourierShape.from_genome(genome, scale=length_scale)
        res = settler.settle(shape, rng)
        win = default_window_2d(res)
        return 1.0 - packing_fraction_2d(shape, res.placements, win)

    bounds = [(-amp, amp)] * (2 * n_modes)
    return PackingProblem(target_void=target_void, bounds=bounds,
                          _eval_void=eval_void,
                          drops_per_eval=drops_per_eval, seed=seed)


def corner_genomes_fourier(n_modes: int = 4, amp: float = 0.6) -> list[np.ndarray]:
    """Bracket the reachable band: round, single-mode lobes, and a spiky mix."""
    g = lambda: np.zeros(2 * n_modes)
    corners = [g()]                                   # circle (densest)
    for k in range(1, n_modes + 1):                   # one strong lobe per mode
        v = g(); v[2 * (k - 1)] = amp; corners.append(v)
    spiky = np.zeros(2 * n_modes)                     # several modes at once
    spiky[1::2] = amp * 0.6
    spiky[0::2] = amp * 0.6
    corners.append(spiky)
    return corners


# --------------------------------------------------------------------------- #
# 3D factory (uses the untested Settler3D scaffold)
# --------------------------------------------------------------------------- #
def make_problem_3d(target_void: float,
                    length_scale: float = 1.0,
                    container_width_L: float = 12.0,
                    n_shapes: int = 400,
                    ratio_bounds=(0.4, 1.0),
                    e_bounds=(0.2, 1.0),          # e<=1 keeps shapes convex
                    drops_per_eval: int = 1,
                    mc_samples: int = 200_000,
                    seed: int = 0) -> PackingProblem:
    container = Container3D(width=container_width_L * length_scale,
                            length_scale=length_scale, n_shapes=n_shapes)
    settler = Settler3D(container)

    def eval_void(genome: np.ndarray, rng: np.random.Generator) -> float:
        shape = Superquadric.from_genome(genome, scale=length_scale)
        res = settler.settle(shape, rng)            # NotImplementedError for now
        win = default_window_3d(res)
        pf = packing_fraction_3d(shape, res.placements, win,
                                 n_samples=mc_samples, rng=rng)
        return 1.0 - pf

    # genome = [b/a, c/a, e1, e2]
    return PackingProblem(target_void=target_void,
                          bounds=[ratio_bounds, ratio_bounds, e_bounds, e_bounds],
                          _eval_void=eval_void,
                          drops_per_eval=drops_per_eval, seed=seed)
