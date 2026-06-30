"""
Naturalistic settling = the fitness "measurement device".

A Settler drops N copies of one shape from random positions into a fixed
container and lets them fall under gravity until they come to rest. This is what
makes the packing *naturalistic* (your definition): the arrangement emerges from
physics, not from an optimiser.

Settler2D is implemented with pymunk (Chipmunk2D) and tested.
Settler3D sketches the identical contract on pybullet; it is written but marked
UNTESTED -- it is the "3D framework to use later".

The GA never touches this module directly; it goes through PackingProblem
(problem.py), which calls settle(...) then the matching measurer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pymunk

from .shapes import Superellipse, Superquadric
from .measure import Window2D, Window3D


# --------------------------------------------------------------------------- #
# Shared container / settling configuration
# --------------------------------------------------------------------------- #
@dataclass
class Container2D:
    width: float                 # interior width
    length_scale: float          # shape size L (shape area = L**2)
    floor_y: float = 0.0
    wall_friction: float = 0.6
    shape_friction: float = 0.6
    elasticity: float = 0.0      # inelastic -> settles instead of bouncing forever
    gravity: float = -900.0

    # Settling controls
    n_shapes: int = 120
    wave_size: int = 18          # bodies released per wave
    steps_between_waves: int = 25
    dt: float = 1.0 / 60.0
    solver_iterations: int = 25
    settle_speed_eps: float = 8.0   # px/s; pile is "at rest" below this
    max_settle_steps: int = 1500


@dataclass
class SettleResult2D:
    placements: np.ndarray       # (N, 3): x, y, angle
    surface_y: float             # 90th-percentile top of pile
    container: Container2D


class Settler2D:
    """Drops superellipses into a 2D box and returns their resting poses."""

    def __init__(self, container: Container2D):
        self.c = container

    def settle(self, shape: Superellipse,
               rng: np.random.Generator) -> SettleResult2D:
        c = self.c
        space = pymunk.Space()
        space.gravity = (0.0, c.gravity)
        space.iterations = c.solver_iterations
        space.damping = 0.92  # bleeds residual jitter so cornered shapes settle

        self._add_static_walls(space)
        # Convex pieces in the shape's local frame; one rigid body is built from
        # all of them. A convex shape yields one piece -> single Poly (fast path);
        # a Fourier shape yields a fan of triangles -> compound body.
        piece_verts = [[tuple(v) for v in p] for p in shape.pieces()]
        br = shape.bounding_radius()

        bodies: list[pymunk.Body] = []
        spawn_x_lo = c.floor_y_xmin() + br
        spawn_x_hi = c.floor_y_xmax() - br
        spawn_y = c.floor_y + 4.0 * br  # first wave height above floor

        n_left = c.n_shapes
        while n_left > 0:
            k = min(c.wave_size, n_left)
            top = self._current_top(bodies, default=c.floor_y) + 6.0 * br
            for _ in range(k):
                body = pymunk.Body()
                body.position = (rng.uniform(spawn_x_lo, spawn_x_hi),
                                 max(spawn_y, top) + rng.uniform(0, 2 * br))
                body.angle = rng.uniform(0, 2 * np.pi)
                shapes = []
                for pv in piece_verts:
                    poly = pymunk.Poly(body, pv, radius=0.3)
                    poly.density = 1.0
                    poly.friction = c.shape_friction
                    poly.elasticity = c.elasticity
                    shapes.append(poly)
                space.add(body, *shapes)
                bodies.append(body)
            n_left -= k
            for _ in range(c.steps_between_waves):
                space.step(c.dt)

        # Let everything come to rest.
        for _ in range(c.max_settle_steps):
            space.step(c.dt)
            if self._max_speed(bodies) < c.settle_speed_eps:
                break

        placements = np.array([[b.position.x, b.position.y, b.angle]
                               for b in bodies])
        tops = placements[:, 1] + br
        surface_y = float(np.percentile(tops, 90))
        return SettleResult2D(placements, surface_y, c)

    # -- internals ---------------------------------------------------------- #
    def _add_static_walls(self, space: pymunk.Space):
        c = self.c
        x0, x1 = c.floor_y_xmin(), c.floor_y_xmax()
        y = c.floor_y
        static = space.static_body
        segs = [
            pymunk.Segment(static, (x0, y), (x1, y), 1.0),           # floor
            pymunk.Segment(static, (x0, y), (x0, y + 1e4), 1.0),     # left wall
            pymunk.Segment(static, (x1, y), (x1, y + 1e4), 1.0),     # right wall
        ]
        for s in segs:
            s.friction = c.wall_friction
            s.elasticity = c.elasticity
            space.add(s)

    @staticmethod
    def _max_speed(bodies) -> float:
        if not bodies:
            return 0.0
        return max(b.velocity.length for b in bodies)

    @staticmethod
    def _current_top(bodies, default: float) -> float:
        if not bodies:
            return default
        return max(b.position.y for b in bodies)


# Convenience geometry on the container (kept off the dataclass body for clarity)
def _xmin(self: Container2D) -> float:
    return -self.width / 2.0


def _xmax(self: Container2D) -> float:
    return self.width / 2.0


Container2D.floor_y_xmin = _xmin
Container2D.floor_y_xmax = _xmax


def default_window_2d(result: SettleResult2D,
                      wall_margin_L: float = 1.5,
                      bottom_margin_L: float = 1.0,
                      top_margin_L: float = 1.5) -> Window2D:
    """Inset measurement window: away from walls, off the floor, below surface."""
    c = result.container
    L = c.length_scale
    return Window2D(
        xmin=c.floor_y_xmin() + wall_margin_L * L,
        xmax=c.floor_y_xmax() - wall_margin_L * L,
        ymin=c.floor_y + bottom_margin_L * L,
        ymax=result.surface_y - top_margin_L * L,
    )


# --------------------------------------------------------------------------- #
# 3D framework (pybullet). UNTESTED -- provided as the "use later" scaffold.
# Contract mirrors Settler2D exactly so problem.py is dimension-agnostic.
# --------------------------------------------------------------------------- #
@dataclass
class Container3D:
    width: float          # x and y extent of the box base
    length_scale: float
    floor_z: float = 0.0
    n_shapes: int = 400
    wave_size: int = 40
    steps_between_waves: int = 60
    gravity: float = -9.81
    sim_hz: float = 240.0
    friction: float = 0.6
    settle_speed_eps: float = 0.02
    max_settle_steps: int = 6000
    mesh_res: int = 28


@dataclass
class SettleResult3D:
    placements: np.ndarray   # (N, 7): x,y,z, qx,qy,qz,qw
    surface_z: float
    container: Container3D


class Settler3D:
    """
    pybullet settling for superquadrics. NOT YET TESTED.

    Implementation notes (everything needed to finish it):
      * `pip install pybullet`, then `import pybullet as p; p.connect(p.DIRECT)`.
      * Build ONE convex collision shape from shape.mesh() via
        p.createCollisionShape(p.GEOM_MESH, vertices=verts.tolist()); pybullet
        takes the convex hull, which is exact for convex superquadrics (e<=1).
        For concave (e>1) run V-HACD (p.vhacd) first.
      * Container = 5 static planes (floor + 4 walls) via GEOM_PLANE / thin boxes.
      * Drop in waves (same loop shape as 2D), stepping p.stepSimulation().
      * Rest test: max linear velocity from p.getBaseVelocity over all bodies.
      * Read poses with p.getBasePositionAndOrientation -> (xyz, quaternion).
      * Pair with packing_fraction_3d() and default_window_3d() below.
    """

    def __init__(self, container: Container3D):
        self.c = container

    def settle(self, shape: Superquadric,
               rng: np.random.Generator) -> SettleResult3D:
        raise NotImplementedError(
            "Settler3D is the untested 3D scaffold. See the docstring for the "
            "step-by-step pybullet implementation; the rest of the pipeline "
            "(Superquadric geometry, packing_fraction_3d, GA) is ready for it."
        )


def default_window_3d(result: "SettleResult3D",
                      wall_margin_L: float = 1.5,
                      bottom_margin_L: float = 1.0,
                      top_margin_L: float = 1.5) -> Window3D:
    c = result.container
    L = c.length_scale
    half = c.width / 2.0
    lo = np.array([-half + wall_margin_L * L,
                   -half + wall_margin_L * L,
                   c.floor_z + bottom_margin_L * L])
    hi = np.array([half - wall_margin_L * L,
                   half - wall_margin_L * L,
                   result.surface_z - top_margin_L * L])
    return Window3D(lo=lo, hi=hi)
