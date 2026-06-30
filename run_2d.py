"""
Entry point for the 2D problem.

    python run_2d.py --target 0.15                      # superellipse family
    python run_2d.py --target 0.30 --family fourier     # complex (lobed/spiky) shapes
    python run_2d.py --target 0.30 --family fourier --modes 5 --plot

Two shape families:
  * superellipse (default): 2 genes, convex, fast. Void band ~0.02-0.22 here.
  * fourier: 2*modes genes, lobed/spiky/concave, slower (compound bodies).
    Reaches HIGHER void (~0.33+) because complex shapes leave more empty space.

Pipeline: probe the achievable band -> (warn/clamp if target is outside it)
-> run the GA -> verify the winner with extra drops -> report the shape.
Add --plot to save the evolved outline and one settled packing to result.png.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import numpy as np

from packing_ga.problem import (make_problem_2d, corner_genomes_2d,
                                 make_problem_2d_fourier, corner_genomes_fourier)
from packing_ga.ga import RealCodedGA, GAConfig
from packing_ga.shapes import Superellipse, FourierShape

L = 12.0  # length scale (shape area = L**2); void fraction is scale-invariant


def describe_superellipse(genome) -> str:
    aspect, n = genome
    if n < 1.3:
        fam = "diamond-like (near-tiling)"
    elif n < 1.8:
        fam = "rounded-diamond"
    elif n < 2.4:
        fam = "ellipse/disk-like"
    else:
        fam = "rounded-rectangle/box-like"
    elong = "circular" if aspect < 1.15 else f"elongated {aspect:.1f}:1"
    return f"{fam}, {elong}"


def describe_fourier(shape: FourierShape) -> str:
    k = shape.dominant_mode()
    lobe = {1: "egg/asymmetric", 2: "oval/peanut", 3: "triangular (3-lobe)",
            4: "square-ish (4-lobe)"}.get(k, f"{k}-lobed / gear-like")
    return f"{lobe}, {'convex' if shape.is_convex() else 'concave/lobed'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.10,
                    help="target void fraction (empty/total), in [0,1]")
    ap.add_argument("--family", choices=["superellipse", "fourier"],
                    default="superellipse")
    ap.add_argument("--modes", type=int, default=4,
                    help="(fourier) number of angular harmonics = shape complexity")
    ap.add_argument("--amp", type=float, default=0.6,
                    help="(fourier) per-mode amplitude bound")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gen", type=int, default=8)
    ap.add_argument("--drops", type=int, default=2,
                    help="independent settles averaged per fitness evaluation")
    ap.add_argument("--n-shapes", type=int, default=0,
                    help="shapes per settle (0 = family default)")
    ap.add_argument("--tol", type=float, default=0.015)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    # -- assemble the problem, range-probe corners, and a shape builder ------- #
    if args.family == "superellipse":
        n_shapes = args.n_shapes or 120
        prob = make_problem_2d(target_void=args.target, n_shapes=n_shapes,
                               drops_per_eval=args.drops, seed=args.seed)
        corners = corner_genomes_2d()
        build = lambda g: Superellipse.from_genome(g, scale=L)
        descr = lambda g: describe_superellipse(g)
        band_hint = ("To reach higher void, use complex shapes: "
                     "--family fourier (optionally raise --amp / --modes).")
    else:
        n_shapes = args.n_shapes or 90
        prob = make_problem_2d_fourier(target_void=args.target, n_modes=args.modes,
                                       amp=args.amp, n_shapes=n_shapes,
                                       drops_per_eval=args.drops, seed=args.seed)
        corners = corner_genomes_fourier(args.modes, args.amp)
        build = lambda g: FourierShape.from_genome(g, scale=L)
        descr = lambda g: describe_fourier(build(g))
        band_hint = ("To reach higher void, raise --amp or --modes "
                     "(spikier shapes leave more empty space).")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(os.path.dirname(__file__), "experiments")
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Family: {args.family}   target void fraction: {args.target:.3f}")
    print("Probing achievable band...")
    lo, hi = prob.probe_range(corners, drops=2)
    print(f"  reachable void in this configuration: ~[{lo:.3f}, {hi:.3f}]")

    margin = 0.01
    target = args.target
    if target < lo - margin or target > hi + margin:
        target = float(np.clip(target, lo, hi))
        print(f"  ! target outside reachable band; clamping to {target:.3f}.")
        print("    " + band_hint)
        prob.target_void = target

    print("\nRunning genetic algorithm...")
    cfg = GAConfig(bounds=prob.bounds, pop_size=args.pop, n_gen=args.gen,
                   tol=args.tol, seed=args.seed)
    ga = RealCodedGA(prob.objective, cfg)
    best = ga.run(verbose=True)

    achieved = prob.void_of(best.genome, drops=6)
    print("\n=== Result ===")
    g = ", ".join(f"{v:.3f}" for v in best.genome)
    print(f"  evolved genome : [{g}]")
    print(f"  shape          : {descr(best.genome)}")
    print(f"  target void    : {target:.3f}")
    print(f"  achieved void  : {achieved:.3f}  (|error| = {abs(achieved-target):.3f})")

    log = {
        "timestamp": ts,
        "params": vars(args),
        "probed_band": {"lo": round(lo, 4), "hi": round(hi, 4)},
        "clamped_target": round(target, 4),
        "result": {
            "genome": [round(v, 4) for v in best.genome.tolist()],
            "shape": descr(best.genome),
            "target_void": round(target, 4),
            "achieved_void": round(achieved, 4),
            "error": round(abs(achieved - target), 4),
        },
    }
    log_path = os.path.join(exp_dir, f"{ts}.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  log -> {log_path}")

    if args.plot:
        plot_path = os.path.join(exp_dir, f"{ts}_result.png")
        _plot(build, best.genome, args.seed, n_shapes, plot_path)


def _plot(build, genome, seed, n_shapes, out_path="result.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon as MplPoly
    except ImportError:
        print("  (matplotlib not installed; skipping --plot)")
        return
    from packing_ga.physics import Settler2D, Container2D, default_window_2d

    shape = build(genome)
    cont = Container2D(width=16 * L, length_scale=L, n_shapes=n_shapes)
    res = Settler2D(cont).settle(shape, np.random.default_rng(seed))
    win = default_window_2d(res)
    v = shape.verts

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 5))
    axA.fill(v[:, 0], v[:, 1], alpha=0.4)
    axA.plot(np.r_[v[:, 0], v[0, 0]], np.r_[v[:, 1], v[0, 1]])
    axA.set_aspect("equal"); axA.set_title("Evolved shape")
    for x, y, ang in res.placements:
        c, s = np.cos(ang), np.sin(ang)
        w = (v @ np.array([[c, -s], [s, c]]).T) + np.array([x, y])
        axB.add_patch(MplPoly(w, closed=True, alpha=0.6, lw=0.3, edgecolor="k"))
    axB.add_patch(plt.Rectangle((win.xmin, win.ymin), win.xmax - win.xmin,
                                win.ymax - win.ymin, fill=False, ec="red", lw=2))
    axB.set_xlim(-9 * L, 9 * L); axB.set_ylim(0, res.surface_y + 2 * L)
    axB.set_aspect("equal"); axB.set_title("Settled packing (red = measurement window)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"  saved plot -> {out_path}")


if __name__ == "__main__":
    main()
