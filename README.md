# packing_ga — evolving shapes to hit a target packing ratio

A genetic algorithm that searches for **shapes** whose *naturalistic* packing
(objects dropped from random positions and settled under gravity) has a chosen
**void fraction** ("packing ratio" = empty space / total space).

2D is implemented and tested. A 3D framework shares the same interfaces and is
ready except for one step (see *Going 3D*).

## Terminology note
Your "packing ratio" (empty / total) is what the literature usually calls
**void fraction** or **porosity**; its complement (filled / total) is the
**packing fraction** or **density**. The code computes packing fraction and
reports `1 - that` as your ratio. (Flagging so results line up with references.)

## How it works
```
genome ──Shape.from_genome──▶ shape geometry
       ──Settler.settle─────▶ resting positions   (naturalistic packing)
       ──packing_fraction───▶ filled fraction
       1 − fraction ────────▶ void fraction
       |void − target| ─────▶ objective the GA minimises
```
The GA only ever sees a flat genome and a scalar objective, so it drives 2D and
3D unchanged.

## Quick start
```bash
pip install pymunk shapely numpy matplotlib
python run_2d.py --target 0.15                    # superellipse family (simple, fast)
python run_2d.py --target 0.32 --family fourier   # complex concave/lobed shapes
python run_2d.py --target 0.35 --family fourier --modes 5 --amp 0.8 --plot
python run_2d.py --target 0.60                    # outside band -> warns + clamps
```
Useful flags: `--family {superellipse,fourier} --modes --amp --pop --gen --drops
--n-shapes --tol --seed --plot`.

## Two shape families
| family | genome | shapes it spans | void band here | speed |
|--------|--------|-----------------|----------------|-------|
| `superellipse` (default) | 2 genes (aspect, exponent) | convex: disk→square→diamond→ellipse | ~0.02–0.22 | fast (~0.1–1 s/drop) |
| `fourier` | 2·`--modes` genes | concave/lobed/spiky via a Fourier-series boundary radius | ~0.23–0.41 | slower (~1.3 s/drop; compound bodies) |

Use `superellipse` for low/moderate void and quick runs; use `fourier` when you
need higher void (concave shapes interlock poorly → more empty space) or want
the GA to search a genuinely rich landscape. Raise `--modes`/`--amp` for spikier
shapes and a higher reachable ceiling.

## Files
| file | role |
|------|------|
| `packing_ga/shapes.py`  | genomes: 2D **superellipse** + **FourierShape** (concave), 3D **superquadric** |
| `packing_ga/physics.py` | settling: `Settler2D` (pymunk, builds bodies from convex `pieces()`), `Settler3D` (pybullet, scaffold) |
| `packing_ga/measure.py` | void fraction: exact union (2D), Monte-Carlo (3D) |
| `packing_ga/ga.py`      | real-coded GA, engine-agnostic, with noise handling |
| `packing_ga/problem.py` | wires genome→shape→settle→measure→objective; range probe |
| `run_2d.py`             | CLI: probe band → GA → verify → optional plot |

## Design choices (and why)
- **Two genomes.** `superellipse` (2 convex genes) for speed; `FourierShape`
  (Fourier-series boundary radius) for concave/lobed shapes and higher void. A
  Fourier radial curve with r(θ)>0 is always a simple **star-shaped** polygon,
  so it decomposes into convex pieces for free — a fan of triangles from the
  radial centre — which is what pymunk needs (no decomposition library). Add a
  further family by implementing the `Shape` interface (`from_genome` + `pieces`).
- **Genome → valid shape always.** If a Fourier genome would drive r(θ)≤0, the
  amplitudes are scaled so the minimum radius = `r_min` (a repair step), so the
  GA never wastes evaluations on broken geometry.
- **Measurement window**, inset from walls and below the uneven free surface,
  so the number reflects bulk packing, not boundary artefacts.
- **Union-based area (2D)** so any solver interpenetration is never
  double-counted (void fraction stays in [0,1]).

## Things to know (they affect how you read the output)
1. **Fitness is noisy.** Naturalistic packing is stochastic, so each shape gives
   a slightly different void fraction per drop (≈ ±0.01 here). The code averages
   `--drops` settles per evaluation and re-evaluates elites each generation so a
   lucky drop can't entrench a genome. Keep `--tol` above the noise floor.
2. **The reachable band is bounded — and family-dependent.** Convex
   (`superellipse`) reaches ~0.02–0.22 here (diamonds nearly tile → ~0.02;
   elongated ellipses leave the most gaps → ~0.22). Concave (`fourier`) reaches
   ~0.23–0.41 and climbs further with higher `--amp`/`--modes`. The runner probes
   the band for the chosen family and clamps + warns if the target is outside it.
3. **Target → shape is one-to-many.** Many shapes hit the same void fraction;
   the GA returns *a* solution, not *the* solution. Different seeds give
   different valid shapes.
4. **`superellipse` (2 genes) often converges in generation 0** — the landscape
   is that simple. `fourier` (2·modes genes) is a genuinely rich landscape where
   the GA iterates across generations and earns its keep.

## Going 3D
Ready and tested: superquadric geometry (volume-normalised), the analytic
inside-test, and Monte-Carlo void measurement (validated against known cases).
Only remaining step: implement `Settler3D.settle` on pybullet — its docstring
gives the exact recipe (convex-hull collision shapes from the mesh, plane
container, wave dropping, velocity-based rest test, read back pose quaternions).
Then `make_problem_3d(...)` runs the identical GA over a 4-gene genome.

## Tested in this environment
pymunk 7.3, shapely 2.1, numpy 2.4. Example results (gravity-settled, 120
shapes, friction 0.6): disk void ≈ 0.22, square ≈ 0.17, diamond ≈ 0.02. These
are configuration-dependent (friction, restitution, drop pattern); the GA
matches a target *within* whatever configuration you set.
