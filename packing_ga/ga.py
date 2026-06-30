"""
Real-coded genetic algorithm. Deliberately dimension-agnostic: it only sees a
scalar objective to MINIMISE and per-gene bounds. It never imports physics or
geometry, so the same GA drives the 2D and 3D problems unchanged.

Noise handling (naturalistic packing is stochastic, so fitness is noisy):
  * Each objective call already averages K independent drops (done in
    PackingProblem), which shrinks the per-evaluation noise.
  * Elites are RE-EVALUATED each generation and combined into a running mean,
    so a single lucky drop cannot keep a genome at the top forever.

Operators: tournament selection, BLX-alpha crossover, Gaussian mutation with
reflection at the bounds, elitism. These are standard, robust choices for
low-dimensional real-vector search; nothing here is exotic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class Individual:
    genome: np.ndarray
    score: float = np.inf      # running-mean objective (lower is better)
    n_evals: int = 0           # how many evaluations went into `score`

    def update(self, new_score: float):
        self.n_evals += 1
        # incremental running mean
        self.score += (new_score - self.score) / self.n_evals if self.n_evals > 1 \
            else (new_score - self.score)
        if self.n_evals == 1:
            self.score = new_score


@dataclass
class GAConfig:
    bounds: list[tuple[float, float]]
    pop_size: int = 12
    n_gen: int = 12
    n_elite: int = 2
    tournament_k: int = 3
    crossover_alpha: float = 0.4   # BLX-alpha
    mutation_frac: float = 0.18    # Gaussian sigma as fraction of each gene range
    mutation_prob: float = 0.7     # per-gene probability of mutation
    tol: float = 0.0               # stop early when best score <= tol
    seed: int = 0


class RealCodedGA:
    def __init__(self, objective: Callable[[np.ndarray], float], cfg: GAConfig):
        self.objective = objective          # genome -> scalar to MINIMISE
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.lo = np.array([b[0] for b in cfg.bounds], float)
        self.hi = np.array([b[1] for b in cfg.bounds], float)
        self.history: list[dict] = []

    # -- public -------------------------------------------------------------- #
    def run(self, verbose: bool = True) -> Individual:
        pop = [self._random_individual() for _ in range(self.cfg.pop_size)]
        for ind in pop:
            ind.update(self.objective(ind.genome))

        for gen in range(self.cfg.n_gen):
            pop.sort(key=lambda d: d.score)
            best = pop[0]
            self.history.append({"gen": gen, "best": best.score,
                                 "best_genome": best.genome.copy()})
            if verbose:
                g = ", ".join(f"{v:.3f}" for v in best.genome)
                print(f"  gen {gen:2d}: best objective={best.score:.4f}  genome=[{g}]")
            if best.score <= self.cfg.tol:
                break

            elites = [Individual(d.genome.copy(), d.score, d.n_evals)
                      for d in pop[:self.cfg.n_elite]]
            # Re-evaluate elites so noise can't entrench a lucky one.
            for e in elites:
                e.update(self.objective(e.genome))

            children: list[Individual] = []
            while len(children) < self.cfg.pop_size - self.cfg.n_elite:
                p1 = self._tournament(pop)
                p2 = self._tournament(pop)
                child_genome = self._crossover(p1.genome, p2.genome)
                child_genome = self._mutate(child_genome)
                child = Individual(child_genome)
                child.update(self.objective(child_genome))
                children.append(child)

            pop = elites + children

        pop.sort(key=lambda d: d.score)
        return pop[0]

    # -- operators ----------------------------------------------------------- #
    def _random_individual(self) -> Individual:
        g = self.rng.uniform(self.lo, self.hi)
        return Individual(genome=g)

    def _tournament(self, pop) -> Individual:
        picks = self.rng.choice(len(pop), size=self.cfg.tournament_k, replace=False)
        return min((pop[i] for i in picks), key=lambda d: d.score)

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # BLX-alpha: sample each gene from an interval extended beyond [a,b].
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        d = hi - lo
        alpha = self.cfg.crossover_alpha
        child = self.rng.uniform(lo - alpha * d, hi + alpha * d)
        return self._clip_reflect(child)

    def _mutate(self, g: np.ndarray) -> np.ndarray:
        sigma = self.cfg.mutation_frac * (self.hi - self.lo)
        mask = self.rng.random(g.shape) < self.cfg.mutation_prob
        g = g + mask * self.rng.normal(0.0, sigma)
        return self._clip_reflect(g)

    def _clip_reflect(self, g: np.ndarray) -> np.ndarray:
        # Reflect at the bounds (keeps diversity better than hard clipping).
        span = self.hi - self.lo
        for i in range(len(g)):
            if span[i] == 0:
                g[i] = self.lo[i]
                continue
            x = (g[i] - self.lo[i]) % (2 * span[i])
            if x > span[i]:
                x = 2 * span[i] - x
            g[i] = self.lo[i] + x
        return g
