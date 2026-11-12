"""
Phase 4 (Step 4.2): Heuristic Weight & Parameter Tuning Engine
Implements the three coursework optimization algorithms for tuning `EvalParams`:
  - Option A: Genetic Algorithm (GA) - tournament selection, crossover, Gaussian mutation
  - Option B: Simulated Annealing (SA) - Boltzmann acceptance exp(dE / T) with cooling
  - Option C: Local Hill Climbing (HC) - greedy neighborhood perturbation
All match evaluations use FIXED DEPTH with color-swapped pairs (rules.txt Section 7.3).
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path
import random
import sys
from typing import List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agents.search_agent import SearchAgent
from agents.tuned_agent import TunedSearchAgent
from benchmarks.arena import run_tournament
from engine.eval_tuned import EvalParams, TUNED_PARAMS


def evaluate_fitness(
    params: EvalParams,
    search_depth: int = 2,
    num_games: int = 8,
    seed: int = 42,
) -> float:
    """
    Evaluate fitness of candidate `params` by playing `num_games` (color-swapped pairs)
    at fixed `search_depth` against the untuned baseline SearchAgent.
    Returns score rate in [0.0, 1.0].
    """
    candidate = TunedSearchAgent(
        max_depth=search_depth,
        params=params,
        use_quiescence=True,
        name="Candidate",
    )
    baseline = SearchAgent(
        max_depth=search_depth,
        use_quiescence=False,
        name="Baseline",
    )
    stats = run_tournament(
        candidate,
        baseline,
        num_games=num_games,
        opening_plies=(2, 4),
        seed=seed,
        verbose=False,
    )
    return stats.score_rate


def mutate_params(params: EvalParams, rng: random.Random, scale: float = 1.0) -> EvalParams:
    """Apply Gaussian perturbation to a candidate parameter vector."""
    vec = params.to_vector()
    sigmas = [10.0, 20.0, 20.0, 25.0, 40.0, 0.15, 8.0, 10.0, 3.0]
    mutated = []
    for val, sig in zip(vec, sigmas):
        if rng.random() < 0.45:
            val += rng.gauss(0.0, sig * scale)
        mutated.append(val)
    return EvalParams.from_vector(mutated)


def crossover_params(p1: EvalParams, p2: EvalParams, rng: random.Random) -> EvalParams:
    """Uniform / arithmetic blend crossover between two parent chromosomes."""
    v1 = p1.to_vector()
    v2 = p2.to_vector()
    child = []
    for a, b in zip(v1, v2):
        if rng.random() < 0.5:
            alpha = rng.uniform(0.25, 0.75)
            child.append(alpha * a + (1.0 - alpha) * b)
        else:
            child.append(a if rng.random() < 0.5 else b)
    return EvalParams.from_vector(child)


def tune_genetic_algorithm(
    pop_size: int = 8,
    generations: int = 4,
    search_depth: int = 2,
    games_per_eval: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[EvalParams, float]:
    """Evolutionary optimization using elitism, tournament selection, crossover, and mutation."""
    rng = random.Random(seed)
    untuned_seed = EvalParams(
        pawn=100, knight=300, bishop=300, rook=500, queen=900,
        pst_scale=1.0, passed_pawn_bonus=20, unblocked_rank4_bonus=25, mopup_weight=8,
    )

    population: List[EvalParams] = [TUNED_PARAMS, untuned_seed]
    while len(population) < pop_size:
        population.append(mutate_params(TUNED_PARAMS, rng, scale=1.0))

    best_params = TUNED_PARAMS
    best_fitness = -1.0

    for gen in range(1, generations + 1):
        scored: List[Tuple[float, EvalParams]] = []
        for cand in population:
            fit = evaluate_fitness(
                cand, search_depth=search_depth, num_games=games_per_eval, seed=seed + gen * 10
            )
            scored.append((fit, cand))
            if fit > best_fitness:
                best_fitness = fit
                best_params = cand

        scored.sort(key=lambda x: x[0], reverse=True)
        if verbose:
            top_fit, top_cand = scored[0]
            print(
                f"[GA Gen {gen}/{generations}] Top Fitness: {top_fit * 100:.1f}% | "
                f"Best Overall: {best_fitness * 100:.1f}% | Params: {top_cand}"
            )

        next_pop: List[EvalParams] = [scored[0][1], scored[1][1]]
        while len(next_pop) < pop_size:
            t1 = rng.sample(scored, k=min(3, len(scored)))
            t2 = rng.sample(scored, k=min(3, len(scored)))
            parent1 = max(t1, key=lambda x: x[0])[1]
            parent2 = max(t2, key=lambda x: x[0])[1]
            child = crossover_params(parent1, parent2, rng)
            child = mutate_params(child, rng, scale=0.8)
            next_pop.append(child)

        population = next_pop

    return best_params, best_fitness


def tune_simulated_annealing(
    iterations: int = 10,
    initial_temp: float = 0.15,
    cooling_rate: float = 0.82,
    search_depth: int = 2,
    games_per_eval: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[EvalParams, float]:
    """Simulated Annealing optimizer with Boltzmann acceptance probability exp(delta / T)."""
    rng = random.Random(seed)
    current = TUNED_PARAMS
    current_fit = evaluate_fitness(current, search_depth=search_depth, num_games=games_per_eval, seed=seed)
    best, best_fit = current, current_fit
    temp = initial_temp

    for step in range(1, iterations + 1):
        neighbor = mutate_params(current, rng, scale=0.7)
        neighbor_fit = evaluate_fitness(
            neighbor, search_depth=search_depth, num_games=games_per_eval, seed=seed + step
        )
        delta = neighbor_fit - current_fit

        if delta >= 0 or rng.random() < math.exp(delta / max(temp, 1e-6)):
            current, current_fit = neighbor, neighbor_fit
            if current_fit > best_fit:
                best, best_fit = current, current_fit

        if verbose:
            print(
                f"[SA Step {step}/{iterations}] Temp={temp:.4f} | Curr={current_fit * 100:.1f}% | "
                f"Best={best_fit * 100:.1f}%"
            )
        temp *= cooling_rate

    return best, best_fit


def tune_hill_climbing(
    steps: int = 8,
    neighbors_per_step: int = 3,
    search_depth: int = 2,
    games_per_eval: int = 8,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[EvalParams, float]:
    """Steepest-ascent / stochastic local Hill Climbing over evaluation parameters."""
    rng = random.Random(seed)
    current = TUNED_PARAMS
    current_fit = evaluate_fitness(current, search_depth=search_depth, num_games=games_per_eval, seed=seed)

    for step in range(1, steps + 1):
        best_neighbor = current
        best_neighbor_fit = current_fit

        for _ in range(neighbors_per_step):
            cand = mutate_params(current, rng, scale=0.5)
            fit = evaluate_fitness(
                cand, search_depth=search_depth, num_games=games_per_eval, seed=seed
            )
            if fit > best_neighbor_fit:
                best_neighbor_fit = fit
                best_neighbor = cand

        if best_neighbor_fit > current_fit:
            current, current_fit = best_neighbor, best_neighbor_fit

        if verbose:
            print(f"[HC Step {step}/{steps}] Best Score Rate: {current_fit * 100:.1f}% | {current}")

    return current, current_fit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune 5x5 Minichess Evaluation Weights")
    parser.add_argument("--method", choices=["ga", "sa", "hc"], default="ga")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--games", type=int, default=8)
    args = parser.parse_args()

    if args.method == "ga":
        best_p, best_f = tune_genetic_algorithm(search_depth=args.depth, games_per_eval=args.games)
    elif args.method == "sa":
        best_p, best_f = tune_simulated_annealing(search_depth=args.depth, games_per_eval=args.games)
    else:
        best_p, best_f = tune_hill_climbing(search_depth=args.depth, games_per_eval=args.games)

    print("\nOptimized Parameters:", best_p)
    print(f"Best Match Fitness  : {best_f * 100:.1f}%")

