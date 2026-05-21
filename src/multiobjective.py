# src/multiobjective.py

import numpy as np
import pickle
from pathlib import Path

from deap import base, creator, tools, algorithms

from src.representation import random_genome, largest_connected_component, EMPTY
from src.fitness import evaluate_genome
from src.ea import (
    mutate_random_flip, mutate_block, mutate_grow_shrink,
    cx_uniform_voxel, cx_one_point_slice, diversity
)

GRID = (8, 8, 8)

# DEAP types (multi-objective)
if not hasattr(creator, "FitnessMO"):
    creator.create("FitnessMO", base.Fitness, weights=(1.0, -1.0))
if not hasattr(creator, "IndividualMO"):
    creator.create("IndividualMO", np.ndarray, fitness=creator.FitnessMO)


def _make_individual_mo():
    return creator.IndividualMO(random_genome(GRID))


def _evaluate_mo(genome: np.ndarray) -> tuple:
    """Two-objective evaluation: (distance, voxel_count)."""
    genome_lcc = largest_connected_component(genome)
    distance   = evaluate_genome(genome_lcc)[0]
    n_voxels   = float((genome_lcc != EMPTY).sum())
    return distance, n_voxels


# ── Multi-objective toolbox ───────────────────────────────────────────────
toolbox_mo = base.Toolbox()
toolbox_mo.register("individual", _make_individual_mo)
toolbox_mo.register("population", tools.initRepeat, list, toolbox_mo.individual)
toolbox_mo.register("evaluate",   _evaluate_mo)
toolbox_mo.register("mate",       cx_uniform_voxel)
toolbox_mo.register("mutate",     mutate_random_flip)
toolbox_mo.register("select",     tools.selNSGA2)       # NSGA-II selection



def run_nsga2(
    pop_size: int = 100,
    n_gen: int = 200,
    cxpb: float = 0.5,
    mutpb: float = 0.3,
    results_dir: str = "results",
    run_tag: str = "nsga2",
    verbose: bool = True,
) -> tuple:


    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    pop = toolbox_mo.population(n=pop_size)

    # Evaluate initial population
    fits = list(map(toolbox_mo.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit

    # Assign initial crowding distance
    pop = toolbox_mo.select(pop, len(pop))

    logbook  = tools.Logbook()
    stats_d  = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_v  = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats_d.register("max",  np.max)
    stats_d.register("mean", np.mean)
    stats_v.register("min",  np.min)
    stats_v.register("mean", np.mean)

    mstats = tools.MultiStatistics(distance=stats_d, voxels=stats_v)

    for gen in range(1, n_gen + 1):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox_mo.clone(o) for o in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox_mo.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox_mo.mutate(mutant)
                del mutant.fitness.values

        # Re-evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox_mo.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        # NSGA-II selection: (μ + λ)
        pop = toolbox_mo.select(pop + offspring, pop_size)

        record = mstats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid),
                       diversity=diversity(pop), **record)
        if verbose:
            print(logbook.stream)

    # Save
    log_path = results_path / f"{run_tag}_log.pkl"
    with open(log_path, "wb") as f:
        pickle.dump(logbook, f)

    print(f"\nNSGA-II complete. Logs saved to {results_path}/")
    return pop, logbook




def get_pareto_front(population) -> list:
    """Extract the non-dominated front from a population."""
    front = tools.sortNondominated(population, len(population), first_front_only=True)
    return front[0]


def knee_point(pareto_front) -> object:
    """Return the knee-point individual (best distance-voxels tradeoff).

    Uses the normalised Euclidean distance to the ideal point (max dist, 0 voxels).
    """
    dists  = np.array([ind.fitness.values[0] for ind in pareto_front])
    voxels = np.array([ind.fitness.values[1] for ind in pareto_front])

    # Normalise to [0,1]
    d_norm = (dists  - dists.min())  / (dists.max()  - dists.min()  + 1e-9)
    v_norm = (voxels - voxels.min()) / (voxels.max() - voxels.min() + 1e-9)

    # Ideal = max distance, min voxels → (1, 0) in normalised space
    scores = np.sqrt((d_norm - 1)**2 + v_norm**2)
    return pareto_front[int(np.argmin(scores))]


def hypervolume_indicator(population, ref_point=(0.0, 1000.0)) -> float:
    """Approximate hypervolume of the Pareto front.

    Computed via sweep-line on 2-D front.  Reference point should be
    dominated by all individuals (worst possible values).
    """
    front = get_pareto_front(population)
    # Sort by first objective descending
    front_sorted = sorted(front,
                          key=lambda ind: ind.fitness.values[0], reverse=True)
    hv = 0.0
    prev_v = ref_point[1]
    for ind in front_sorted:
        d, v = ind.fitness.values
        if v < prev_v and d > ref_point[0]:
            hv += (d - ref_point[0]) * (prev_v - v)
            prev_v = v
    return hv
