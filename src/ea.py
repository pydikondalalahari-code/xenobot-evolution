# src/ea.py
"""
Evolutionary Algorithm using DEAP for Xenobot morphological search.
Mutation  : mutate_random_flip, mutate_block, mutate_grow_shrink
Crossover : cx_uniform_voxel, cx_one_point_slice
Selection : tournament (via toolbox.register)
"""

import numpy as np
import pickle
import json
from pathlib import Path

from deap import base, creator, tools, algorithms

from src.representation import (
    random_genome, largest_connected_component, N_MATERIALS,
    EMPTY, PASSIVE, ACTIVE_P, ACTIVE_N
)
from src.fitness import evaluate_genome

GRID = (8, 8, 8)

# DEAP Type Setup

if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #Individual = NumPy genome array + fitness metadata
if not hasattr(creator, "Individual"):
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


toolbox = base.Toolbox()

#random_genome(GRID) creates an 8×8×8 voxel genome
#creator.Individual(...) wraps it as a DEAP individual with fitness metadata
def make_individual() -> "creator.Individual":
    return creator.Individual(random_genome(GRID))


toolbox.register("individual", make_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate_genome)


#randomly change some voxels

def mutate_random_flip(individual: np.ndarray, flip_prob: float = 0.05):
    #If flip_prob=0.05, each voxel has 5% chance to be selected.
    mask = np.random.random(individual.shape) < flip_prob
    #Creates random new material values: 0, 1, 2, or 3.
    new_vals = np.random.randint(0, N_MATERIALS, size=individual.shape, dtype=np.int32)
    #only selected voxels get changed
    individual[mask] = new_vals[mask]
    del individual.fitness.values   
    return individual,


def mutate_block(individual: np.ndarray, block_size: int = 2):

    X, Y, Z = individual.shape
    # Pick a random corner so the block fits inside the grid
    x0 = np.random.randint(0, max(1, X - block_size + 1))
    y0 = np.random.randint(0, max(1, Y - block_size + 1))
    z0 = np.random.randint(0, max(1, Z - block_size + 1))

    bx = min(block_size, X - x0)
    by = min(block_size, Y - y0)
    bz = min(block_size, Z - z0)
    #Creates random materials for the block.
    new_block = np.random.randint(0, N_MATERIALS,
                                  size=(bx, by, bz), dtype=np.int32)
    individual[x0:x0+bx, y0:y0+by, z0:z0+bz] = new_block
    del individual.fitness.values
    return individual,

#Adds or removes a voxel near the robot surface.
#Simulates body growth or erosion in a biologically inspired way.
#Makes gradual shape modifications instead of random noise.
def mutate_grow_shrink(individual: np.ndarray,
                       prob_grow: float = 0.3,
                       prob_shrink: float = 0.3):
  
    roll = np.random.random()

    if roll < prob_grow:
        filled = (individual != EMPTY)
        candidates = []
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            shifted = np.roll(filled, (dx, dy, dz), axis=(0, 1, 2))
            # EMPTY voxels neighbouring a filled voxel
            empty_neighbour = (~filled) & shifted
            coords = list(zip(*np.where(empty_neighbour)))
            candidates.extend(coords)

        if candidates:
            x, y, z = candidates[np.random.randint(len(candidates))]
            individual[x, y, z] = np.random.randint(1, N_MATERIALS)  # non-empty
            del individual.fitness.values

    elif roll < prob_grow + prob_shrink:
        # Shrink: remove one surface voxel 
        filled = (individual != EMPTY)
        empty  = ~filled
        surface = []
        for dx, dy, dz in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
            shifted = np.roll(empty, (dx, dy, dz), axis=(0, 1, 2))
            on_surface = filled & shifted
            coords = list(zip(*np.where(on_surface)))
            surface.extend(coords)

        if surface:
            x, y, z = surface[np.random.randint(len(surface))]
            individual[x, y, z] = EMPTY
            del individual.fitness.values

    return individual,


#Mixes two parent robots voxel-by-voxel randomly.
#Each voxel has a probability of being swapped between parents.
#Creates children with mixed traits from both parents.
def cx_uniform_voxel(ind1: np.ndarray, ind2: np.ndarray, indpb: float = 0.5):

    mask = np.random.random(ind1.shape) < indpb
    # Swap where mask is True
    tmp = ind1[mask].copy()
    ind1[mask] = ind2[mask]
    ind2[mask] = tmp
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2

#Splits parents at one z-layer and swaps the upper sections.
#Preserves larger spatial body regions compared to uniform crossover.
#Produces more structurally coherent offspring
def cx_one_point_slice(ind1: np.ndarray, ind2: np.ndarray):

    Z = ind1.shape[2]
    cut = np.random.randint(1, Z)          
    tmp = ind1[:, :, cut:].copy()
    ind1[:, :, cut:] = ind2[:, :, cut:]
    ind2[:, :, cut:] = tmp
    del ind1.fitness.values
    del ind2.fitness.values
    return ind1, ind2

#Swaps large connected body regions between parents.
#Repairs disconnected pieces after crossover using connectivity repair.
#Designed specifically for valid voxel robot morphologies.
def cx_morphological(ind1: np.ndarray, ind2: np.ndarray):

    axis = np.random.randint(0, 3)          # 0=x, 1=y, 2=z
    cut  = np.random.randint(1, ind1.shape[axis])

    # Build slice objects for the "upper" half
    slices = [slice(None)] * 3
    slices[axis] = slice(cut, None)
    slices = tuple(slices)

    tmp = ind1[slices].copy()
    ind1[slices] = ind2[slices]
    ind2[slices] = tmp

    # Repair connectivity
    ind1[:] = largest_connected_component(ind1)
    ind2[:] = largest_connected_component(ind2)

    if hasattr(ind1, "fitness"):
       del ind1.fitness.values

    if hasattr(ind2, "fitness"):
       del ind2.fitness.values
        
    return ind1, ind2



toolbox.register("mate",   cx_uniform_voxel)
toolbox.register("mutate", mutate_random_flip)
toolbox.register("select", tools.selTournament, tournsize=3)


#Measures how different the robots are from each other.
#Uses average Hamming distance across the population.
#Helps detect premature convergence during evolution.

def diversity(population) -> float:

    flat = np.array([ind.flatten() for ind in population])
    n = len(flat)
    if n <= 1:
        return 0.0
    total = sum(
        (flat[i] != flat[j]).mean()
        for i in range(n)
        for j in range(i + 1, n)
    )
    return total / (n * (n - 1) / 2)



#Creates statistics tracking for the evolutionary run.
#Tracks maximum, mean, and standard deviation of fitness values.
#Used to monitor evolutionary progress over generations.
def build_stats() -> tools.Statistics:
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("std",  np.std)
    return stats


#Runs the complete evolutionary algorithm loop.
#Creates populations, evaluates fitness, applies mutation/crossover, and selects better robots repeatedly.
#Saves logs and stores the best evolved robot.

def run_evolution(
    pop_size: int = 50,
    n_gen: int = 200,
    mu: int = 50,
    lambda_: int = 100,
    cxpb: float = 0.5,
    mutpb: float = 0.3,
    results_dir: str = "results",
    run_tag: str = "baseline",
    verbose: bool = True,
) -> tuple:

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(5, similar=lambda a, b: (a == b).all())
    stats = build_stats()
    logbook = tools.Logbook()

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(pop),
                   diversity=diversity(pop), **record)
    if verbose:
        print(logbook.stream)

    for gen in range(1, n_gen + 1):
        offspring = algorithms.varOr(pop, toolbox, lambda_, cxpb, mutpb)

        # Evaluate only invalid (modified) offspring
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fits = list(map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, fits):
            ind.fitness.values = fit

        # (μ+λ) selection
        pop = toolbox.select(pop + offspring, mu)
        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid),
                       diversity=diversity(pop), **record)
        if verbose:
            print(logbook.stream)

    # Save log
    log_path = results_path / f"{run_tag}_log.pkl"
    with open(log_path, "wb") as f:
        pickle.dump(logbook, f)

    # Save best genome as JSON
    best = hof[0]
    best_path = results_path / f"{run_tag}_best.json"
    with open(best_path, "w") as f:
        json.dump({"genome": best.tolist(),
                   "fitness": best.fitness.values[0]}, f)

    print(f"\nBest fitness: {hof[0].fitness.values[0]:.4f}")
    print(f"Logs saved to {results_path}/")
    return pop, logbook, hof
