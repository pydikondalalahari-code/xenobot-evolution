# src/diagnostics.py
"""
Diagnostic visualisations for the Xenobot evolutionary run.
It does visualisation and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from src.representation import EMPTY, N_MATERIALS

# my understanding : A fitness curve shows how the population improves during evolution over generations.Max fitness → best robot in each generation
#Mean fitness → average performance of all robots
#Shaded std band → variation/spread in population fitness

def plot_fitness_curve(logbook, save_path=None, label="run", color="steelblue"):

    gen  = logbook.select("gen")
    maxf = logbook.select("max")
    meanf= logbook.select("mean")
    stdf = logbook.select("std")

    maxf  = np.array(maxf)
    meanf = np.array(meanf)
    stdf  = np.array(stdf)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gen, maxf,  label=f"{label} – max",  color=color, lw=2)
    ax.plot(gen, meanf, label=f"{label} – mean", color=color, lw=1.5,
            linestyle="--", alpha=0.8)
    ax.fill_between(gen, meanf - stdf, meanf + stdf,
                    color=color, alpha=0.15, label="±1 std")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (displacement)")
    ax.set_title("Fitness Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

#This compares evolved better robots done through random flip mutation run, block mutation run and grow/shrink mutation run

def plot_multi_fitness_curves(logbooks, labels, colors=None, save_path=None):

    if colors is None:
        colors = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5))
    #first need to go through each experiment
    for logbook, label, color in zip(logbooks, labels, colors):
        gen   = logbook.select("gen")
        maxf  = np.array(logbook.select("max"))
        meanf = np.array(logbook.select("mean"))
        stdf  = np.array(logbook.select("std"))
        ax.plot(gen, maxf, label=f"{label} max", color=color, lw=2)
        ax.fill_between(gen, maxf - stdf, maxf + stdf,
                        color=color, alpha=0.12)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.set_title("Operator Comparison – Fitness Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#logbook → stores evolutionary statistics generation-wise
def plot_diversity_curve(logbook, save_path=None):
#Gets all generation numbers from logbook.
    gen = np.array(logbook.select("gen"))

    #Checks whether diversity values were recorded.
    if "diversity" not in logbook[0]:
        print("No 'diversity' key in logbook – did you record it?")
        return None, None
    #get diversity values for all generations
    div = np.array(logbook.select("diversity"))
    #If diversity falls below the threshold, the population becomes too similar
    threshold = 0.10 * div[0] if div[0] > 0 else 0.0

    collapse_gen = None
    for g, d in zip(gen, div):
        if d < threshold:
            collapse_gen = g
            break

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(gen, div, color="darkorange", lw=2)
    ax.axhline(threshold, color="red", linestyle="--", lw=1,
               label=f"10% of initial ({threshold:.3f})")
    if collapse_gen is not None:
        ax.axvline(collapse_gen, color="red", linestyle=":", lw=1.5,
                   label=f"Collapse at gen {collapse_gen}")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Mean Hamming distance")
    ax.set_title("Population Diversity Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, collapse_gen

#exploring seatch space.This function generates many random robot genomes and evaluates their fitness scores.  
def plot_fitness_landscape(evaluate_fn, n_samples=2000, save_path=None):
    from src.representation import random_genome
    fitnesses = []
    for _ in range(n_samples):
        g = random_genome()
        f = evaluate_fn(g)[0]
        fitnesses.append(f)

    fitnesses = np.array(fitnesses)
    frac_pos = (fitnesses > 0).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(fitnesses, bins=40, color="teal", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", lw=1.5, label="fitness = 0")
    ax.set_xlabel("Fitness (displacement proxy)")
    ax.set_ylabel("Count")
    ax.set_title(f"Fitness Landscape Sample (n={n_samples})\n"
                 f"Fraction > 0: {frac_pos:.1%}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, frac_pos


#helps in identifying recurring morphological patterns or body structures associated with high fitness. Analyses the best evolved robots and shows where different material types commonly appear in their bodies.  

#hall of fame has best evolved robots
def plot_material_heatmap(hall_of_fame, top_n=10, save_path=None):
    
    individuals = list(hall_of_fame)[:top_n]
    if not individuals:
        print("Hall of Fame is empty.")
        return None

    shape = individuals[0].shape
    # Takes all top robots and stacks them together.So that function can analyse all robots simultaneously.
    stack = np.stack([np.array(ind) for ind in individuals], axis=0)

    material_names = {1: "Passive", 2: "Active+", 3: "Active−"}
    material_colors = {1: "Blues", 2: "Greens", 3: "Reds"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (mat_id, name) in zip(axes, material_names.items()):
        # Frequency = mean over individuals × z-layers
        freq = (stack == mat_id).mean(axis=(0, 3))   # (X, Y) averaged over z
        im = ax.imshow(freq.T, origin="lower", cmap=material_colors[mat_id],
                       vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"{name} voxel frequency")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Material Heatmap", y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


#This function generates many random robot genomes and counts how many disconnected body fragments each one contains.
def plot_component_distribution(n_samples=500, grid_size=(8, 8, 8),
 
    from src.representation import random_genome
    from scipy.ndimage import label

   #number of connected body fragments for each robot.
    n_components = []

#generate one random robot each time and analyze connectivity
    for _ in range(n_samples):
        g = random_genome(grid_size)
        filled = (g != EMPTY).astype(int)
        _, n = label(filled)
        n_components.append(n)

    n_components = np.array(n_components)
    frac_disc = (n_components > 1).mean()
    frac_empty = (n_components == 0).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = range(0, n_components.max() + 2)
    ax.hist(n_components, bins=bins, align="left", color="mediumslateblue",
            edgecolor="white", rwidth=0.7)
    ax.set_xlabel("Number of connected components")
    ax.set_ylabel("Count")
    ax.set_title(f"Connected-Component Distribution (n={n_samples})\n"
                 f"Disconnected: {frac_disc:.1%}  |  Entirely empty: {frac_empty:.1%}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, frac_disc


#Multi-objective fitness means evaluating solutions using multiple goals simultaneously instead of a single score.
def plot_pareto_front(population, save_path=None):

# maximise movement distance
    distances = [ind.fitness.values[0] for ind in population]
    voxels    = [ind.fitness.values[1] for ind in population]

    # Identify Pareto-optimal individuals 
    #If another robot moves farther AND uses fewer voxels then current robot is worse.
    #add non dominated one into list.
    pareto = []
    for i, ind in enumerate(population):
        dominated = False
        for j, other in enumerate(population):
            if i == j:
                continue
            if (other.fitness.values[0] >= ind.fitness.values[0] and
                    other.fitness.values[1] <= ind.fitness.values[1] and
                    other.fitness.values != ind.fitness.values):
                dominated = True
                break
        if not dominated:
            pareto.append(i)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(distances, voxels, alpha=0.4, label="population", color="steelblue")

    px = [distances[i] for i in pareto]
    py = [voxels[i]    for i in pareto]
    ax.scatter(px, py, color="red", zorder=5, label="Pareto front")

    # Knee point – closest to origin in normalised space
    if pareto:
        px_n = (np.array(px) - min(distances)) / (max(distances) - min(distances) + 1e-9)
        py_n = (np.array(py) - min(voxels))    / (max(voxels) - min(voxels) + 1e-9)
        knee_idx = pareto[np.argmin(np.sqrt(px_n**2 + py_n**2))]
        ax.scatter(distances[knee_idx], voxels[knee_idx],
                   color="gold", s=200, zorder=6,
                   edgecolors="black", label="knee point")

    ax.set_xlabel("Locomotion distance")
    ax.set_ylabel("Voxel count (body mass proxy)")
    ax.set_title("NSGA-II Pareto Front")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
