from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .body import BodyGenome
from .controller import ControllerGenome
from .fitness import scalar_fitness

@dataclass
class Individual:
    body: BodyGenome
    controller: ControllerGenome
    fitness: float | None = None
    metrics: dict | None = None


def random_population(n: int, width: int, height: int, rng: np.random.Generator) -> list[Individual]:
    return [Individual(BodyGenome.random(width, height, rng), ControllerGenome.random((height, width), rng)) for _ in range(n)]


def evolve(env, cfg: dict, ctrl_cfg: dict, rng: np.random.Generator):
    pop = random_population(cfg["population_size"], cfg["grid"]["width"], cfg["grid"]["height"], rng)
    history = []
    evo_cfg = cfg["evolution"]
    for gen in range(evo_cfg["generations"]):
        for ind in pop:
            metrics = env.evaluate(ind.body, ind.controller, ctrl_cfg)
            ind.metrics = metrics
            ind.fitness = scalar_fitness(metrics)
        pop.sort(key=lambda x: x.fitness, reverse=True)
        best = pop[0]
        history.append({"generation": gen, "best_fitness": best.fitness, **best.metrics})
        elites = pop[:evo_cfg["elite_count"]]
        children = list(elites)
        while len(children) < evo_cfg["population_size"]:
            parent = rng.choice(elites)
            children.append(Individual(
                parent.body.mutate(evo_cfg["mutation_rate"], rng),
                parent.controller.mutate(evo_cfg["mutation_rate"], rng),
            ))
        pop = children
    return pop[0], history
