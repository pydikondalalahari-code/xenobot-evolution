# src/fitness.py
#first simultor → evosoro sim → heuristic (if one fails, another as backup)

import numpy as np
from src.representation import (
    largest_connected_component, EMPTY, PASSIVE, ACTIVE_P, ACTIVE_N
)


_BACKEND = "heuristic"

try:
    from xenobot_alife.simulate import simulate_genome   
    _BACKEND = "xenobot_alife"
except ImportError:
    pass

if _BACKEND == "heuristic":
    try:
        from evosoro.softbot import SoftBot                     
        from evosoro.networks import DirectEncoding             
        from evosoro.objectives import displacement             
        _BACKEND = "evosoro"
    except ImportError:
        pass


# in this function, the genome and 1000simulator steps are parameters
def evaluate_genome(genome: np.ndarray, sim_time: int = 1000) -> tuple:
 
    # first keep only the largest connected body
    genome = largest_connected_component(genome)

    #Reject empty bodies immediately
    if (genome != EMPTY).sum() == 0:
        return (0.0,)

    
    if _BACKEND == "xenobot_alife":
        return _eval_xenobot_alife(genome, sim_time)
    elif _BACKEND == "evosoro":
        return _eval_evosoro(genome, sim_time)
    else:
        return _eval_heuristic(genome)



def _eval_xenobot_alife(genome: np.ndarray, sim_time: int) -> tuple:
    """Evaluate using the xenobot_alife course package."""
    try:
        result = simulate_genome(genome, steps=sim_time)
        # xenobot_alife may return scalar or dict.measures fitness through displacement
        if isinstance(result, dict):
            fitness = float(result.get("displacement", result.get("fitness", 0.0)))
        else:
            fitness = float(result)
        return (max(0.0, fitness),)
    except Exception as e:
        print(f"[fitness] xenobot_alife simulation error: {e}; falling back to heuristic")
        return _eval_heuristic(genome)


def _eval_evosoro(genome: np.ndarray, sim_time: int) -> tuple:
    """Evaluate using the evosoro soft-body physics simulator."""
    try:
        net = DirectEncoding(output_node_names=["material"],
                             orig_size_xyz=genome.shape)
        # Inject genome values directly into network outputs
        net.graph.nodes["material"]["state"] = genome.flatten().tolist()
        bot = SoftBot(max_id=0, objective_dict={"fitness": {"name": "displacement",
                                                             "maximize": True,
                                                             "tag": "<fitness>"}},
                      genotype=net)
        bot.phenotype = genome
        fitness = displacement(bot, sim_time=sim_time)
        return (max(0.0, float(fitness)),)
    except Exception as e:
        print(f"[fitness] evosoro error: {e}; falling back to heuristic")
        return _eval_heuristic(genome)


def _eval_heuristic(genome: np.ndarray) -> tuple:
    X, Y, Z = genome.shape
    #check genome is empty or not
    filled_mask = genome != EMPTY

    # if no body 0 fitness=0
    n_filled = filled_mask.sum()
    if n_filled == 0:
        return (0.0,)

    # get body centre coordinates
    xs, ys, zs = np.where(filled_mask)
    cx = xs.mean() / X    

    # find active muscles
    active_mask = (genome == ACTIVE_P) | (genome == ACTIVE_N)
    #count how many active muscles there are
    n_active = active_mask.sum()
    #if no active then 0
    if n_active == 0:
        return (0.0,)

    #compare body centre and muscle centre..if same it is probably wiggiling at same place.oif assymettry is more then it is likely pushing from one side
    ax = np.where(active_mask)[0].mean() / X
    asymmetry = abs(ax - cx)   

    # count both muscle types.if both are equal good diversity.moving in rythum..if no then poor
    n_p = (genome == ACTIVE_P).sum()
    n_n = (genome == ACTIVE_N).sum()
    phase_diversity = 2 * min(n_p, n_n) / max(n_active, 1)

    # checks for percentage of bidy which is muscle.the less the bad moment.it should be 0.4 if it is farther penalized.
    ratio = n_active / n_filled
    ratio_score = 1.0 - abs(ratio - 0.4) / 0.4

    # Size penalty-very tiny or very dense bodies are penalised
    density = n_filled / genome.size
    size_score = np.exp(-((density - 0.35) ** 2) / (2 * 0.2 ** 2))

    # Vertical centre of mass (lower = more stable)
    z_mean = zs.mean() / Z
    stability = 1.0 - z_mean

    # Combine
    fitness = (asymmetry * 0.35
               + phase_diversity * 0.25
               + ratio_score * 0.15
               + size_score * 0.15
               + stability * 0.10)

    # Add tiny Gaussian noise to avoid neutral networks being completely flat
    fitness += np.random.normal(0, 0.005)
    fitness = max(0.0, float(fitness))

    return (fitness,)


def get_backend() -> str:
    return _BACKEND
