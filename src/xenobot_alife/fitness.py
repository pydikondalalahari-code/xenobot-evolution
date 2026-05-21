def scalar_fitness(metrics: dict) -> float:
    connectivity_bonus = 5.0 if metrics.get("connected") else -10.0
    return (
        10.0 * metrics["collected"]
        - 0.5 * metrics["energy"]
        - 0.2 * metrics["mass"]
        + connectivity_bonus
    )
