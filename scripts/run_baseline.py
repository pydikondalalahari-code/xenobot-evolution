from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from xenobot_alife.env import CleanupEnv, CleanupEnvConfig
from xenobot_alife.evolution import evolve
from xenobot_alife.render import plot_body, plot_trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = np.random.default_rng(cfg["seed"])
    env_cfg = CleanupEnvConfig(**cfg["env"])
    env = CleanupEnv(env_cfg, rng)
    best, history = evolve(env, cfg, cfg["controller"], rng)
    out = Path("runs/baseline")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).drop(columns=["trajectory"], errors="ignore").to_csv(out / "history.csv", index=False)
    plot_body(best.body, out / "best_body.png")
    plot_trajectory(best.metrics, out / "best_trajectory.png")
    print("Best fitness:", best.fitness)
    print({k: v for k, v in best.metrics.items() if k != "trajectory"})
    print(f"Artifacts saved to {out}")

if __name__ == "__main__":
    main()
