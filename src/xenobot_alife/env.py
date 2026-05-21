from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .body import BodyGenome
from .controller import ControllerGenome

@dataclass
class CleanupEnvConfig:
    world_size: tuple[float, float] = (12.0, 6.0)
    debris_count: int = 40
    episode_steps: int = 400
    flow_mode: str = "vortex"
    flow_strength: float = 0.04
    collector_radius: float = 0.35
    energy_budget: float = 1.0

class CleanupEnv:
    def __init__(self, cfg: CleanupEnvConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def reset_debris(self) -> np.ndarray:
        w, h = self.cfg.world_size
        x = self.rng.uniform(w * 0.25, w * 0.95, size=self.cfg.debris_count)
        y = self.rng.uniform(h * 0.10, h * 0.90, size=self.cfg.debris_count)
        return np.column_stack([x, y])

    def flow(self, pos: np.ndarray) -> np.ndarray:
        w, h = self.cfg.world_size
        if self.cfg.flow_mode == "laminar":
            return np.column_stack([np.full(len(pos), -self.cfg.flow_strength), np.zeros(len(pos))])
        center = np.array([w / 2, h / 2])
        rel = pos - center
        swirl = np.column_stack([-rel[:, 1], rel[:, 0]]) / max(w, h)
        return -0.5 * self.cfg.flow_strength + self.cfg.flow_strength * swirl

    def body_thrust(self, body: BodyGenome, ctrl: ControllerGenome, t: int, frequency: float, amplitude: float) -> np.ndarray:
        act = ctrl.actuation(body, t, frequency, amplitude)
        yy, xx = np.indices(body.voxels.shape)
        center = np.array([(body.voxels.shape[1] - 1) / 2, (body.voxels.shape[0] - 1) / 2])
        lever = np.column_stack([(xx - center[0]).ravel(), (yy - center[1]).ravel()])
        a = act.ravel()[:, None]
        # Propulsive voxels push opposite their lever arm; contractile voxels bias toward shape change.
        thrust = np.sum(a * lever, axis=0) / (body.mass + 1e-6)
        return 0.05 * thrust

    def evaluate(self, body: BodyGenome, ctrl: ControllerGenome, ctrl_cfg: dict) -> dict:
        debris = self.reset_debris()
        robot = np.array([1.0, self.cfg.world_size[1] / 2])
        collected = np.zeros(len(debris), dtype=bool)
        energy = 0.0
        trajectory = []
        for t in range(self.cfg.episode_steps):
            thrust = self.body_thrust(body, ctrl, t, ctrl_cfg["frequency"], ctrl_cfg["amplitude"])
            robot = robot + thrust + self.flow(robot[None, :])[0]
            robot = np.clip(robot, [0, 0], self.cfg.world_size)
            debris = debris + self.flow(debris)
            debris = np.clip(debris, [0, 0], self.cfg.world_size)
            dist = np.linalg.norm(debris - robot, axis=1)
            newly = dist < self.cfg.collector_radius
            collected |= newly
            energy += float(np.sum(np.abs(thrust)))
            trajectory.append(robot.copy())
        return {
            "collected": int(collected.sum()),
            "energy": energy,
            "mass": body.mass,
            "connected": body.is_connected(),
            "trajectory": np.array(trajectory),
        }
