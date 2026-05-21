from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .body import BodyGenome

@dataclass
class ControllerGenome:
    phase: np.ndarray
    gain: np.ndarray

    @classmethod
    def random(cls, shape: tuple[int, int], rng: np.random.Generator) -> "ControllerGenome":
        return cls(
            phase=rng.uniform(0, 2 * np.pi, size=shape),
            gain=rng.normal(0.0, 1.0, size=shape),
        )

    def mutate(self, rate: float, rng: np.random.Generator) -> "ControllerGenome":
        phase = self.phase.copy(); gain = self.gain.copy()
        mask = rng.random(phase.shape) < rate
        phase[mask] += rng.normal(0, 0.25, size=mask.sum())
        gain[mask] += rng.normal(0, 0.15, size=mask.sum())
        return ControllerGenome(phase=np.mod(phase, 2 * np.pi), gain=gain)

    def actuation(self, body: BodyGenome, t: int, frequency: float, amplitude: float) -> np.ndarray:
        # TODO: Add local coupling between neighboring oscillator phases.
        signal = amplitude * np.sin(frequency * t + self.phase) * np.tanh(self.gain)
        return signal * (body.voxels >= 2)
