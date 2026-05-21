from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# 0 empty, 1 passive tissue, 2 contractile tissue, 3 ciliated/propulsive tissue

@dataclass
class BodyGenome:
    voxels: np.ndarray

    @classmethod
    def random(cls, width: int, height: int, rng: np.random.Generator) -> "BodyGenome":
        voxels = rng.choice([0, 1, 2, 3], size=(height, width), p=[0.45, 0.25, 0.20, 0.10])
        voxels[height // 2, width // 2] = 1
        return cls(voxels=voxels.astype(np.int8))

    def mutate(self, rate: float, rng: np.random.Generator) -> "BodyGenome":
        child = self.voxels.copy()
        mask = rng.random(child.shape) < rate
        child[mask] = rng.integers(0, 4, size=mask.sum())
        child[child.shape[0] // 2, child.shape[1] // 2] = max(child[child.shape[0] // 2, child.shape[1] // 2], 1)
        return BodyGenome(child)

    @property
    def mass(self) -> float:
        return float(np.count_nonzero(self.voxels))

    @property
    def actuator_count(self) -> int:
        return int(np.count_nonzero(self.voxels >= 2))

    def is_connected(self) -> bool:
        occupied = self.voxels > 0
        if not occupied.any():
            return False
        start = tuple(np.argwhere(occupied)[0])
        stack = [start]
        seen = {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1]:
                    if occupied[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc)); stack.append((nr, nc))
        return len(seen) == int(occupied.sum())

    def repair_or_penalize(self) -> "BodyGenome":
        # TODO: Advanced students: keep largest connected component rather than returning self.
        return self
