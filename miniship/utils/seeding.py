import numpy as np

class RNG:
    """简单的可复现随机数封装。"""
    def __init__(self, seed: int | None = None):
        self.seed(seed)

    def seed(self, seed: int | None):
        self._seed = int(seed) if seed is not None else None
        self._rng = np.random.default_rng(self._seed)

    @property
    def rs(self) -> np.random.Generator:
        return self._rng

