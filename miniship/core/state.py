from dataclasses import dataclass
from typing import List
import numpy as np
from ..dynamics.ship import Ship

@dataclass
class WorldState:
    ships: List[Ship]
    t: float
    step: int

@dataclass
class StepCache:
    tc0: np.ndarray | None = None
    dc0: np.ndarray | None = None
    v_cap: np.ndarray | None = None
    bestX0: np.ndarray | None = None

    tc: np.ndarray | None = None
    dc: np.ndarray | None = None
    risk: np.ndarray | None = None
    dmin_i: np.ndarray | None = None
    vj_max: np.ndarray | None = None
