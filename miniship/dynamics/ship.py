
from dataclasses import dataclass
import math
import numpy as np
from ..utils.math import wrap_to_pi

@dataclass
class Ship:
    sid: int
    pos: np.ndarray   # (2,)
    goal: np.ndarray  # (2,)
    psi: float        # heading [rad]
    v: float          # speed (m/s)
    reached: bool = False

    def advance(self, dpsi_cmd: float, v_target: float, dt: float,
                dpsi_max: float, dv_max: float, v_min: float, v_max: float):
        """一阶质点模型：限幅转向/加速度 + 欧拉积分推进。"""
        dpsi_cmd = float(np.clip(dpsi_cmd, -dpsi_max, dpsi_max))
        self.psi = wrap_to_pi(self.psi + dpsi_cmd)

        dv = float(np.clip(v_target - self.v, -dv_max, dv_max))
        self.v = float(np.clip(self.v + dv, v_min, v_max))

        vel = np.array([math.cos(self.psi), math.sin(self.psi)], dtype=np.float64) * self.v
        self.pos = self.pos + dt * vel

    def check_reached(self, goal_tol: float):
        self.reached = self.reached or (np.linalg.norm(self.pos - self.goal) <= goal_tol)
