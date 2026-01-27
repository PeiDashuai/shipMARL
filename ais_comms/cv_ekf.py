# ais_comms/cv_ekf.py
from __future__ import annotations
import numpy as np


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


class CV_EKF:
    """
    Constant Velocity KF（线性）+ sog/cog 速度融合 + NIS 门控

    状态: x = [px, py, vx, vy]^T
    观测: z = [px_meas, py_meas, vx_meas, vy_meas]^T
    """

    def __init__(self,
                 process_std_acc: float = 0.2,
                 meas_std_pos: float = 3.0,
                 meas_std_vel: float = 0.6,
                 nis_thresh: float = 16.0):
        # 状态 & 协方差
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 1e3

        self.std_a = float(process_std_acc)
        self.std_p = float(meas_std_pos)
        self.std_v = float(meas_std_vel)
        self.nis_thresh = float(nis_thresh)

        self.R = np.diag([
            self.std_p ** 2, self.std_p ** 2,
            self.std_v ** 2, self.std_v ** 2
        ])

        self.last_ts: float | None = None

    # ---------- 初始化 ----------
    def _init_from_measurement(self, ts: float, meas: dict):
        px = float(meas["x"])
        py = float(meas["y"])
        sog = max(0.0, float(meas.get("sog", 0.0)))
        cog = float(meas.get("cog", 0.0))
        vx = sog * np.cos(cog)
        vy = sog * np.sin(cog)

        self.x[:] = np.array([px, py, vx, vy], dtype=float)

        self.P = np.diag([
            5.0 ** 2,
            5.0 ** 2,
            1.0 ** 2,
            1.0 ** 2,
        ])

        self.last_ts = float(ts)

    # ---------- 预测 ----------
    def predict(self, dt: float):
        if dt <= 0.0:
            return
        dt = float(dt)

        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = self.std_a ** 2

        Q = np.array([
            [0.25 * q * dt4, 0.0,             0.5 * q * dt3, 0.0],
            [0.0,            0.25 * q * dt4,  0.0,           0.5 * q * dt3],
            [0.5 * q * dt3,  0.0,             q * dt2,       0.0],
            [0.0,            0.5 * q * dt3,   0.0,           q * dt2],
        ], dtype=float)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    # ---------- 更新 ----------
    def update(self, meas: dict):
        px = float(meas["x"])
        py = float(meas["y"])
        sog = max(0.0, float(meas.get("sog", 0.0)))
        cog = float(meas.get("cog", 0.0))
        vx = sog * np.cos(cog)
        vy = sog * np.sin(cog)

        z = np.array([px, py, vx, vy], dtype=float)

        H = np.eye(4, dtype=float)
        z_pred = H @ self.x
        y = z - z_pred
        S = H @ self.P @ H.T + self.R

        # ------ NIS 门控 ------
        nis = float(y.T @ np.linalg.inv(S) @ y)
        if nis > self.nis_thresh:
            # 认为是 outlier，忽略本次测量
            return

        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P

    # ---------- 对外接口 ----------
    def step(self, ts: float, meas: dict):
        ts = float(ts)
        if self.last_ts is None:
            self._init_from_measurement(ts, meas)
            return

        dt = ts - self.last_ts
        if dt > 0.0:
            self.predict(dt)

        self.update(meas)
        self.last_ts = ts

    def predict_to(self, ts: float):
        if self.last_ts is None:
            return
        ts = float(ts)
        dt = ts - self.last_ts
        if dt > 0.0:
            self.predict(dt)
            self.last_ts = ts

    def get_state(self):
        return self.x.copy()

    def get_pos(self):
        px, py, _, _ = self.x
        return float(px), float(py)

    def get_sog_cog(self):
        _, _, vx, vy = self.x
        sog = np.hypot(vx, vy)
        cog = np.arctan2(vy, vx)
        return float(sog), float(wrap_angle(cog))
