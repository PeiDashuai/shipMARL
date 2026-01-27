# ais_comms/ctrv_ekf.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CTRVConfig:
    """CTRV-EKF 的噪声与量测配置"""
    process_std_a: float = 0.2          # 纵向加速度噪声 σ_a  [m/s²]
    process_std_yawdd: float = 0.05     # 角加速度噪声 σ_yawdd [rad/s²]
    meas_std_pos: float = 5.0           # 位置噪声 σ_pos      [m]
    meas_std_sog: float = 0.5           # SOG 噪声 σ_sog      [m/s]
    meas_std_cog_deg: float = 1.0       # COG 噪声 σ_cog      [deg]
    nis_thresh: Optional[float] = 25.0  # NIS 门限（None 表示不用 gating）


class CTRV_EKF:
    """
    5 维 CTRV 模型扩展卡尔曼滤波器：

      x = [px, py, v, yaw, yaw_rate]^T

    量测使用 4 维：
      z = [px, py, sog, cog]^T
      sog ≈ v
      cog ≈ yaw

    - predict(dt):   按 CTRV 运动模型外推
    - update(z):     用位置 + sog/cog 融合修正
    - step(ts, meas): 先 predict 再 update
    """

    def __init__(self, cfg: CTRVConfig | None = None):
        self.cfg = cfg or CTRVConfig()

        # 状态向量与协方差
        self.x = np.zeros(5, dtype=float)
        self.P = np.eye(5, dtype=float) * 1e3  # 初始化较大不确定度

        # 时间戳
        self.last_ts: Optional[float] = None
        self.initialized: bool = False

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def init_from_meas(self, ts: float, meas: dict):
        """
        meas 要求包含：
          - 'x', 'y' : 位置 [m]
          - 'sog'    : 对地速度 [m/s]
          - 'cog'    : 航向 [rad]
        """
        px = float(meas["x"])
        py = float(meas["y"])
        v = float(meas.get("sog", 0.0))
        yaw = float(meas.get("cog", 0.0))
        yaw_rate = 0.0

        self.x[:] = [px, py, v, yaw, yaw_rate]

        # 位置给小一点协方差，速度/角度大一些
        self.P = np.diag([
            self.cfg.meas_std_pos ** 2,
            self.cfg.meas_std_pos ** 2,
            max(1.0, self.cfg.meas_std_sog ** 2),
            math.radians(self.cfg.meas_std_cog_deg) ** 2 * 5.0,
            0.1,  # yaw_rate 初始不确定度
        ])

        self.last_ts = float(ts)
        self.initialized = True

    # ------------------------------------------------------------------
    # 预测步
    # ------------------------------------------------------------------
    def predict(self, dt: float):
        if not self.initialized:
            return
        if dt <= 0.0:
            return

        px, py, v, yaw, yawd = self.x

        # 状态转移
        if abs(yawd) > 1e-4:
            px_p = px + v / yawd * (math.sin(yaw + yawd * dt) - math.sin(yaw))
            py_p = py + v / yawd * (-math.cos(yaw + yawd * dt) + math.cos(yaw))
        else:
            px_p = px + v * dt * math.cos(yaw)
            py_p = py + v * dt * math.sin(yaw)

        v_p = v
        yaw_p = yaw + yawd * dt
        yawd_p = yawd

        # 加入过程噪声项（离散化简化版）
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        qa2 = self.cfg.process_std_a ** 2
        qy2 = self.cfg.process_std_yawdd ** 2

        # 这里采用一个相对简单但足够用的 Q 结构：
        Q = np.zeros((5, 5), dtype=float)
        Q[0, 0] = 0.25 * dt4 * qa2
        Q[1, 1] = 0.25 * dt4 * qa2
        Q[0, 2] = 0.5 * dt3 * qa2
        Q[2, 0] = Q[0, 2]
        Q[1, 2] = 0.5 * dt3 * qa2
        Q[2, 1] = Q[1, 2]
        Q[2, 2] = dt2 * qa2
        Q[3, 3] = dt2 * qy2
        Q[4, 4] = dt2 * qy2

        # 雅可比 F
        F = np.eye(5, dtype=float)
        if abs(yawd) > 1e-4:
            # 对 yaw, yawd 的偏导比较复杂，这里采用数值上较稳的近似
            v_over_yawd = v / yawd
            sin_yaw = math.sin(yaw)
            cos_yaw = math.cos(yaw)
            sin_yaw_yawd_dt = math.sin(yaw + yawd * dt)
            cos_yaw_yawd_dt = math.cos(yaw + yawd * dt)

            F[0, 2] = (sin_yaw_yawd_dt - sin_yaw) / yawd
            F[1, 2] = (-cos_yaw_yawd_dt + cos_yaw) / yawd

            F[0, 3] = v_over_yawd * (cos_yaw_yawd_dt - cos_yaw)
            F[1, 3] = v_over_yawd * (sin_yaw_yawd_dt - sin_yaw)

            term = v / (yawd * yawd)
            F[0, 4] = term * (sin_yaw - sin_yaw_yawd_dt) + v_over_yawd * dt * cos_yaw_yawd_dt
            F[1, 4] = term * (cos_yaw_yawd_dt - cos_yaw) + v_over_yawd * dt * sin_yaw_yawd_dt
        else:
            F[0, 2] = dt * math.cos(yaw)
            F[1, 2] = dt * math.sin(yaw)
            F[0, 3] = -v * dt * math.sin(yaw)
            F[1, 3] = v * dt * math.cos(yaw)

        # 更新
        self.x[:] = [px_p, py_p, v_p, yaw_p, yawd_p]
        self.P = F @ self.P @ F.T + Q

    # ------------------------------------------------------------------
    # 更新步（融合位置 + sog/cog）
    # ------------------------------------------------------------------
    def update(self, meas: dict):
        if not self.initialized:
            return

        z = np.array([
            float(meas["x"]),
            float(meas["y"]),
            float(meas.get("sog", 0.0)),
            float(meas.get("cog", 0.0)),
        ])

        # 量测模型: h(x) = [px, py, v, yaw]
        px, py, v, yaw, _ = self.x
        h = np.array([px, py, v, yaw])

        # 雅可比 H
        H = np.zeros((4, 5), dtype=float)
        H[0, 0] = 1.0   # ∂px/∂px
        H[1, 1] = 1.0   # ∂py/∂py
        H[2, 2] = 1.0   # ∂v/∂v
        H[3, 3] = 1.0   # ∂yaw/∂yaw

        # 量测噪声协方差
        R = np.diag([
            self.cfg.meas_std_pos ** 2,
            self.cfg.meas_std_pos ** 2,
            self.cfg.meas_std_sog ** 2,
            math.radians(self.cfg.meas_std_cog_deg) ** 2,
        ])

        # 卡尔曼更新
        y = z - h  # 创新
        # 对 yaw 差值做 wrap，使其落在 [-pi, pi)
        y[3] = (y[3] + math.pi) % (2.0 * math.pi) - math.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # NIS gating（可选）
        if self.cfg.nis_thresh is not None:
            nis = float(y.T @ np.linalg.inv(S) @ y)
            if nis > self.cfg.nis_thresh:
                # 量测过于离谱，丢弃本次更新，但保留预测
                return

        self.x = self.x + K @ y
        I = np.eye(5, dtype=float)
        self.P = (I - K @ H) @ self.P

    # ------------------------------------------------------------------
    # 一步处理：给定绝对时间戳
    # ------------------------------------------------------------------
    def step(self, ts: float, meas: dict):
        if not self.initialized:
            self.init_from_meas(ts, meas)
            return

        ts = float(ts)
        dt = ts - float(self.last_ts)
        if dt > 0.0:
            self.predict(dt)

        self.update(meas)
        self.last_ts = ts
