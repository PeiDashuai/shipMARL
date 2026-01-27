# ais_comms/delay_robust_ctrv_ekf.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


@dataclass
class DelayRobustCTRVConfig:
    # --------- 状态过程噪声（简单 CTRV：v, yaw 常值，只有“扰动”）---------
    process_std_a: float = 0.2       # 线加速度噪声 σ_a [m/s^2]
    process_std_yaw: float = 0.02    # 航向随机游走噪声 σ_yaw [rad/s]

    # --------- 基础测量噪声（未放大前）---------
    meas_std_pos: float = 5.0        # 位置测量噪声 σ_pos [m]，作用于 px, py
    meas_std_sog: float = 0.5        # 速度测量噪声 σ_sog [m/s]
    meas_std_cog_deg: float = 1.0    # 航向测量噪声 σ_cog [deg]

    # --------- AGE gating 参数（单位：秒）---------
    age_fresh: float = 1.0           # < age_fresh: 正常
    age_stale: float = 3.0           # [fresh, stale): 中度陈旧
    age_very_stale: float = 5.0      # [stale, very_stale): 很陈旧
    age_max: float = 8.0             # >= age_max: 直接丢弃

    # 对应各区间测量噪声放大因子（最终乘在 R 对角线上）
    scale_fresh: float = 1.0         # [0, age_fresh)
    scale_stale: float = 4.0         # [age_fresh, age_stale)
    scale_very_stale: float = 10.0   # [age_stale, age_very_stale)

    # （可选）在不同通道上使用不同缩放因子
    pos_scale_factor: float = 1.0
    vel_scale_factor: float = 1.0
    cog_scale_factor: float = 1.0

    # --------- NIS gating ----------
    nis_thresh: Optional[float] = 25.0  # None 关闭；否则 χ² 阈值


class DelayRobustCTRV_EKF:
    """
    延迟鲁棒型 CTRV-EKF（简化版）：

      - 状态: x = [px, py, v, yaw]
      - 观测: z = [px, py, v, yaw]

    与 Simple CTRV-EKF 的区别：
      * predict() 形式完全一致；
      * 在 step_delay_robust() 中，根据量测的 age 调整测量噪声 R(age)，
        age 过大时可以直接丢弃该量测。
    """

    def __init__(self, cfg: DelayRobustCTRVConfig):
        self.cfg = cfg

        # 状态向量与协方差
        self.x = np.zeros(4, dtype=float)          # [px, py, v, yaw]
        self.P = np.eye(4, dtype=float) * 1e6      # 初始协方差给很大

        self.last_ts: Optional[float] = None       # 记录“上一次状态时间”
        self.initialized: bool = False

        # 预构建基础测量噪声协方差 R0
        self.R0 = self._build_base_R()

    # --------------------------------------------------------------
    # 基础测量噪声
    # --------------------------------------------------------------
    def _build_base_R(self) -> np.ndarray:
        """构造 age=0 时的基础测量协方差 R0。"""
        c = self.cfg
        R = np.zeros((4, 4), dtype=float)
        R[0, 0] = c.meas_std_pos ** 2
        R[1, 1] = c.meas_std_pos ** 2
        R[2, 2] = c.meas_std_sog ** 2
        R[3, 3] = math.radians(c.meas_std_cog_deg) ** 2
        return R

    # --------------------------------------------------------------
    # AGE → R_eff(age) 的映射
    # --------------------------------------------------------------
    def _age_to_R(self, age: float) -> Optional[np.ndarray]:
        """
        根据 age 构造有效测量噪声协方差 R(age)。
        返回 None 表示应该丢弃该量测。
        """
        c = self.cfg
        age = float(max(age, 0.0))

        if age >= c.age_max:
            # 太老，直接拒绝本次更新
            return None

        # 选取缩放因子 s
        if age < c.age_fresh:
            s = c.scale_fresh
        elif age < c.age_stale:
            s = c.scale_stale
        elif age < c.age_very_stale:
            s = c.scale_very_stale
        else:
            # [very_stale, age_max) 再额外放大一点（线性插值）
            extra = (age - c.age_very_stale) / max(c.age_max - c.age_very_stale, 1e-6)
            s = c.scale_very_stale * (1.0 + 4.0 * extra)   # 最多再 *4

        # 在不同通道上应用不同的放大系数
        s_pos = s * c.pos_scale_factor
        s_vel = s * c.vel_scale_factor
        s_cog = s * c.cog_scale_factor

        R_eff = np.zeros_like(self.R0)
        R_eff[0, 0] = self.R0[0, 0] * s_pos
        R_eff[1, 1] = self.R0[1, 1] * s_pos
        R_eff[2, 2] = self.R0[2, 2] * s_vel
        R_eff[3, 3] = self.R0[3, 3] * s_cog
        return R_eff

    # --------------------------------------------------------------
    # 初始化：用第一条（或均值后的）测量
    # --------------------------------------------------------------
    def init_from_meas(self, ts: float, meas: dict):
        """
        meas: dict(x, y, sog, cog)
        """
        px = float(meas["x"])
        py = float(meas["y"])
        v = float(meas["sog"])
        yaw = float(meas["cog"])

        self.x[:] = np.array([px, py, v, yaw], dtype=float)

        # 初始协方差：位置相对可信，速度/航向稍微放大一些
        pos_var = self.cfg.meas_std_pos ** 2
        v_var = (self.cfg.meas_std_sog * 2.0) ** 2
        yaw_var = (math.radians(self.cfg.meas_std_cog_deg) * 2.0) ** 2

        self.P = np.diag([pos_var, pos_var, v_var, yaw_var])

        self.last_ts = float(ts)
        self.initialized = True

    # --------------------------------------------------------------
    # CTRV 预测（简化版：v, yaw 常值）
    # --------------------------------------------------------------
    def predict(self, dt: float):
        """
        dt: 从 last_ts 推进到新的时间的间隔（秒）
        """
        if not self.initialized or dt <= 0.0:
            return

        c = self.cfg
        px, py, v, yaw = self.x

        # 状态预测
        px_p = px + v * math.cos(yaw) * dt
        py_p = py + v * math.sin(yaw) * dt
        v_p = v
        yaw_p = yaw

        self.x[:] = np.array([px_p, py_p, v_p, yaw_p], dtype=float)

        # 过程噪声 Q（CV 模型近似）
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        qa2 = c.process_std_a ** 2
        qy2 = c.process_std_yaw ** 2

        Q = np.zeros((4, 4), dtype=float)

        # 参照常速模型的常见近似：把加速度噪声投影到 px, py, v 上
        Q[0, 0] = 0.25 * dt4 * qa2
        Q[1, 1] = 0.25 * dt4 * qa2
        Q[0, 2] = 0.5 * dt3 * qa2
        Q[2, 0] = 0.5 * dt3 * qa2
        Q[1, 2] = 0.5 * dt3 * qa2
        Q[2, 1] = 0.5 * dt3 * qa2
        Q[2, 2] = dt2 * qa2

        # yaw 的随机游走噪声
        Q[3, 3] = dt2 * qy2

        # 线性化状态转移 Jacobian F（对 x 的一阶偏导）
        F = np.eye(4, dtype=float)
        F[0, 2] = math.cos(yaw) * dt
        F[1, 2] = math.sin(yaw) * dt
        F[0, 3] = -v * math.sin(yaw) * dt
        F[1, 3] =  v * math.cos(yaw) * dt

        self.P = F @ self.P @ F.T + Q
        self.last_ts = (self.last_ts or 0.0) + dt

    # --------------------------------------------------------------
    # 测量更新（给定 R）
    # --------------------------------------------------------------
    def _update(self, z: np.ndarray, R: np.ndarray):
        """
        经典 EKF 更新，观测模型：
            z = [px, py, v, yaw]
        """
        px, py, v, yaw = self.x
        z_pred = np.array([px, py, v, yaw], dtype=float)

        # 观测矩阵 H：直接观测各个状态分量
        H = np.eye(4, dtype=float)

        # 创新
        y = z - z_pred
        # yaw 创新 wrap 到 [-pi, pi]
        y[3] = (y[3] + math.pi) % (2.0 * math.pi) - math.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # NIS gating（可选）
        if self.cfg.nis_thresh is not None:
            nis = float(y.T @ np.linalg.inv(S) @ y)
            if nis > self.cfg.nis_thresh:
                # 视为异常量测：丢弃这次更新
                return

        self.x = self.x + K @ y
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P

    # --------------------------------------------------------------
    # 主接口：延迟鲁棒的 step
    # --------------------------------------------------------------
    def step_delay_robust(self, meas_ts: float, meas: dict, age: float):
        """
        meas_ts: 该 AIS 报文的 measurement time（在当前仿真中一般用 arrival_time）
        age: 在使用该量测时的“陈旧度”，通常 = t_env - meas_ts，>=0
        meas: dict(x, y, sog, cog)
        """
        if not self.initialized:
            # 首次调用，直接用量测初始化
            self.init_from_meas(meas_ts, meas)
            return

        # 1) 先把状态从上一次时间预测到 meas_ts
        if self.last_ts is None:
            self.last_ts = meas_ts
        dt = float(meas_ts - self.last_ts)
        if dt > 0.0:
            self.predict(dt)

        # 2) 根据 age 构造 R(age)；若 None 则拒绝更新
        R_eff = self._age_to_R(age)
        if R_eff is None:
            # measurement 太老：只做 predict，不做 update
            return

        # 3) 构造测量向量并做 EKF 更新
        z = np.array(
            [
                float(meas["x"]),
                float(meas["y"]),
                float(meas["sog"]),
                float(meas["cog"]),
            ],
            dtype=float,
        )
        self._update(z, R_eff)
        # self.last_ts 已在 predict 中更新为 meas_ts
