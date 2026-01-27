# miniship/risk/guard_controller.py

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np

from ..risk.tcpa_dcpa import tcpa_dcpa_matrix
from ..risk.safety_guard import (
    safety_guard_hard,
    safety_guard_ghost,
    safety_guard_hybrid,
)
# 如果以后有 strict_guard_multi，可以在这里 import


class GuardController:
    """
    护栏控制模块（不直接依赖具体 Env），只操作：
      - 输入：ships 列表、当前 dpsi_rl / v_cmd、风险阈值、时间步等
      - 输出：修正后的 dpsi_rl / v_cmd、本步护栏触发向量和其均值

    训练/评估模式、guard_mode 等开关都在这里集中管理。
    """

    def __init__(self, cfg: Dict[str, Any], N: int):
        # 训练 / 评估开关
        self.guard_train: bool = bool(cfg.get("guard_train", True))   # 训练期默认开
        self.guard_eval: bool = bool(cfg.get("guard_eval", False))    # 评估期默认关

        # 训练期护栏模式
        self.guard_mode: str = str(getattr(cfg, "guard_mode", cfg.get("guard_mode", "hybrid")))
        self.guard_warmup_steps: int = int(getattr(cfg, "guard_warmup_steps", 80))
        self.guard_hybrid_p: float = float(getattr(cfg, "guard_hybrid_p", 0.2))

        # 评估期护栏模式（可与训练期不同）
        self.guard_eval_mode: str = str(getattr(
            cfg, "guard_eval_mode", cfg.get("guard_eval_mode", "none")
        ))

        # 护栏惩罚系数（给 reward 用）
        self.lam_guard: float = float(getattr(cfg, "lam_guard", cfg.get("lam_guard", 2.0)))

        # 用于统计整个 episode 的护栏触发率
        self.N = int(N)
        self._guard_trig_sum: float = 0.0
        self._guard_trig_steps: int = 0

    # ----------------- episode 级接口 -----------------

    def reset_episode_stats(self) -> None:
        """每个 episode 开始时重置护栏统计计数器。"""
        self._guard_trig_sum = 0.0
        self._guard_trig_steps = 0

    def get_episode_guard_rate(self) -> float:
        """返回整局平均护栏触发强度（对 N 条船取均值、再对步数取均值）。"""
        return self._guard_trig_sum / max(1, self._guard_trig_steps)

    # ----------------- 单步护栏应用 -----------------

    def apply(
        self,
        ships,
        dpsi_rl: np.ndarray,
        v_cmd: np.ndarray,
        dt: float,
        dpsi_max: float,
        v_min: float,
        v_max: float,
        risk_T_thr: float,
        risk_D_thr: float,
        collide_thr: float,
        step_idx: int,
        eval_mode: bool,
        yaw_rate_max=None,
        brake_gain: float = 0.4,
        steer_gain: float = 0.4,
        K_guard: int = 4,
        use_float32: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """
        单步护栏处理。

        返回：
          - dpsi_out: 修正后的转向命令
          - v_cmd_out: 修正后的速度命令
          - guard_trig: shape=(N,) 的触发强度向量
          - guard_step_mean: 本步对 N 条船触发强度的均值
          - guard_enabled: 本步是否实际启用了护栏
        """
        # 默认：不改变动作、不触发护栏
        N = self.N
        dpsi_out = np.asarray(dpsi_rl, dtype=np.float32 if use_float32 else np.float64)
        v_cmd_out = np.asarray(v_cmd, dtype=np.float32 if use_float32 else np.float64)
        guard_trig = np.zeros(N, dtype=np.float64)

        # 1) 决定当前是否启用护栏
        use_guard = (self.guard_eval if eval_mode else self.guard_train)
        if not use_guard:
            guard_step_mean = 0.0
            # 不启用护栏，直接返回原动作
            return dpsi_out, v_cmd_out, guard_trig, guard_step_mean, False

        # 2) 计算推进前风险（tc0/dc0）
        tc0, dc0, _ = tcpa_dcpa_matrix(ships)

        guard_kwargs = dict(
            ships=ships,
            dpsi=dpsi_out,
            v_cmd=v_cmd_out,
            dt=dt,
            dpsi_max=dpsi_max,
            v_min=v_min,
            v_max=v_max,
            risk_T_thr=risk_T_thr,
            risk_D_thr=risk_D_thr,
            collide_thr=collide_thr,
            tc=tc0,
            dc=dc0,
            yaw_rate_max=yaw_rate_max,
            brake_gain=brake_gain,
            steer_gain=steer_gain,
            K_guard=K_guard,
            use_float32=use_float32,
        )

        # 3) 根据训练/评估模式选择 guard 模式

        if eval_mode:
            # 评估期：根据 guard_eval_mode 决定是否/如何应用护栏
            mode = (self.guard_eval_mode or "none").lower()
            if not self.guard_eval or mode == "none":
                guard_step_mean = 0.0
                guard_enabled = False
            elif mode == "ghost":
                dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_ghost(**guard_kwargs)
                guard_enabled = True
            elif mode == "hard":
                dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_hard(**guard_kwargs)
                guard_enabled = True
            elif mode == "strict":
                # 如果你有 strict_guard_multi，可以替换这里
                # dpsi_out, v_cmd_out, guard_trig = strict_guard_multi(**guard_kwargs)
                guard_trig = np.zeros(N, dtype=np.float64)
                guard_enabled = True
            else:
                guard_step_mean = 0.0
                guard_enabled = False
        else:
            # 训练期：沿用原 hard/ghost/hybrid 逻辑
            mode = (self.guard_mode or "hard").lower()
            warmup = int(getattr(self, "guard_warmup_steps", 80))
            hybrid_p = float(getattr(self, "guard_hybrid_p", 0.5))

            if mode == "hard":
                dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_hard(**guard_kwargs)
                guard_enabled = True
            elif mode == "ghost":
                dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_ghost(**guard_kwargs)
                guard_enabled = True
            elif mode == "hybrid":
                if step_idx < warmup:
                    dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_hard(**guard_kwargs)
                else:
                    dpsi_out, v_cmd_out, guard_trig, _ = safety_guard_hybrid(
                        **guard_kwargs,
                        mode_prob=float(hybrid_p),
                    )
                guard_enabled = True
            else:
                guard_enabled = False

        # 4) 统计本步护栏触发均值
        guard_trig = np.asarray(guard_trig, dtype=np.float64)
        guard_step_mean = float(np.mean(guard_trig)) if guard_trig is not None else 0.0

        self._guard_trig_sum += guard_step_mean
        self._guard_trig_steps += 1

        return dpsi_out, v_cmd_out, guard_trig, guard_step_mean, guard_enabled
