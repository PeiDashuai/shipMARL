# miniship/reward/dual_manager.py

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import copy
from .lagrangian import (
    DualState,
    combine_reward,
    dual_update_end_of_episode,
    dual_govern_by_performance,   # 新增
)

class DualManager:
    """
     拉格朗日对偶 + Governor 自适应控制模块
 
    与 scripts/train_rllib_ppo_pf.py 对齐约定：
      - 训练侧 driver 维护 snapshot: {"dual_version","lam","running_c"}（lam only near/rule）
      - 环境侧 DualManager 在 external_driver=True 时，只接收 snapshot，不在 on_episode_end 内部更新

     职责：
     - 内部维护 DualState（λ、η、ctarget、EMA 等）
     - 提供 combine_reward() 用于合成拉格朗日奖励
     - 提供 on_episode_end() 在每局结束时更新 λ 与 EMA
     - 提供 get_snapshot()/set_snapshot() 做 JSON 可序列化快照
    """

    def __init__(self, cfg: dict):
        # ===== 1. DualState 初始化 =====
        lam0 = cfg.get("lambda_init", {"near": 3.0, "rule": 2.0, "coll": 20.0, "time": 1.0})
        dual_eta_cfg = cfg.get("dual_eta", {"near": 0.02, "rule": 0.02, "time": 0.02})
        ctarget_cfg = cfg.get("ctarget", {"near": 0.05, "rule": 0.02, "time": 0.08})

        # 外部驱动模式：dual 只接收 snapshot，不在 env 内部更新/冻结
        # 为了与你的 train_rllib_ppo_pf.py 目标一致，这里默认开启外部驱动；
        # 若你要回到“env 内部对偶更新”，显式设置 dual_external_driver=False。
        self.external_driver = bool(cfg.get("dual_external_driver", True))


        self.dual = DualState(
            lam_near=float(lam0.get("near", 3.0)),
            lam_rule=float(lam0.get("rule", 2.0)),
            # 你原代码中 lam_coll/eta_coll/ctarget_coll 目前未使用，可以先置 0
            lam_coll=0.0,
            lam_time=float(lam0.get("time", 1.0)),

            eta_near=float(dual_eta_cfg.get("near", 0.02)),
            eta_rule=float(dual_eta_cfg.get("rule", 0.02)),
            eta_coll=0.0,
            eta_time=float(dual_eta_cfg.get("time", 0.02)),

            beta=float(cfg.get("dual_beta", 0.95)),

            ctarget_near=float(ctarget_cfg.get("near", 0.06)),
            ctarget_rule=float(ctarget_cfg.get("rule", 0.02)),
            ctarget_coll=0.0001,
            ctarget_time=float(ctarget_cfg.get("time", 0.08)),
        )

        # ===== 2. Governor 配置（可从 cfg 中覆盖）=====
        self._gov_band = cfg.get("gov_band", {
            "tout": (0.20, 0.35),  # 期望 timeout 率区间
            "coll": (0.04, 0.10),  # 期望 collision 率区间
            "rule": (0.02, 0.08),  # 期望规则违规率区间
        })

        self._gov_eta = cfg.get("gov_eta", {
            "tout": 0.02,
            "coll": 0.02,
            "rule": 0.01,
        })

        self._gov_clamp = cfg.get("gov_clamp", {
            "near": (2.0, 3.5),
            "rule": (0.8, 3.0),
            "coll": (15.0, 22.0),
            # 原来是 (1.0, 3.0)，改小：
            "time": (0.4, 1.8),
        })

        self._gov_W = int(cfg.get("gov_W", 64))
        self._gov_beta = float(cfg.get("gov_beta", 0.9))
        self._gov_cooldown = int(cfg.get("gov_cooldown", 8))

        # Ema 缓存 & 滑窗缓存
        self._dual_buf = []  # 最近 W 个 episode 的 {"tout","coll","rule"}
        self._dual_ema: Dict[str, Optional[float]] = {"tout": None, "coll": None, "rule": None}
        self._dual_cool = 0  # 冷却计数器

        # ===== 3. Time-rate 控制器配置 =====
        self._tout_hist = []
        self._time_W = int(cfg.get("time_W", 64))

        # 原来: 0.005，相当于强制超时率 < 0.5%，太严了
        self._time_tau_target = float(cfg.get("time_tau_target", 0.02))
        # ↑ 目标超时率放宽到 2%，在 AIS 噪声下更现实，也减少频繁“加压”

        # 把增益整体调小
        self._time_up_gain = float(cfg.get("time_up_gain", 0.02))  # 原来 0.05
        self._time_dn_gain = float(cfg.get("time_dn_gain", 0.01))  # 原来 0.02

        # 容差区间适当调宽，让小波动不触发大动作
        self._time_tol_hi = float(cfg.get("time_tol_hi", 1.5))     # 原来 1.25
        self._time_tol_lo = float(cfg.get("time_tol_lo", 0.75))    # 保持或略调，问题不大

        # 给 DualState 留一个 version 字段（可选），方便排查同步问题
        if not hasattr(self.dual, "version"):
            self.dual.version = 0
        if not hasattr(self.dual, "eta_scale"):
            # 你的 DualState 里如果有这个字段，就用它；否则这里兼容一下
            self.dual.eta_scale = 0.33

        # 仅用于 debug/审计：记录最近一次 episode 末的观测（不用于更新）
        self._last_episode_obs = None


        # 如果 env_config 直接注入了 dual_snapshot，则优先按该 snapshot 初始化，
        # 同时强制 external_driver=True（避免 episode_end 再改写）。
        snap0 = cfg.get("dual_snapshot", None)
        if isinstance(snap0, dict) and snap0:
            self.external_driver = True
            self.set_snapshot(snap0)

    # ------------------------------------------------------------------
    # 对外接口：reward 合成 & episode 结束更新
    # ------------------------------------------------------------------
    def combine_reward(
        self,
        r_task: np.ndarray,
        c_near: np.ndarray,
        c_rule: np.ndarray,
        c_coll: np.ndarray,
        c_time: np.ndarray,
        r_clip: float,
    ) -> np.ndarray:
        """
        合成拉格朗日奖励。直接调用你原来的 combine_reward。
        """
        return combine_reward(
            r_task,
            c_near,
            c_rule,
            c_coll,
            c_time,
            self.dual,
            clip=r_clip,
        )

    def on_episode_end(
        self,
        c_near_mean: float,
        c_rule_mean: float,
        c_coll_max: float,
        c_time_max: float,
        succ_ep: float,
        coll_ep: float,
        tout_ep: float,
        rule_rate_ep: Optional[float] = None,
    ) -> None:
        """
        每个 episode 结束时调用：
        1) 先做 dual_update_end_of_episode（按均值/最大值更新 λ）
        2) 再交给窗口治理 _dual_window_govern 做微调+限幅
        """

        # 外部驱动：绝不在 env 内部更新 dual（λ/EMA/governor/time controller 全部禁止）
        if self.external_driver:
            if rule_rate_ep is None:
                rule_rate_ep = 1.0 if float(c_rule_mean) > 0.0 else 0.0
            self._last_episode_obs = {
                "c_near_mean": float(c_near_mean),
                "c_rule_mean": float(c_rule_mean),
                "c_coll_max": float(c_coll_max),
                "c_time_max": float(c_time_max),
                "succ_ep": float(succ_ep),
                "coll_ep": float(coll_ep),
                "tout_ep": float(tout_ep),
                "rule_rate_ep": float(rule_rate_ep),
                "dual_version": int(getattr(self.dual, "version", 0)),
                "lam": {
                    "near": float(self.dual.lam_near),
                    "rule": float(self.dual.lam_rule),
                    "coll": float(self.dual.lam_coll),
                    "time": float(self.dual.lam_time),
                },
                "running_c": {
                    "near": float(getattr(self.dual, "ema_near", 0.0) or 0.0),
                    "rule": float(getattr(self.dual, "ema_rule", 0.0) or 0.0),
                    "coll": float(getattr(self.dual, "ema_coll", 0.0) or 0.0),
                    "time": float(getattr(self.dual, "ema_time", 0.0) or 0.0),
                },
            }
            return

        # 记录调用前的 λ（用于 governor “回滚”）
        lam_prev = {
            "near": float(self.dual.lam_near),
            "rule": float(self.dual.lam_rule),
            "coll": float(self.dual.lam_coll),
            "time": float(self.dual.lam_time),
        }

        # 标准 dual update
        dual_update_end_of_episode(
            float(c_near_mean),
            float(c_rule_mean),
            float(c_coll_max),
            float(c_time_max),
            self.dual,
        )

        # rule_rate_ep 默认为 “本集平均 c_rule 是否>0”
        if rule_rate_ep is None:
            rule_rate_ep = 1.0 if float(c_rule_mean) > 0.0 else 0.0

        # Governor 窗口治理 + timeout 自适应控制
        self._dual_window_govern(
            succ_ep=float(succ_ep),
            coll_ep=float(coll_ep),
            tout_ep=float(tout_ep),
            rule_rate_ep=float(rule_rate_ep),
            lam_prev=lam_prev,
        )
        # ★ 新增：基于整体性能的“精修/退火”层
        dual_govern_by_performance(
            self.dual,
            succ_ep=float(succ_ep),
            coll_ep=float(coll_ep),
            tout_ep=float(tout_ep),
        )
    # ------------------------------------------------------------------
    # Snapshot 接口（给 RLlib 或外部管理器用）
    # ------------------------------------------------------------------
    def get_snapshot(self) -> dict:
        """
        提取对偶/治理层的“纯数值快照”，保证 JSON 可序列化，且向后兼容。

        与 train_rllib_ppo_pf.py 对齐的主结构（推荐使用）：
          {
            "dual_version": int,
            "lam": {"near","rule","coll","time"},
            "running_c": {"near","rule","coll","time"}
          }

        同时保留旧的扁平字段（lam_near/ema_near/...），避免破坏旧代码。
        """
        d = self.dual

        def _f(name: str, default=None):
            try:
                v = getattr(d, name)
            except Exception:
                return default
            return None if v is None else float(v)

        ver = int(getattr(d, "version", 0))
        lam_near = _f("lam_near", 3.0)
        lam_rule = _f("lam_rule", 2.0)
        lam_coll = _f("lam_coll", 0.0)
        lam_time = _f("lam_time", 1.0)
        ema_near = _f("ema_near", 0.0)
        ema_rule = _f("ema_rule", 0.0)
        ema_coll = _f("ema_coll", 0.0)
        ema_time = _f("ema_time", 0.0)

        snap = {
            # === train_rllib_ppo_pf.py canonical keys ===
            "dual_version": ver,
            "lam": {
                "near": lam_near,
                "rule": lam_rule,
                "coll": lam_coll,
                "time": lam_time,
            },
            "running_c": {
                "near": ema_near,
                "rule": ema_rule,
                "coll": ema_coll,
                "time": ema_time,
            },

            # === legacy flat keys (backward compatible) ===
            "lam_near": lam_near,
            "lam_rule": lam_rule,
            "lam_coll": lam_coll,
            "lam_time": lam_time,

            "ema_near": ema_near,
            "ema_rule": ema_rule,
            "ema_coll": ema_coll,
            "ema_time": ema_time,

            "ema_succ": _f("ema_succ", 0.0),
            "ema_tout": _f("ema_tout", 1.0),
            "eta_scale": _f("eta_scale", 0.33),

            "version": ver,
            "external_driver": bool(self.external_driver),
        }
        return snap

    def set_snapshot(self, snap: dict) -> None:
        """
        用快照覆盖本地对偶状态。任何缺失字段均跳过，确保多版本兼容。
        """
        if not snap:
            return
        d = self.dual

        # --- 1) 兼容 train_rllib_ppo_pf.py snapshot 结构 ---
        #   {"dual_version": int, "lam":{near,rule}, "running_c":{near,rule}}
        if isinstance(snap.get("lam", None), dict):
            lam = snap.get("lam") or {}
            if "near" in lam:
                try: d.lam_near = float(lam["near"])
                except Exception: pass
            if "rule" in lam:
                try: d.lam_rule = float(lam["rule"])
                except Exception: pass
            # 允许带 coll/time（即便训练侧通常不带）
            if "coll" in lam:
                try: d.lam_coll = float(lam["coll"])
                except Exception: pass
            if "time" in lam:
                try: d.lam_time = float(lam["time"])
                except Exception: pass

        if isinstance(snap.get("running_c", None), dict):
            rc = snap.get("running_c") or {}
            # running_c 与 DualState 的 EMA 语义对齐
            if "near" in rc:
                try: d.ema_near = float(rc["near"])
                except Exception: pass
            if "rule" in rc:
                try: d.ema_rule = float(rc["rule"])
                except Exception: pass
            if "coll" in rc:
                try: d.ema_coll = float(rc["coll"])
                except Exception: pass
            if "time" in rc:
                try: d.ema_time = float(rc["time"])
                except Exception: pass

        def _set(name: str, cast=float):
            if name in snap and snap[name] is not None:
                try:
                    setattr(d, name, cast(snap[name]))
                except Exception:
                    pass

        # --- 2) legacy flat keys ---
        # lambdas (flat)
        _set("lam_near")
        _set("lam_rule")
        _set("lam_coll")
        _set("lam_time")

        # EMAs
        _set("ema_near")
        _set("ema_rule")
        _set("ema_coll")
        _set("ema_time")

        # governor EMAs
        _set("ema_succ")
        _set("ema_tout")

        # knobs
        _set("eta_scale")

        # train_rllib_ppo_pf.py uses dual_version
        if "dual_version" in snap:
            try:
                d.version = int(snap["dual_version"])
            except Exception:
                # 外部驱动：不擅自推进版本号
                if not self.external_driver:
                    d.version = getattr(d, "version", 0) + 1
        elif "version" in snap:
            try:
                d.version = int(snap["version"])
            except Exception:
                if not self.external_driver:
                    d.version = getattr(d, "version", 0) + 1
        else:
            # 外部驱动：没有版本号就不动；非外部驱动：保持旧行为
            if not self.external_driver:
                d.version = getattr(d, "version", 0) + 1

    # ------------------------------------------------------------------
    # 与训练脚本命名对齐的别名（env 侧通常直接调用 set_dual/get_dual_snapshot）
    # ------------------------------------------------------------------
    def get_dual_snapshot(self) -> dict:
        return self.get_snapshot()

    def set_dual(self, snap: dict) -> None:
        self.set_snapshot(snap)

    # ------------------------------------------------------------------
    # 一些便捷访问接口
    # ------------------------------------------------------------------
    def get_lambdas(self) -> Dict[str, float]:
        """
        返回当前 λ 的字典，用于写入 infos 或调试。
        """
        return {
            "near": float(self.dual.lam_near),
            "rule": float(self.dual.lam_rule),
            "coll": float(self.dual.lam_coll),
            "time": float(self.dual.lam_time),
        }

    # ------------------------------------------------------------------
    # 内部：timeout 自适应控制器 + 窗口治理
    # ------------------------------------------------------------------
    def _time_rate_controller(self, tout_ep: float) -> None:
        """
        对 lam_time 的超时率自适应控制器：
        - 始终工作（冷却/带内/越带之后都可调），与窗口治理互补
        - 目标：把 timeout 率拉向 τ_tgt；调节幅度很小
        """
        # 记录本集 0/1 超时标记
        self._tout_hist.append(float(tout_ep))
        if len(self._tout_hist) > self._time_W:
            self._tout_hist.pop(0)

        tout_rate = float(np.mean(self._tout_hist))

        tau_tgt = self._time_tau_target
        up_gain = self._time_up_gain
        dn_gain = self._time_dn_gain
        tol_hi = self._time_tol_hi
        tol_lo = self._time_tol_lo

        lo, hi = self._gov_clamp["time"]
        lam_t = float(self.dual.lam_time)

        if tout_rate > tau_tgt * tol_hi:
            lam_t = lam_t * (1.0 + up_gain)
        elif tout_rate < tau_tgt * tol_lo:
            lam_t = lam_t * (1.0 - dn_gain)

        self.dual.lam_time = float(np.clip(lam_t, lo, hi))

    def _dual_window_govern(
        self,
        succ_ep: float,
        coll_ep: float,
        tout_ep: float,
        rule_rate_ep: float,
        lam_prev: Dict[str, float],
    ) -> str:
        """
        Episode 级治理：
        - 维护滑窗&EMA（tout/coll/rule）
        - 在目标带内：near/rule/coll 回滚+进入冷却，但不回滚 time；
          同时总是运行“超时率自适应控制器”微调 lam_time
        - 越带：按小步从 lam_prev 微调（含 time），之后仍运行“超时率控制器”作细修
        """
        # A) 滑窗 & EMA 维护
        entry = {
            "tout": float(tout_ep),
            "coll": float(coll_ep),
            "rule": float(rule_rate_ep),
        }
        self._dual_buf.append(entry)
        if len(self._dual_buf) > self._gov_W:
            self._dual_buf.pop(0)

        wins = {
            k: float(np.mean([x[k] for x in self._dual_buf]))
            for k in ["tout", "coll", "rule"]
        }

        for k in wins:
            m = self._dual_ema[k]
            self._dual_ema[k] = (
                self._gov_beta * m + (1.0 - self._gov_beta) * wins[k]
            ) if m is not None else wins[k]

        ema = self._dual_ema
        band = self._gov_band

        # B) 冷却期：near 夹持 + lam_time 控制器，立即返回
        if self._dual_cool > 0:
            self._dual_cool -= 1
            self.dual.lam_near = float(
                np.clip(self.dual.lam_near, *self._gov_clamp["near"])
            )
            self._time_rate_controller(tout_ep)
            return "cooldown"

        # C) 检查是否越带
        flagged = []
        for k, (lo, hi) in band.items():
            v = ema[k]
            if v < lo * 0.98:
                flagged.append((k, -1, (lo - v) / max(1e-6, lo)))
            elif v > hi * 1.02:
                flagged.append((k, +1, (v - hi) / max(1e-6, hi)))

        # D) 带内：near/rule/coll 回滚 + 冷却；time 不回滚
        if not flagged:
            self.dual.lam_near = lam_prev["near"]
            self.dual.lam_rule = lam_prev["rule"]
            self.dual.lam_coll = lam_prev["coll"]
            self._dual_cool = self._gov_cooldown

            self._time_rate_controller(tout_ep)
            return "freeze"

        # E) 越带：从 lam_prev 小步微调
        def _step(name_lambda: str, k_key: str, sgn: int, mag: float) -> float:
            step = self._gov_eta[k_key] * float(np.clip(mag, 0.0, 1.0))
            newv = lam_prev[name_lambda] + sgn * step
            lo, hi = self._gov_clamp[name_lambda]
            return float(np.clip(newv, lo, hi))

        for k_key, sgn, mag in flagged:
            if k_key == "tout":
                self.dual.lam_time = _step("time", "tout", sgn, mag)
            elif k_key == "coll":
                self.dual.lam_coll = _step("coll", "coll", sgn, mag)
            elif k_key == "rule":
                self.dual.lam_rule = _step("rule", "rule", sgn, mag)

        # near 仍只限幅不主动调
        self.dual.lam_near = float(
            np.clip(lam_prev["near"], *self._gov_clamp["near"])
        )

        self._time_rate_controller(tout_ep)
        return "nudged"
