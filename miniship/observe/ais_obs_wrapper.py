# ais_obs_wrapper.py
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from pettingzoo import ParallelEnv

from ..dynamics.ship import Ship
from .builder import build_observations, zero_observation


@dataclass
class AISTrack:
    """单条船舶的 AIS 估计状态（最近一次量测 + 推演）"""
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    psi: float = 0.0
    t_meas: float = 0.0     # 最近一次“量测时刻”（env 内部时间）
    valid: bool = False     # 是否已经有过量测


class AISObsWrapper(ParallelEnv):
    """
    将底层几何环境（例如 MiniShipCoreEnv）包装成 “AIS 观测版” 环境：

    - 动作 / 奖励 / 终止逻辑 完全沿用底层 env；
    - 仅对观测进行改写：
        * AIS 风格的慢速、带噪的量测 + dead-reckoning
        * 用估计出的 est_ships 列表调用原来的 build_observations(...)
        * 保持“每艘船的观测向量结构固定”：
              self(6) + K*neighbor_feat(8) + K*edge_feat(8) + id(1)
            即总长度 = 6 + K*16 + 1

    这样 RL 策略看到的是 “AIS 估计世界”，而 reward / guard / dual 仍然可以用几何真值。
    """

    metadata = {"render_modes": ["none"]}

    def __init__(self, env: ParallelEnv, cfg: Dict[str, Any]):
        super().__init__()
        self.env = env

        # ---------- agent & space ----------
        self.agents = list(getattr(env, "agents", []))
        self.possible_agents = list(getattr(env, "possible_agents", self.agents))
        self._lazy_agents = len(self.agents) == 0

        self._act_space = None
        if hasattr(env, "action_spaces"):
            act_spaces = env.action_spaces
            if isinstance(act_spaces, dict) and self.agents:
                self._act_space = act_spaces[self.agents[0]]

        self.K_neighbors = int(getattr(env, "K_neighbors", cfg.get("numNeighbors", 4)))
        self.spawn_mode = str(getattr(env, "spawn_mode", cfg.get("spawn_mode", "circle_center")))
        self.spawn_area = float(getattr(env, "spawn_area", cfg.get("spawn_area", 240.0)))
        self.spawn_len = float(getattr(env, "spawn_len", cfg.get("spawn_len", 180.0)))
        self.v_max = float(getattr(env, "v_max", cfg.get("v_max", 3.0)))

        # ---------- AIS 参数（更接近真实） ----------
        # 观测周期、丢包
        self.ais_period    = float(cfg.get("ais_period", 1.0))     # 1 s 一次位置更新
        self.ais_loss_prob = float(cfg.get("ais_loss_prob", 0.05)) # 5% 丢包

        # 测量噪声
        self.ais_pos_noise = float(cfg.get("ais_pos_noise", 2.0))               # [m]
        self.ais_v_noise   = float(cfg.get("ais_v_noise", 0.03))                # [m/s]
        self.ais_psi_noise = float(cfg.get("ais_psi_noise", math.radians(0.3))) # 0.3°

        # dead-reckoning 限制的最大年龄（超过就不再往前推）
        self.ais_max_age   = float(cfg.get("ais_max_age", 2.0))    # [s]

        # 一阶滤波系数（越小越平滑）
        self.ais_alpha = float(cfg.get("ais_alpha", 0.4))

        # outlier 阈值（预留，将来可加入“跳变剔除”逻辑）
        self.ais_jump_dist_thr  = float(cfg.get("ais_jump_dist_thr", 25.0)) # [m]
        self.ais_jump_speed_thr = float(cfg.get("ais_jump_speed_thr", 8.0)) # [m/s] ≈ 16 kn

        # 内部随机数（只管 AIS 的噪声/丢包）
        self._rng = np.random.default_rng(int(cfg.get("ais_seed", 12345)))

        # ship_id -> AISTrack
        self._tracks: Dict[int, AISTrack] = {}

        # 观测空间：第一次 reset 时推断
        self._obs_space = None

        print("[AISObsWrapper] loaded from", __file__)

    # ------------------------------------------------------------------
    # PettingZoo ParallelEnv 接口
    # ------------------------------------------------------------------

    @property
    def observation_spaces(self):
        if self._obs_space is None:
            if hasattr(self.env, "observation_spaces") and self.agents:
                base_space = self.env.observation_spaces[self.agents[0]]
                self._obs_space = base_space
            else:
                raise RuntimeError(
                    "AISObsWrapper._obs_space is not initialized; call reset() once before accessing observation_spaces."
                )
        return {aid: self._obs_space for aid in self.agents}

    @property
    def action_spaces(self):
        if self._act_space is None and hasattr(self.env, "action_spaces") and self.agents:
            self._act_space = self.env.action_spaces[self.agents[0]]
        return {aid: self._act_space for aid in self.agents}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def seed(self, seed: int | None = None):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)
        if seed is not None:
            self._rng = np.random.default_rng(int(seed) + 1000)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, options: Dict[str, Any] | None = None):
        obs_core, infos = self.env.reset(seed=seed, options=options)

        if self._lazy_agents:
            self.agents = list(obs_core.keys())
            self.possible_agents = list(self.agents)
            self._lazy_agents = False

        self._init_tracks(force_first_measure=True)
        obs_ais = self._build_ais_observations()

        if self._obs_space is None:
            from gymnasium.spaces import Box
            sample = obs_ais[self.agents[0]]
            self._obs_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=sample.shape,
                dtype=np.float32,
            )

        return obs_ais, infos

    def step(self, actions: Dict[str, Any]):
        obs_core, rewards, terminations, truncations, infos = self.env.step(actions)

        done_global = all(terminations.values()) or all(truncations.values())
        if done_global:
            return obs_core, rewards, terminations, truncations, infos

        self._update_tracks()
        obs_ais = self._build_ais_observations()

        max_age, vis_cnt = self._debug_ais_stats()
        for aid in self.agents:
            info = infos.get(aid, {})
            info["ais_max_age"] = float(max_age)
            info["ais_visible_cnt"] = int(vis_cnt)
            infos[aid] = info

        return obs_ais, rewards, terminations, truncations, infos

    # ------------------------------------------------------------------
    # AIS track 管理与观测构造
    # ------------------------------------------------------------------
    def _get_world_state(self):
        state = getattr(self.env, "state", None)
        if state is None or not hasattr(state, "ships"):
            raise RuntimeError("Underlying env must expose .state with .ships list.")
        return state

    def _init_tracks(self, force_first_measure: bool = True):
        state = self._get_world_state()
        ships = state.ships
        t = float(getattr(state, "t", 0.0))

        self._tracks = {}
        for idx, ship in enumerate(ships):
            sid_attr = getattr(ship, "id", None)
            sid = int(sid_attr) if sid_attr is not None else (idx + 1)

            tr = AISTrack()
            if force_first_measure:
                tr.x = float(ship.pos[0])
                tr.y = float(ship.pos[1])
                tr.v = float(ship.v)
                tr.psi = self._wrap_angle(float(ship.psi))
                tr.t_meas = t
                tr.valid = True

            self._tracks[sid] = tr

    def _update_tracks(self):
        state = self._get_world_state()
        ships = state.ships
        t = float(getattr(state, "t", 0.0))

        for idx, ship in enumerate(ships):
            sid_attr = getattr(ship, "id", None)
            sid = int(sid_attr) if sid_attr is not None else (idx + 1)

            if sid not in self._tracks:
                self._tracks[sid] = AISTrack()
            tr = self._tracks[sid]

            age = t - tr.t_meas if tr.valid else float("inf")
            need_try_measure = (not tr.valid) or (age >= self.ais_period)
            if not need_try_measure:
                continue

            # 丢包
            if self._rng.random() < self.ais_loss_prob:
                continue

            # 真值 + 噪声
            x_meas = float(ship.pos[0]) + self._rng.normal(0.0, self.ais_pos_noise)
            y_meas = float(ship.pos[1]) + self._rng.normal(0.0, self.ais_pos_noise)
            v_meas = float(ship.v)      + self._rng.normal(0.0, self.ais_v_noise)
            psi_meas = float(ship.psi)  + self._rng.normal(0.0, self.ais_psi_noise)
            psi_meas = self._wrap_angle(psi_meas)

            alpha = self.ais_alpha
            if tr.valid:
                tr.x = (1.0 - alpha) * tr.x + alpha * x_meas
                tr.y = (1.0 - alpha) * tr.y + alpha * y_meas
                tr.v = (1.0 - alpha) * tr.v + alpha * max(0.0, v_meas)

                c1, s1 = math.cos(tr.psi), math.sin(tr.psi)
                c2, s2 = math.cos(psi_meas), math.sin(psi_meas)
                c = (1.0 - alpha) * c1 + alpha * c2
                s = (1.0 - alpha) * s1 + alpha * s2
                tr.psi = math.atan2(s, c)
            else:
                tr.x = x_meas
                tr.y = y_meas
                tr.v = max(0.0, v_meas)
                tr.psi = psi_meas

            tr.t_meas = t
            tr.valid = True

    def _build_ais_observations(self) -> Dict[str, np.ndarray]:
        state = self._get_world_state()
        ships_true = state.ships
        t = float(getattr(state, "t", 0.0))

        est_ships: list[Ship] = []
        for ship in ships_true:
            sid_attr = getattr(ship, "id", None)
            sid = int(sid_attr) if sid_attr is not None else (len(est_ships) + 1)
            tr = self._tracks.get(sid, None)

            if tr is None or (not tr.valid):
                x_est = float(ship.pos[0])
                y_est = float(ship.pos[1])
                v_est = float(ship.v)
                psi_est = float(ship.psi)
            else:
                age = max(0.0, t - tr.t_meas)
                dt = min(age, self.ais_max_age)

                x_est = tr.x + tr.v * math.cos(tr.psi) * dt
                y_est = tr.y + tr.v * math.sin(tr.psi) * dt
                v_est = tr.v
                psi_est = tr.psi

            est_ship = Ship(
                sid,
                np.array([x_est, y_est], dtype=np.float64),
                ship.goal.copy(),
                float(psi_est),
                float(v_est),
            )
            est_ship.reached = getattr(ship, "reached", False)
            est_ships.append(est_ship)

        obs_dict = build_observations(
            est_ships,
            self.K_neighbors,
            self.spawn_mode,
            self.spawn_area,
            self.spawn_len,
            self.v_max,
        )

        obs_out: Dict[str, np.ndarray] = {}
        for i, aid in enumerate(self.agents):
            if aid in obs_dict:
                obs_out[aid] = obs_dict[aid]
            else:
                obs_out[aid] = zero_observation(self._obs_space)
        return obs_out

    # ------------------------------------------------------------------
    # 调试：导出 true vs AIS 估计状态
    # ------------------------------------------------------------------
    def get_debug_snapshot(self) -> list[dict]:
        state = self._get_world_state()
        ships_true = state.ships
        t = float(getattr(state, "t", 0.0))

        rows: list[dict] = []
        for idx, ship in enumerate(ships_true):
            sid_attr = getattr(ship, "id", None)
            sid = int(sid_attr) if sid_attr is not None else (idx + 1)

            tr = self._tracks.get(sid, None)

            if tr is not None and tr.valid:
                age = t - tr.t_meas
                dt = min(max(0.0, age), self.ais_max_age)

                x_est = tr.x + tr.v * math.cos(tr.psi) * dt
                y_est = tr.y + tr.v * math.sin(tr.psi) * dt
                v_est = tr.v
                psi_est = tr.psi
                ais_valid = True
                ais_visible = age <= self.ais_max_age
            else:
                x_est = float(ship.pos[0])
                y_est = float(ship.pos[1])
                v_est = float(ship.v)
                psi_est = float(ship.psi)
                age = float("inf")
                ais_valid = False
                ais_visible = False

            rows.append(
                {
                    "t": t,
                    "ship_id": sid,
                    "x_true": float(ship.pos[0]),
                    "y_true": float(ship.pos[1]),
                    "v_true": float(ship.v),
                    "psi_true": float(ship.psi),
                    "x_est": float(x_est),
                    "y_est": float(y_est),
                    "v_est": float(v_est),
                    "psi_est": float(psi_est),
                    "ais_age": float(age),
                    "ais_valid": bool(ais_valid),
                    "ais_visible": bool(ais_visible),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # 小工具函数
    # ------------------------------------------------------------------
    def _debug_ais_stats(self) -> Tuple[float, int]:
        state = self._get_world_state()
        t = float(getattr(state, "t", 0.0))

        max_age = 0.0
        vis_cnt = 0
        for sid, tr in self._tracks.items():
            if not tr.valid:
                continue
            age = t - tr.t_meas
            if age <= self.ais_max_age:
                vis_cnt += 1
            max_age = max(max_age, age)
        return max_age, vis_cnt

    @staticmethod
    def _wrap_angle(x: float) -> float:
        return (x + math.pi) % (2.0 * math.pi) - math.pi

    # ------------------------------------------------------------------
    # 透明转发其它接口
    # ------------------------------------------------------------------
    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()
        return None
