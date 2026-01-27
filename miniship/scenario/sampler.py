import math, numpy as np
from .config import SpawnConfig
from ..utils.math import wrap_to_pi, unit

class ScenarioSampler:
    """只负责生成 starts、goals、初始 ψ 和 v0；与环境弱耦合。"""
    def __init__(self, cfg: SpawnConfig):
        self.cfg = cfg

    def sample(self, rng: np.random.Generator):
        if self.cfg.mode == "circle_center":
            return self._circle_center(rng)
        elif self.cfg.mode == "random_fixedlen":
            return self._random_fixedlen(rng)
        raise ValueError(f"Unknown spawn mode: {self.cfg.mode}")

    # ==========================================================
    # circle_center：保持原样
    # ==========================================================
    def _circle_center(self, rng):
        N, L = self.cfg.N, self.cfg.spawn_area
        R = 0.8 * L
        SS_thr = max(self.cfg.spawn_min_sep, self.cfg.collide_thr)
        SG_thr = max(self.cfg.spawn_goal_min_sep, self.cfg.collide_thr)

        starts = np.zeros((N,2), float)
        goals  = np.zeros((N,2), float)
        ang = np.full(N, np.nan)

        for i in range(N):
            tries=0; maxTry=200
            while True:
                tries += 1
                a = rng.uniform(0, 2*math.pi)
                s = np.array([R*math.cos(a), R*math.sin(a)], float)
                ok = True
                if i>0:
                    dAng = np.abs((a - ang[:i] + math.pi)%(2*math.pi)-math.pi)
                    chord = 2*R*np.sin(dAng/2.0)
                    if chord.size and np.any(chord < SS_thr): ok=False
                    if ok:
                        for j in range(i):
                            gj = -starts[j]
                            if np.linalg.norm(s - gj) < SG_thr:
                                ok=False; break
                if ok:
                    ang[i]=a; starts[i]=s; goals[i]=-s; break
                if tries>=maxTry:
                    SS_thr = max(self.cfg.collide_thr, SS_thr*0.95); tries=0

        psi0 = np.array([math.atan2(*(goals[i]-starts[i])[::-1]) for i in range(N)], float)
        v0 = self._init_speed(rng, N)
        return starts, goals, psi0, v0

    # ==========================================================
    # random_fixedlen：加入“ICS 检查 + 整局重采样”
    # ==========================================================
    def _random_fixedlen(self, rng):
        """
        生成 random_fixedlen 场景：
          1) 按原逻辑采样 starts / goals / psi0 / v0（几何分离约束）；
          2) 做一次简化 ICS 检查（直航 DCPA/TCPA）：
             - 若在一个时间窗口 T_win 内存在 d_min < 0.9 * collide_thr，
               判定为 ICS-like，整局重采样；
          3) 若多次全局重采样仍无法通过 ICS 检查，则降级为“几何-only”场景：
             - 不再 raise，而是打印 warning，返回最后一次几何合法布局。
        """
        N, L = self.cfg.N, self.cfg.spawn_area
        SS_thr = max(self.cfg.spawn_min_sep, self.cfg.collide_thr)
        GG_thr = max(self.cfg.spawn_goal_min_sep, self.cfg.collide_thr)
        SG_thr = max(self.cfg.spawn_min_sep, self.cfg.spawn_goal_min_sep, self.cfg.collide_thr)

        # 全局重采样上限
        global_retry = 64

        last_geom_starts = None
        last_geom_goals = None

        def sample_in_quadrant(q):
            lo, hi = self.cfg.spawn_margin, L - self.cfg.spawn_margin
            r = lambda a, b: float(rng.uniform(a, b))
            return {
                1: ( r(lo, hi),  r(lo, hi)),
                2: ( r(lo, hi), -r(lo, hi)),
                3: (-r(lo, hi), -r(lo, hi)),
                4: (-r(lo, hi),  r(lo, hi)),
            }[q]

        for g_try in range(global_retry):
            starts = np.zeros((N, 2), float)
            goals  = np.full((N, 2), np.nan)

            base = (np.arange(1, N + 1) + rng.integers(1, 5) - 1) % 4 + 1
            qs   = rng.permutation(base)

            ok_all_ships = True

            for i in range(N):
                ok_ship = False
                for _ in range(self.cfg.spawn_retry):
                    # -------- 采样起点 --------
                    s = np.array(sample_in_quadrant(int(qs[i])), float)

                    # 起点与已有起点/目标的分离约束
                    if i > 0:
                        dSS = np.linalg.norm(starts[:i] - s, axis=1)
                        if dSS.size and np.min(dSS) < SS_thr:
                            continue
                        maskG = np.isfinite(goals[:i, 0])
                        if np.any(maskG):
                            dSG = np.linalg.norm(goals[:i][maskG] - s, axis=1)
                            if dSG.size and np.min(dSG) < SG_thr:
                                continue

                    # -------- 采样目标（对向象限 + 抖动）--------
                    opp_q = 5 - int(qs[i])
                    ok_goal = False
                    for _ in range(self.cfg.spawn_retry):
                        gx, gy = sample_in_quadrant(opp_q)
                        g_ref = np.array([gx, gy], float)
                        dirv = g_ref - s
                        if np.linalg.norm(dirv) < 1e-9:
                            ang0 = rng.uniform(0, 2 * math.pi)
                            dirv = np.array([math.cos(ang0), math.sin(ang0)], float)
                        else:
                            dirv = unit(dirv)

                        jd = math.radians(self.cfg.spawn_dir_jitter_deg)
                        if jd > 0:
                            ang = math.atan2(dirv[1], dirv[0]) + rng.uniform(-jd, jd)
                            dirv = np.array([math.cos(ang), math.sin(ang)], float)

                        g = s + self.cfg.spawn_len * dirv

                        # 目标不得出界
                        if np.any(np.abs(g) > (L - self.cfg.spawn_margin)):
                            continue

                        # 目标与已有目标/起点的分离约束
                        if i > 0:
                            hasG = np.isfinite(goals[:i, 0])
                            if np.any(hasG):
                                dGG = np.linalg.norm(goals[:i][hasG] - g, axis=1)
                                if dGG.size and np.min(dGG) < GG_thr:
                                    continue
                            dGS = np.linalg.norm(starts[:i] - g, axis=1)
                            if dGS.size and np.min(dGS) < SG_thr:
                                continue

                        starts[i] = s
                        goals[i]  = g
                        ok_goal   = True
                        break

                    if ok_goal:
                        ok_ship = True
                        break

                if not ok_ship:
                    ok_all_ships = False
                    break

            if not ok_all_ships:
                # 本轮几何布局失败，整局重采样
                continue

            # 几何已合法，先记录下来作为“最后一次合法布局”
            last_geom_starts = starts.copy()
            last_geom_goals  = goals.copy()

            # 生成 heading 和 speed
            psi0 = np.array(
                [math.atan2(*(goals[i] - starts[i])[::-1]) for i in range(N)],
                float
            )
            v0 = self._init_speed(rng, N)

            # ICS 检查（放宽版）：若判定为 ICS-like，则整局重采样
            if self._has_ics_straightline(starts, psi0, v0):
                continue

            # 通过 ICS 检查，直接返回
            return starts, goals, psi0, v0

        # global_retry 次仍未找到 ICS-free 场景：降级为几何-only 布局
        if last_geom_starts is None or last_geom_goals is None:
            # 连几何布局都没采到，才是真正异常
            raise RuntimeError("random_fixedlen: geometry sampling failed (no valid layout)")

        print("[ScenarioSampler] WARNING: ICS-free sampling failed after "
              f"{global_retry} trials, fall back to geometry-only scenario.")

        starts = last_geom_starts
        goals  = last_geom_goals
        psi0 = np.array(
            [math.atan2(*(goals[i] - starts[i])[::-1]) for i in range(N)],
            float
        )
        v0 = self._init_speed(rng, N)
        return starts, goals, psi0, v0

    # ==========================================================
    # 直航 DCPA/TCPA 检查：筛掉必然碰撞/极端危险的初始布局
    # ==========================================================
    def _has_ics_straightline(self, starts, psi0, v0) -> bool:
        """
        ICS 近似判定（放宽版）：
          - 船舶保持当前 heading 和 speed 直航不操纵；
          - 对每一对 (i,j) 计算相对运动下的最小距离 d_min 和时间 t_star；
          - 只在一个较短时间窗口 T_win 内，且 d_min < 0.9 * collide_thr 时，
            才认为该布局过于“必撞”，需要重采样。
        """
        N = starts.shape[0]
        if N <= 1:
            return False

        coll_thr = float(self.cfg.collide_thr)
        if coll_thr <= 0.0:
            return False

        # 时间窗口：只关心比较近的潜在碰撞
        T_win = 45.0

        vel = np.zeros((N, 2), float)
        for i in range(N):
            vel[i, 0] = v0[i] * math.cos(psi0[i])
            vel[i, 1] = v0[i] * math.sin(psi0[i])

        for i in range(N):
            for j in range(i + 1, N):
                pij = starts[j] - starts[i]
                vij = vel[j] - vel[i]

                vv = float(np.dot(vij, vij))
                if vv < 1e-8:
                    # 相对速度接近 0：d_min 就是当前距离
                    d_min = float(np.linalg.norm(pij))
                    t_star = 0.0
                else:
                    t_star = -float(np.dot(pij, vij)) / vv
                    t_star = max(0.0, t_star)
                    d_min = float(np.linalg.norm(pij + t_star * vij))

                # 放宽判据：只过滤“短时间 + 特别近”的必撞情况
                if (t_star <= T_win) and (d_min < coll_thr * 0.9):
                    return True

        return False


    # ==========================================================
    # 速度初始化：保持你的原逻辑
    # ==========================================================
    def _init_speed(self, rng, N):
        r = rng.uniform(0.6, 0.95, size=N)
        r = r - np.mean(r) + 0.8
        r = np.clip(r, 0.55, 0.98)
        return self.cfg.v_min + r * (self.cfg.v_max - self.cfg.v_min)
