#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math, numpy as np, os
from typing import Dict
from ais_comms.datatypes import AgentId, ShipId, Ts, TrueState
from ais_comms.ais_comms import AISCommsSim
from ais_comms.obs_builder import AISObservationBuilder
import argparse



class MiniShipParallelEnv:
    def __init__(self, N: int = 4, dt: float = 1.0, seed: int = 0, cfg_path: str | None = None):
        self.N = int(N)
        self.dt = float(dt)
        self.t  = 0.0
        self.rng = np.random.default_rng(seed)   # ← 绑定到 self

        self.agents   = [f"{chr(ord('A')+i)}" for i in range(self.N)]
        self.ship_ids = [100 + i for i in range(self.N)]  # 简单全局唯一
        self.agent_of_ship = {sid: self.agents[i] for i, sid in enumerate(self.ship_ids)}

        # 透传 cfg_path；base_seed 用 seed，保证可复现实验
        self.ais = AISCommsSim(rng=self.rng, cfg_path=cfg_path, base_seed=seed)

        self.obs_builder = AISObservationBuilder(slot_K=6, slot_ttl=60.0)
        self.states: Dict[ShipId, TrueState] = {}
        self.vel: Dict[ShipId, tuple[float, float]] = {}   # ← 保存 vx,vy


    def reset(self):
        self.t = 0.0
        R = 500.0; SPEED = 2.0
        thetas = np.linspace(0, 2*np.pi, self.N, endpoint=False)

        self.states.clear()
        self.vel.clear()

        for i, sid in enumerate(self.ship_ids):
            x = R*np.cos(thetas[i]); y = R*np.sin(thetas[i])
            n = np.hypot(-x, -y)
            vx, vy = SPEED*(-x)/(n+1e-9), SPEED*(-y)/(n+1e-9)
            sog = float(SPEED)
            cog = float(math.atan2(vy, vx))
            # TrueState 需为 (sid, x, y, sog, cog) —— 与 AIS 使用一致
            self.states[sid] = TrueState(sid, x=x, y=y, vx=vx, vy=vy)
            self.vel[sid] = (vx, vy)  # 单独保存速度向量

        self.ais.reset(ships=self.ship_ids, t0=self.t, agent_map=self.agent_of_ship)
        self.obs_builder.reset(agent_ids=self.agents, t0=self.t)

        ready = self.ais.step(self.t, self.states)
        obs = self.obs_builder.build(
            ready, self.t,
            {self.agent_of_ship[sid]: self.states[sid] for sid in self.ship_ids}
        )
        return obs, {"t": self.t}


    def step(self, actions=None):
        # 物理推进
        for sid, st in self.states.items():
            vx, vy = self.vel[sid]
            st.x += vx * self.dt
            st.y += vy * self.dt
            # sog/cog 会通过属性自动从 vx,vy 计算，无需赋值

        self.t += self.dt

        ready = self.ais.step(self.t, self.states)
        obs = self.obs_builder.build(
            ready, self.t,
            {self.agent_of_ship[sid]: self.states[sid] for sid in self.ship_ids}
        )
        rews   = {a: 0.0 for a in self.agents}
        terms  = {a: False for a in self.agents}
        truncs = {a: False for a in self.agents}
        infos  = {a: {"t": self.t} for a in self.agents}
        return obs, rews, terms, truncs, infos

def print_metrics(t, env, obs):
    m = env.ais.metrics_snapshot()
    print(f"[t={int(t)}]")
    for aid in env.agents:
        print(f"  {aid} obs={obs[aid]}")
    print(f"  [COMMS] attempts={int(m['tx_attempts'])} passed={int(m['passed'])} "
          f"dropped={int(m['dropped'])} delivered={int(m['delivered'])} "
          f"PPR={m['ppr']:.3f} PDR={m['pdr']:.3f}  bad_occ={m['bad_occupancy']:.3f}")
    print(f"          delay_avg={m['delay_avg']:.3f}s p95={m['delay_p95']:.3f}s   "
          f"age_avg={m['age_avg']:.3f}s p95={m['age_p95']:.3f}s")
    for aid in env.agents:
        keys = sorted(env.obs_builder.last_seen_by_mmsi.get(aid, {}).keys())
        print(f"  DBG {aid} seen_mmsi={keys}  count={len(keys)}/{env.N-1}")


def run_demo(N=4, seed=42, steps=120, print_every=10,
             out_csv="ais_metrics.csv", out_png="ais_metrics.png",
             cfg_path=None):
    rng = np.random.default_rng(seed)
    env = MiniShipParallelEnv(N=N, seed=seed, cfg_path=cfg_path)  # ← 传 cfg_path
    obs, _ = env.reset()
    print("=== Demo start (metrics+timeseries) ===")
    print("Initial obs:"); 
    for aid, o in obs.items(): print(f"  {aid}: {o}")
    for k in range(steps):
        actions = {aid: rng.uniform(-0.2, 0.2, size=(2,)).astype(np.float32) for aid in env.agents}
        obs, rews, terms, truncs, infos = env.step(actions)
        if ((k + 1) % print_every == 0):
            print_metrics(env.t, env, obs)
    env.ais.export_csv(out_csv)
    print(f"\n[Saved] timeseries CSV -> {os.path.abspath(out_csv)}")
    try:
        import matplotlib.pyplot as plt
        t = env.ais.ts
        plt.figure(); plt.plot(t, env.ais.pdr_s, label="PDR"); plt.plot(t, env.ais.ppr_s, label="PPR"); plt.legend(); plt.xlabel("t"); plt.ylabel("ratio"); plt.title("PDR & PPR")
        plt.figure(); plt.plot(t, env.ais.dly_avg_s, label="delay_avg"); plt.plot(t, env.ais.age_avg_s, label="age_avg"); plt.legend(); plt.xlabel("t"); plt.ylabel("sec"); plt.title("Delay/Age avg")
        plt.tight_layout(); plt.savefig(out_png, dpi=150)
        print(f"[Saved] plot -> {os.path.abspath(out_png)}")
    except Exception as e:
        print("[Note] matplotlib not available, skipped plotting:", e)
    print("\n=== Demo end ===")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4, help="number of ships/agents")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--print-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cfg", type=str, default=None, help="path to ais_config.yaml")
    args = ap.parse_args()

    run_demo(N=args.N, seed=args.seed, steps=args.steps,
             print_every=args.print_every, cfg_path=args.cfg)  # ← 传进去

