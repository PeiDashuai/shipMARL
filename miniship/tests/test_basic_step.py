import numpy as np
import math
from miniship.core.env import MiniShipParallelEnv

def test_basic_random_roll():
    cfg = dict(
        spawn_mode="random_fixedlen",
        N=4,
        numNeighbors=3,
        dt=0.5, T_max=60.0,
        v_max=3.0, v_min=0.1, dv_max=0.8, dpsi_max=math.radians(20),
        goal_tol=10.0, collide_thr=12.0,
        spawn_area=160.0, spawn_margin=12.0,
        spawn_min_sep=40.0, spawn_goal_min_sep=60.0,
        spawn_len=140.0, spawn_retry=80, spawn_dir_jitter_deg=6.0,
        risk_T_thr=110.0, risk_D_thr=40.0,
        enable_debug=False,
    )
    env = MiniShipParallelEnv(cfg)
    obs, infos = env.reset(seed=42)

    # 观测/动作空间存在且一致
    for aid in env.agents:
        sp_o = env.observation_space(aid)
        sp_a = env.action_space(aid)
        assert sp_o.contains(obs[aid])
        a = sp_a.sample()
        assert sp_a.contains(a)

    steps = 0
    done = False
    while not done and steps < 200:
        # 随机动作
        acts = {aid: env.action_space(aid).sample() for aid in env.agents}
        obs, rew, term, trunc, infos = env.step(acts)

        # 类型与有限性检查
        assert set(obs.keys()) == set(env.agents)
        assert set(rew.keys()) == set(env.agents)
        for aid in env.agents:
            x = obs[aid]
            r = rew[aid]
            assert np.isfinite(x).all()
            assert np.isfinite(r)
        done = any(term.values()) or any(trunc.values())
        steps += 1

    env.close()
