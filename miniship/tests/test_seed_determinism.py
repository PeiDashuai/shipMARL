import numpy as np
import math
from miniship.core.env import MiniShipParallelEnv

def test_seed_reproducibility():
    cfg = dict(
        spawn_mode="random_fixedlen",
        N=3,
        numNeighbors=2,
        dt=0.5, T_max=40.0,
        v_max=3.0, v_min=0.1, dv_max=0.8, dpsi_max=math.radians(20),
        goal_tol=10.0, collide_thr=12.0,
        spawn_area=120.0, spawn_margin=8.0,
        spawn_min_sep=30.0, spawn_goal_min_sep=50.0,
        spawn_len=110.0, spawn_retry=40, spawn_dir_jitter_deg=6.0,
        risk_T_thr=110.0, risk_D_thr=40.0,
        enable_debug=False,
    )

    env1 = MiniShipParallelEnv(cfg)
    env2 = MiniShipParallelEnv(cfg)

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)
    # 初始观测逐 agent 一致
    for a in env1.agents:
        assert np.allclose(obs1[a], obs2[a], atol=1e-8)

    # 给相同动作，检查首步转移一致
    acts1 = {a: env1.action_space(a).sample() for a in env1.agents}
    # 复制同一份动作到 env2
    acts2 = {a: acts1[a].copy() for a in env1.agents}

    nobs1, rew1, term1, trunc1, info1 = env1.step(acts1)
    nobs2, rew2, term2, trunc2, info2 = env2.step(acts2)

    for a in env1.agents:
        assert np.allclose(nobs1[a], nobs2[a], atol=1e-8)
        assert rew1[a] == rew2[a]
    assert term1 == term2 and trunc1 == trunc2
