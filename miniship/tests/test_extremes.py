# tests/test_extremes.py
import numpy as np

def test_single_agent_no_neighbors(env_factory):
    env = env_factory({"N": 1, "numNeighbors": 4})
    obs, _ = env.reset(seed=1)
    x = list(obs.values())[0]
    # 自6 + K*8 + ID，其中邻居段应为 0
    six = x[:6]; neigh = x[6:-1]; aid = x[-1]
    assert np.isfinite(x).all()
    assert np.allclose(neigh, 0.0)

def test_larger_N_smoke(env_factory):
    env = env_factory({"N": 12, "numNeighbors": 4})
    obs, _ = env.reset(seed=5)
    acts = {aid: np.array([0.0, 0.0], np.float32) for aid in env.agents}
    for _ in range(5):
        obs, rew, term, trunc, infos = env.step(acts)
    # 不抛异常即可

def test_edge_params(env_factory):
    # v_min ≈ v_max
    env = env_factory({"v_min": 1.0, "v_max": 1.0})
    obs, _ = env.reset(seed=9)
    acts = {aid: np.array([0.0, 1.0], np.float32) for aid in env.agents}
    o2, r, t, tr, i = env.step(acts)
    # 不应出现 NaN/Inf
    for aid in env.agents:
        assert np.isfinite(o2[aid]).all()
