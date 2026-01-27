# tests/test_reproducibility.py
import numpy as np

def test_reset_seed_repro(env_factory):
    env1 = env_factory()
    env2 = env_factory()
    obs1, _ = env1.reset(seed=2024)
    obs2, _ = env2.reset(seed=2024)
    for aid in env1.agents:
        assert np.allclose(obs1[aid], obs2[aid])

def test_first_step_repro(env_factory):
    env1 = env_factory(); env2 = env_factory()
    obs1, _ = env1.reset(seed=33); obs2, _ = env2.reset(seed=33)
    acts = {aid: np.array([0.0, 0.6], np.float32) for aid in env1.agents}
    o1, r1, t1, tr1, i1 = env1.step(acts)
    o2, r2, t2, tr2, i2 = env2.step(acts)
    for aid in env1.agents:
        assert np.allclose(o1[aid], o2[aid])
        assert r1[aid] == r2[aid]
        assert t1[aid] == t2[aid]
        assert tr1[aid] == tr2[aid]
