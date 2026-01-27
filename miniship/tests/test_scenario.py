# tests/test_scenario.py
import numpy as np

def test_circle_center_constraints(sampler_factory, rng):
    sampler = sampler_factory(mode="circle_center")
    starts, goals, psi0, v0 = sampler.sample(rng)
    N = starts.shape[0]

    # 对置目标、最小间距与边界
    R = 0.8 * sampler.cfg.spawn_area
    for i in range(N):
        # 对置（允许微小数值误差）
        assert np.allclose(goals[i], -starts[i], atol=1e-6)
        assert np.linalg.norm(starts[i]) <= R + 1e-6
        assert np.linalg.norm(goals[i]) <= R + 1e-6

    # start-start 间距 & start-goal(i!=j)
    for i in range(N):
        for j in range(i+1, N):
            dSS = np.linalg.norm(starts[i] - starts[j])
            assert dSS >= sampler.cfg.spawn_min_sep - 1e-6
            dGG = np.linalg.norm(goals[i] - goals[j])
            assert dGG >= sampler.cfg.spawn_goal_min_sep - 1e-6
            dSG = np.linalg.norm(starts[i] - goals[j])
            assert dSG >= max(sampler.cfg.spawn_min_sep, sampler.cfg.spawn_goal_min_sep) - 1e-6

def test_random_fixedlen_length(sampler_factory):
    sampler = sampler_factory(mode="random_fixedlen")
    starts, goals, psi0, v0 = sampler.sample(np.random.default_rng(0))
    dist = np.linalg.norm(goals - starts, axis=1)
    assert np.allclose(dist, sampler.cfg.spawn_len, atol=1e-6)
