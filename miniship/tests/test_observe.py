# tests/test_observe.py
import numpy as np
from miniship.observe.builder import build_observations

def test_observations_match_builder(env_factory):
    env = env_factory()
    obs, info = env.reset(seed=123)
    # reset 一致
    rebuild = build_observations(env.state.ships, env.K_neighbors, env.spawn_mode, env.spawn_area, env.v_max)
    for aid in env.agents:
        assert np.allclose(obs[aid], rebuild[aid], atol=1e-8)

    # step 一次后也一致
    acts = {aid: np.array([0.0, 0.6], np.float32) for aid in env.agents}
    obs2, _, _, _, infos = env.step(acts)
    rebuild2 = build_observations(env.state.ships, env.K_neighbors, env.spawn_mode, env.spawn_area, env.v_max)
    for aid in env.agents:
        assert np.allclose(obs2[aid], rebuild2[aid], atol=1e-6)

def test_observation_no_nan(env_factory):
    env = env_factory()
    obs, _ = env.reset(seed=321)
    for aid in env.agents:
        x = obs[aid]
        assert np.isfinite(x).all()
