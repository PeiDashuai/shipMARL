# tests/test_interface.py
import numpy as np

def test_pettingzoo_shapes(env_factory):
    env = env_factory()
    obs, infos = env.reset(seed=42)
    assert isinstance(obs, dict) and isinstance(infos, dict)
    assert set(obs.keys()) == set(env.agents) == set(infos.keys())

    # action/obs spaces
    for aid in env.agents:
        assert obs[aid].shape == env.observation_spaces[aid].shape
        a = env.action_spaces[aid].sample()
        assert a.shape == (2,)

    # one step
    acts = {aid: env.action_spaces[aid].sample() for aid in env.agents}
    obs2, rew, term, trunc, infos = env.step(acts)
    assert set(obs2) == set(rew) == set(term) == set(trunc) == set(infos) == set(env.agents)
    for aid in env.agents:
        assert isinstance(rew[aid], float)
        assert isinstance(term[aid], bool)
        assert trunc[aid] is False
