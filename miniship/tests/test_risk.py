# tests/test_risk.py
import numpy as np
from miniship.risk.tcpa_dcpa import tcpa_dcpa_matrix

def test_tcpa_dcpa_symmetry(env_factory):
    env = env_factory()
    env.reset(seed=999)
    tc, dc, _ = tcpa_dcpa_matrix(env.state.ships)
    assert np.allclose(tc, tc.T)
    assert np.allclose(dc, dc.T)
    assert np.all(tc >= 0.0)
    assert np.all(dc >= 0.0)

def test_post_risk_matches_debug(env_factory):
    env = env_factory()
    _, _ = env.reset(seed=11)
    acts = {aid: np.array([0.0, 0.6], np.float32) for aid in env.agents}
    _, _, _, _, infos = env.step(acts)
    dbg = infos[env.agents[0]]["debug"]
    post = dbg["post"]
    tc, dc, _ = tcpa_dcpa_matrix(env.state.ships)
    assert np.allclose(tc, np.array(post["tc"]), atol=1e-8)
    assert np.allclose(dc, np.array(post["dc"]), atol=1e-8)
