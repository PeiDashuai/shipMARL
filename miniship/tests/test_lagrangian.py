# tests/test_lagrangian.py
import numpy as np
from miniship.reward.lagrangian import DualState, combine_reward, dual_update_end_of_episode

def test_combine_and_clip():
    dual = DualState(lam_near=2.0, lam_rule=1.0, lam_coll=10.0, lam_time=1.0)
    r_task = np.array([1.0, 2.0, 3.0])
    c_near = np.array([0.1, 0.0, 0.0])
    c_rule = np.array([0.0, 0.2, 0.0])
    c_coll = np.array([0.0, 0.0, 1.0])
    c_time = np.array([0.0, 0.0, 0.0])
    r = combine_reward(r_task, c_near, c_rule, c_coll, c_time, dual, clip=5.0)
    # 手算： [1-0.2, 2-0.2, 3-10] = [0.8, 1.8, -7] → clip → [-5] 最小
    assert np.allclose(r, np.array([0.8, 1.8, -5.0]))

def test_dual_update_direction():
    dual = DualState(lam_near=1.0, ctarget_near=0.05, eta_near=0.1, beta=0.0)
    lam0 = dual.lam_near
    dual_update_end_of_episode(c_near_mean=0.20, c_rule_mean=0.0, c_coll_max=0.0, c_time_max=0.0, dual=dual)
    assert dual.lam_near > lam0
    lam1 = dual.lam_near
    dual_update_end_of_episode(c_near_mean=0.00, c_rule_mean=0.0, c_coll_max=0.0, c_time_max=0.0, dual=dual)
    assert dual.lam_near < lam1
