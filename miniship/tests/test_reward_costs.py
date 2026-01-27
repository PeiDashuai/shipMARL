# tests/test_reward_costs.py
import numpy as np
from miniship.reward.shaping import task_reward, progress_phi
from miniship.reward.costs import cost_near, cost_coll_time
import math
from miniship.risk.tcpa_dcpa import tcpa_dcpa_matrix
from miniship.reward.costs import cost_rule


def test_arrival_bonus_and_timeout(env_factory):
    env = env_factory()
    env.reset(seed=7)

    # 手动制造超时
    env.state.t = env.T_max + 1e-9
    coll, pair, c_coll, c_time, timeout = \
        cost_coll_time(env.state.ships, env.collide_thr, env.state.t, env.T_max)
    assert timeout and c_time.max() == 1.0 and c_coll.sum() == 0.0

# tests/test_reward_costs.py 片段替换
def test_cost_near_monotonic(env_factory):
    env = env_factory()
    env.reset(seed=17)

    # 明确拉开与 collide_thr 的关系：近 < 阈值 < 中 < 远
    thr = env.collide_thr  # 12.0
    risk   = np.array([0.6, 0.6, 0.6, 0.6])
    vj_max = np.array([2.0, 2.0, 2.0, 2.0])
    v_self = np.array([1.0, 1.0, 1.0, 1.0])

    dmin_near = np.array([0.5*thr]*4)   # 明确小于阈值 → 必有罚
    dmin_mid  = np.array([5.0*thr]*4)   # 大于阈值 → 常为 0
    dmin_far  = np.array([20.0*thr]*4)  # 远离 → 0

    from miniship.reward.costs import cost_near

    c_near = cost_near(risk, vj_max, v_self, dmin_near, thr, tau=6.0).mean()
    c_mid  = cost_near(risk, vj_max, v_self, dmin_mid,  thr, tau=6.0).mean()
    c_far  = cost_near(risk, vj_max, v_self, dmin_far,  thr, tau=6.0).mean()

    assert c_near > 0.0
    assert c_mid  >= c_far >= 0.0
    assert c_near > c_mid


def test_rule_cost_relative(env_factory):
    # 构造确定性右横交叉：N=2
    env = env_factory({"N": 2, "numNeighbors": 1})
    env.reset(seed=0)

    # 手动布置几何：ship0 在原点朝东；ship1 在其右舷上方，朝南
    s0, s1 = env.state.ships[0], env.state.ships[1]
    s0.pos[:] = np.array([0.0, 0.0]);  s0.psi = 0.0;                s0.v = 2.0
    s1.pos[:] = np.array([80.0, 80.0]); s1.psi = -math.pi/2.0;      s1.v = 2.0

    # 计算本步的 TCPA/DCPA（作为 cost_rule 的输入）
    tc, dc, _ = tcpa_dcpa_matrix(env.state.ships)

    # 两组动作（只关心船0的相对罚）
    # 左转+加速（给路船不该这样） vs 右转+减速（符合规则）
    dpsi_max = env.dpsi_max; dv_max = env.dv_max
    dpsi_rl_left  = np.array([+dpsi_max, 0.0])   # ship0 左转，ship1 保持
    dpsi_rl_right = np.array([-dpsi_max, 0.0])   # ship0 右转
    dv_act_fast   = np.array([+dv_max, 0.0])     # ship0 加速
    dv_act_slow   = np.array([-dv_max, 0.0])     # ship0 减速

    c_left = cost_rule(env.state.ships, tc, dc, env.risk_T_thr, env.risk_D_thr,
                       env.thHeadOn, env.thCross, dpsi_rl_left,  dv_act_fast, env.dv_max)
    c_right= cost_rule(env.state.ships, tc, dc, env.risk_T_thr, env.risk_D_thr,
                       env.thHeadOn, env.thCross, dpsi_rl_right, dv_act_slow, env.dv_max)

    # 断言：至少船0的 cost 在“左转+加速”下更大
    assert c_left[0] > c_right[0]
