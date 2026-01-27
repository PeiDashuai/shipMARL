import math
import numpy as np
from pettingzoo.test import parallel_api_test
from miniship.core.env import MiniShipParallelEnv

def test_parallel_api_contract():
    # 用一个小配置快速跑官方并行接口一致性测试
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
    env = MiniShipParallelEnv(cfg)
    parallel_api_test(env, num_cycles=30)  # 官方测试：step/reset/close 等契约检查
