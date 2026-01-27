# tests/conftest.py
import math
import numpy as np
import pytest
from miniship.core.env import MiniShipParallelEnv
from miniship.scenario.config import SpawnConfig
from miniship.scenario.sampler import ScenarioSampler

@pytest.fixture
def cfg_base():
    return dict(
        N=4, dt=0.5, T_max=60.0,
        v_max=3.0, v_min=0.1, dv_max=0.8, dpsi_max=math.radians(20),
        goal_tol=10.0, collide_thr=12.0,
        spawn_area=180.0, spawn_margin=12.0,
        spawn_min_sep=40.0, spawn_goal_min_sep=60.0,
        spawn_len=160.0, spawn_retry=80, spawn_mode="circle_center",
        spawn_dir_jitter_deg=6.0,
        risk_T_thr=110.0, risk_D_thr=40.0,
        numNeighbors=3,
        vCruiseK=0.80, alphaOpen=0.20, capPow=1.30, capGain=0.30,
        K_release=4, M_boost=10, dvBoost=1.50, thrRisk_gate=0.35,
        enable_debug=True,  # 关键：很多测试用 debug 数据
    )

@pytest.fixture
def env_factory(cfg_base):
    def _make_env(overrides: dict = None):
        cfg = dict(cfg_base)
        if overrides:
            cfg.update(overrides)
        return MiniShipParallelEnv(cfg)
    return _make_env

@pytest.fixture
def rng():
    return np.random.default_rng(123)

@pytest.fixture
def sampler_factory(cfg_base):
    def _make_sampler(mode="circle_center", overrides: dict = None):
        scfg = SpawnConfig(
            N=overrides.get("N", cfg_base["N"]) if overrides else cfg_base["N"],
            spawn_area=overrides.get("spawn_area", cfg_base["spawn_area"]) if overrides else cfg_base["spawn_area"],
            spawn_margin=overrides.get("spawn_margin", cfg_base["spawn_margin"]) if overrides else cfg_base["spawn_margin"],
            spawn_min_sep=overrides.get("spawn_min_sep", cfg_base["spawn_min_sep"]) if overrides else cfg_base["spawn_min_sep"],
            spawn_goal_min_sep=overrides.get("spawn_goal_min_sep", cfg_base["spawn_goal_min_sep"]) if overrides else cfg_base["spawn_goal_min_sep"],
            spawn_len=overrides.get("spawn_len", cfg_base["spawn_len"]) if overrides else cfg_base["spawn_len"],
            spawn_retry=overrides.get("spawn_retry", cfg_base["spawn_retry"]) if overrides else cfg_base["spawn_retry"],
            spawn_dir_jitter_deg=overrides.get("spawn_dir_jitter_deg", cfg_base["spawn_dir_jitter_deg"]) if overrides else cfg_base["spawn_dir_jitter_deg"],
            collide_thr=overrides.get("collide_thr", cfg_base["collide_thr"]) if overrides else cfg_base["collide_thr"],
            v_min=overrides.get("v_min", cfg_base["v_min"]) if overrides else cfg_base["v_min"],
            v_max=overrides.get("v_max", cfg_base["v_max"]) if overrides else cfg_base["v_max"],
            mode=mode,
        )
        return ScenarioSampler(scfg)
    return _make_sampler
