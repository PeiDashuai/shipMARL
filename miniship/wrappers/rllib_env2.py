from __future__ import annotations
from typing import Tuple, Dict, Any, Set

from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from miniship.wrappers.lagrangian_pz_env import MiniShipLagrangianParallelEnv
from miniship.envs.miniship_ais_comms_env import MiniShipAISCommsEnv


class MiniShipMultiAgentWrapper(MultiAgentEnv):
    """将 PettingZoo ParallelEnv 适配为 RLlib 的 MultiAgentEnv。

    关键点：
    - 保留原始 agent id（'ship_1', 'ship_2', ...），不再做任何重命名；
    - reset/step 直接转发底层 env 的 obs / reward / info；
    - 如果 terminations/truncations 中没有 '__all__'，这里自动补充；
    - 暴露 observation_space/action_space 属性，供 RLlib 和自定义模型推断维度；
    - 维护 _agent_ids，避免 RLlib 的 env 检查器报错。
    """

    def __init__(self, cfg: Dict[str, Any], use_pf_ais_env: bool):
        super().__init__()
        if use_pf_ais_env:
            print(
                "[rllib_env] MiniShipMultiAgentWrapper: using MiniShipAISCommsEnv "
                "+ PF track manager as underlying env."
            )
            self._env = MiniShipAISCommsEnv(cfg)
        else:
            print(
                "[rllib_env] MiniShipMultiAgentWrapper: using MiniShipLagrangianParallelEnv "
                "(legacy core/lagrangian env; may use AISObsWrapper internally)."
            )
            self._env = MiniShipLagrangianParallelEnv(cfg)

        # 当前 episode 中存活的 agents
        # PettingZoo ParallelEnv 一般有 agents / possible_agents 两个属性
        if getattr(self._env, "agents", None):
            self._agent_ids: Set[str] = set(self._env.agents)
        elif getattr(self._env, "possible_agents", None):
            self._agent_ids = set(self._env.possible_agents)
        else:
            raise RuntimeError(
                "Underlying env has no 'agents' or 'possible_agents' attribute."
            )

        # 假设所有 agent 共享同一观测/动作空间（对称多智能体）
        any_agent = next(iter(self._agent_ids))
        self.observation_space = self._env.observation_space(any_agent)
        self.action_space = self._env.action_space(any_agent)
        self._last_ep_infos: Dict[str, dict] = {}   # 新增：保存上一局的 infos

    @property
    def agents(self):
        # RLlib 在很多地方会直接访问 env.agents
        return list(self._agent_ids)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        """Gymnasium 风格 reset，返回 (obs_dict, info_dict)。"""
        obs, infos = self._env.reset(seed=seed, options=options)

        # 底层 env 可能在 reset 时更新 agents 列表，这里同步一次
        if getattr(self._env, "agents", None):
            self._agent_ids = set(self._env.agents)
        self._last_ep_infos = {}   # 新增：每局开始先清空
        return obs, infos

    def step(self, action_dict: Dict[str, Any]):
        """Gymnasium 风格 step，返回 (obs, rewards, terminations, truncations, infos)。"""
        # 底层 env（MiniShipAISCommsEnv 或 MiniShipLagrangianParallelEnv）
        # 返回的是 gymnasium ParallelEnv 风格：
        #   obs, rewards, terminations, truncations, infos
        obs, rewards, terminations, truncations, infos = self._env.step(action_dict)

        # 同步 agent ids（有些并行 env 在 episode 过程中也会 pop 掉 done 的 agent）
        if getattr(self._env, "agents", None):
            self._agent_ids = set(self._env.agents)

        # 确保 __all__ 键存在：RLlib 的 MultiAgentEnv 约定
        terminations = dict(terminations)
        truncations = dict(truncations)

        if "__all__" not in terminations and "__all__" not in truncations:
            # 如果底层 env 没给 __all__，根据任意 agent 终止/截断来判断 episode 是否结束
            any_done = False
            for aid in self._agent_ids:
                if terminations.get(aid, False) or truncations.get(aid, False):
                    any_done = True
                    break
            terminations["__all__"] = any_done
            truncations["__all__"] = False
        else:
            terminations.setdefault("__all__", False)
            truncations.setdefault("__all__", False)

        # ========= 关键新增：记录本局最后一步的 infos =========
        done_all = terminations.get("__all__", False) or truncations.get("__all__", False)
        if done_all:
            last_infos = {}
            for aid, inf in infos.items():
                if aid == "__all__":
                    continue
                if isinstance(inf, dict):
                    # 复制一份，避免后面被修改
                    last_infos[aid] = inf.copy()
                else:
                    last_infos[aid] = {"raw_info": inf}
            self._last_ep_infos = last_infos
        # =====================================================

        return obs, rewards, terminations, truncations, infos


    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()


def register_miniship_env(base_env_cfg: Dict[str, Any]) -> Tuple[str, Any, Any]:
    """向 RLlib 注册 MiniShip 环境。

    现在支持两种模式：
      1) use_pf_ais_env=True:  使用 MiniShipAISCommsEnv（核心环境 + AISComms + PF tracking）
      2) use_pf_ais_env=False: 使用原有 MiniShipLagrangianParallelEnv

    二者都通过 MiniShipMultiAgentWrapper 适配为 RLlib 的 MultiAgentEnv，
    不再依赖 RLlib 自带的 ParallelPettingZooEnv，从而：
      - 保持 agent id 为 'ship_1', 'ship_2', ...，避免 '1', '2' 这样的重命名；
      - 完整保留 info 里的 term_reason / c_near / c_rule / guard_trig_step_mean 等字段；
      - 解决之前 env 检查器报的 agent_ids 不一致、callback 拿不到 term_reason 的问题。
    """

    env_name = "MiniShipLagrangianPZ-v0"

    def env_creator(env_config: Dict[str, Any]):
        # 1) 合并 base_env_cfg 和 RLlib 传入的 env_config
        cfg = dict(base_env_cfg)
        if env_config:
            cfg.update(env_config)

        # 2) 读取开关：是否使用 PF 通信环境作为顶层 env
        use_pf_ais_env = bool(cfg.get("use_pf_ais_env", False))

        # 3) 直接用我们自己的 MultiAgent 包装器（不再走 ParallelPettingZooEnv）
        return MiniShipMultiAgentWrapper(cfg, use_pf_ais_env=use_pf_ais_env)

    # 注册给 RLlib
    register_env(env_name, env_creator)

    # 为训练脚本拿到单智能体空间：构造一个“临时 env”
    tmp_cfg = dict(base_env_cfg)
    use_pf_ais_env_tmp = bool(tmp_cfg.get("use_pf_ais_env", False))

    tmp_env = MiniShipMultiAgentWrapper(tmp_cfg, use_pf_ais_env=use_pf_ais_env_tmp)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()

    return env_name, obs_space, act_space
