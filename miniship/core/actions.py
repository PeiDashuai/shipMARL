# miniship/core/actions.py
import numpy as np
from gymnasium.spaces import Box
import re

def _maybe_remap_numeric_keys(raw_dict, agents):
    """
    如果 raw_dict 的 key 是 '1','2',... 而 agents 是 'ship_1','ship_2',...
    则尝试做一次显式映射：'1' -> 'ship_1', '2' -> 'ship_2'。

    只在下面情形触发：
      1) len(keys) == len(agents)
      2) 所有 key 都是纯数字字符串
      3) 所有 agent 都是以 'ship_' 开头且后面跟数字
    """
    keys = list(raw_dict.keys())
    if len(keys) != len(agents):
        # 长度都不一致，直接放弃映射
        return raw_dict

    if not all(isinstance(k, str) and k.isdigit() for k in keys):
        # key 不是 '1','2' 这种形式，也不映射
        return raw_dict

    if not all(isinstance(a, str) and a.startswith("ship_") for a in agents):
        # agent 不是 'ship_1' 这种形式，也不映射
        return raw_dict

    # 提取 agent 后缀数字
    def suffix_id(a):
        m = re.search(r"(\d+)$", a)
        return int(m.group(1)) if m else None

    agent_ids = [suffix_id(a) for a in agents]
    if any(i is None for i in agent_ids):
        return raw_dict

    # 这里我们假定 keys = ['1','2',...,'N']，agents = ['ship_1',...,'ship_N']
    # 显式打印 debug，方便你以后确认这步有没有触发
    #print("[decode_actions][INFO] remap numeric keys -> ship_* agents")
    #print("  raw_dict keys: ", keys)
    #print("  agents:        ", agents)

    remapped = {}
    for a in agents:
        sid = suffix_id(a)          # 1,2,...
        k = str(sid)                # '1','2',...
        if k not in raw_dict:
            print("[decode_actions][WARN] expected key", k, "for agent", a, "but missing in raw_dict; remap aborted.")
            return raw_dict         # 不强行映射，直接原样返回，让后面抛错
        remapped[a] = raw_dict[k]

    return remapped


def decode_actions(raw_dict, action_space: Box, agents):
    """
    raw_dict{agent_id: array(2,)} -> A[N,2] clipped to [-1,1].

    这里显式做了一层“ key 规范化 ”：
    - 首选：raw_dict 的 key 就是 agents 里的 'ship_1','ship_2',...
    - 如果 key 是 '1','2',... 且 agents 是 'ship_1','ship_2',...，
      则尝试 remap 到 'ship_i'。
    - 如果以上都不满足，就让后面的 KeyError 正常抛出，并打印详细上下文。
    """
    # --- 1) 首先尝试把 '1','2' -> 'ship_1','ship_2' ---
    if not set(agents).issubset(set(raw_dict.keys())):
        raw_dict = _maybe_remap_numeric_keys(raw_dict, agents)

    N = len(agents)
    A = np.zeros((N, 2), np.float64)

    for i, aid in enumerate(agents):
        try:
            a = np.asarray(raw_dict[aid], dtype=np.float64).reshape(-1)
        except KeyError as e:
            # 这里保留你的详细 debug，而不是“吃掉”错误
            #print("[decode_actions][ERROR] KeyError when decoding actions!")
            #print("  missing aid:", aid)
            #print("  agents list:", agents)
            #print("  raw_dict keys:", list(raw_dict.keys()))
            #print("  raw_dict full:", raw_dict)
            raise e

        A[i, :] = np.clip(a[:2], -1.0, 1.0)

    return A


def map_to_commands(A, dpsi_max, v_min, v_max):
    """把归一化动作映射为物理量：dpsi_rl, v_cmd。

    约定：A[:,0] 控制速度（映射为 v_cmd），A[:,1] 控制转向（映射为 dpsi_rl）。
    """
    A = np.asarray(A, dtype=np.float32)
    dpsi_rl = A[:, 1] * dpsi_max
    v_cmd = (A[:, 0] + 1.0) * 0.5 * (v_max - v_min) + v_min
    return dpsi_rl, v_cmd