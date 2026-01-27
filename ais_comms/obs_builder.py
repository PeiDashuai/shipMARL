# obs_builder.py 
import math, numpy as np
from typing import Dict, List
from .datatypes import AgentId, Ts, TrueState, RxMsg

class AISObservationBuilder:
    """
    兼容旧输出（单/latest obs per agent）且内部维护：
    - last_seen_by_mmsi[agent][mmsi] = RxMsg
    - mmsi_to_slot[agent][mmsi] = slot_index  (stable per agent)
    - slot_last_seen_time[agent][slot] = last timestamp (用于回收)
    当前仍返回与旧版相同的单向量 obs（若想扩展到 K slots，可在此基础上扩展）
    """
    def __init__(self, slot_K: int = 8, slot_ttl: float = 60.0):
        self.last_seen_by_mmsi: Dict[AgentId, Dict[int, RxMsg]] = {}
        self.mmsi_to_slot: Dict[AgentId, Dict[int, int]] = {}
        self.slot_last_seen_time: Dict[AgentId, Dict[int, float]] = {}
        self.slot_K = int(slot_K)
        self.slot_ttl = float(slot_ttl)    # 若 slot 超过此未见，回收
        self.last_time: Dict[AgentId, Ts] = {}
        self.miss_run: Dict[AgentId, int] = {}
        self.miss_run_max: Dict[AgentId, int] = {}
        self.eps = 1e-6

    def reset(self, agent_ids: List[AgentId], t0: Ts = 0.0):
        for aid in agent_ids:
            self.last_seen_by_mmsi[aid] = {}
            self.mmsi_to_slot[aid] = {}
            self.slot_last_seen_time[aid] = {}
            # 初始化可用槽（0..K-1）都为空
            for s in range(self.slot_K):
                self.slot_last_seen_time[aid][s] = -1.0
            self.last_time[aid] = float(t0)
            self.miss_run[aid] = 0
            self.miss_run_max[aid] = 0

    def _assign_slot(self, aid: AgentId, mmsi: int, now: float) -> int:
        """
        给一个 mmsi 分配稳定槽位：若已有映射返回原槽位；
        否则找最久未用的槽位回收（slot_last_seen_time 最小者）。
        """
        if mmsi in self.mmsi_to_slot[aid]:
            slot = self.mmsi_to_slot[aid][mmsi]
            self.slot_last_seen_time[aid][slot] = now
            return slot

        # 找最老的槽位回收
        slot_times = self.slot_last_seen_time[aid]
        # choose slot with min last_seen_time
        slot = min(slot_times.keys(), key=lambda s: slot_times[s] if slot_times[s] >= 0 else -1e9)
        self.mmsi_to_slot[aid][mmsi] = slot
        self.slot_last_seen_time[aid][slot] = now
        return slot

    def _cleanup_slots(self, aid: AgentId, now: float):
        """
        回收长期未见的 mmsi->slot 映射（slot idle for > slot_ttl）。
        """
        to_del = []
        for mmsi, slot in self.mmsi_to_slot[aid].items():
            last = self.slot_last_seen_time[aid].get(slot, -1.0)
            if now - last > self.slot_ttl:
                to_del.append(mmsi)
        for m in to_del:
            slot = self.mmsi_to_slot[aid].pop(m, None)
            if slot is not None:
                self.slot_last_seen_time[aid][slot] = -1.0
                self.last_seen_by_mmsi[aid].pop(m, None)

    def build(self, ready_msgs: Dict[AgentId, List[RxMsg]], t: Ts, own_true: Dict[AgentId, TrueState]):
        out = {}
        for aid, own in own_true.items():
            msgs = ready_msgs.get(aid, [])
            # 先把新到的消息缓存到 per-agent per-mmsi map，并更新 slot map
            if msgs:
                for m in msgs:
                    mmsi = int(m.mmsi)
                    if aid not in self.last_seen_by_mmsi:
                        self.last_seen_by_mmsi[aid] = {}
                    self.last_seen_by_mmsi[aid][mmsi] = m
                    # assign / refresh slot
                    _ = self._assign_slot(aid, mmsi, m.arrival_time)
                # 做 slot 回收
                self._cleanup_slots(aid, t)

            # 选出最新的一条消息（与旧逻辑一致），但我们拥有按 mmsi 的缓存可用于扩展
            last_msg = None
            # prefer newest by arrival_time among last_seen_by_mmsi
            mdict = self.last_seen_by_mmsi.get(aid, {})
            if mdict:
                last_msg = max(mdict.values(), key=lambda z: z.arrival_time)

            if last_msg is None:
                age_now = min(30.0, t - self.last_time.get(aid, 0.0))
                obs = np.array([0, 0, 1, 0, age_now, 1.0], dtype=np.float32)
                self.miss_run[aid] += 1
                self.miss_run_max[aid] = max(self.miss_run_max[aid], self.miss_run[aid])
            else:
                dx = last_msg.reported_x - own.x
                dy = last_msg.reported_y - own.y
                R = float(math.hypot(dx, dy))
                bearing = float(math.atan2(dy, dx))
                sinb, cosb = math.sin(bearing), math.cos(bearing)
                vx_t = float(last_msg.reported_sog * math.cos(last_msg.reported_cog))
                vy_t = float(last_msg.reported_sog * math.sin(last_msg.reported_cog))
                v_rel = (vx_t - own.vx) * (dx/(R+self.eps)) + (vy_t - own.vy) * (dy/(R+self.eps))
                obs = np.array([R, sinb, cosb, v_rel, float(last_msg.age), 0.0], dtype=np.float32)
                self.miss_run[aid] = 0

            self.last_time[aid] = t
            out[aid] = obs
        return out
