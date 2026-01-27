# ais_comms/event_queue.py
from __future__ import annotations
from typing import Dict, List, Tuple
import heapq
import math


class ArrivalQueue:
    """
    简单到达队列：
      - push(at, rx_agent, msg): 以到达时间 'at' 入堆
      - pop_ready(t): 弹出所有 at <= t 的报文，按 rx_agent 分桶返回
      - set_limits(ttl_s, max_inflight): 配置 TTL 与 in-flight 上限
        * TTL: 若报文在弹出时 (t - arrival_time) > ttl_s，则直接丢弃
        * max_inflight: 超过上限时丢弃新入队的报文（最简单且可预期）

    为避免 heapq 在 arrival_time、rx_agent 相同的情况下比较 msg 对象本身，
    堆中元素采用 (arrival_time, counter, rx_agent, msg) 四元组，其中 counter
    为单调递增的整型计数器，作为 tie-breaker。
    """

    def __init__(self, ttl_s: float = 5.0, max_inflight: int = 10000):
        self.ttl_s = float(ttl_s)
        self.max_inflight = int(max_inflight)

        # (arrival_time, counter, rx_agent, msg)
        self._heap: List[Tuple[float, int, str, object]] = []
        self._inflight = 0

        # 单调递增计数器，确保堆中元素在 (arrival_time, counter, ...) 上有全序
        self._counter: int = 0

        # 统计（目前仅给 bad/丢弃分析留钩子，不强制使用）
        self.dropped_overflow = 0
        self.dropped_ttl = 0

        # --- per-link stats ---
        # link_key: (tx_sid, rx_agent) e.g. (1, "B")
        self.delivered_by_link = {}   # pushed into queue
        self.popped_by_link = {}      # popped out by pop_ready
        self.dropped_ttl_by_link = {} # dropped due TTL


    def reset_metrics(self):
        self.dropped_overflow = 0
        self.dropped_ttl = 0

    def set_limits(self, *, ttl_s: float, max_inflight: int):
        self.ttl_s = float(ttl_s)
        self.max_inflight = int(max_inflight)

    def push(self, arrival_time: float, rx_agent: str, msg, link_key=None):
        """
        入队一条将在 arrival_time 送达 rx_agent 的报文。

        为避免 TypeError: '<' not supported between instances of 'RxMsg' and 'RxMsg'，
        这里在堆元素中加入自增 counter 作为第二排序关键字。
        """
        # 容量保护：超过 in-flight 上限就丢弃新报文
        if self._inflight >= self.max_inflight:
            self.dropped_overflow += 1
            return

        self._counter += 1

        if link_key is not None:
            self.delivered_by_link[link_key] = self.delivered_by_link.get(link_key, 0) + 1

        heapq.heappush(self._heap, (float(arrival_time), self._counter, rx_agent, link_key, msg))

        self._inflight += 1

    def pop_ready(self, t: float) -> Dict[str, List[object]]:
        """
        弹出所有 arrival_time <= t 的报文，按 rx_agent 分桶返回。

        若 (t - arrival_time) > ttl_s，则认为该报文超时未取，丢弃并计入 dropped_ttl。
        """
        ready: Dict[str, List[object]] = {}
        t = float(t)

        while self._heap and self._heap[0][0] <= t:
            at, _, rx, link_key, msg = heapq.heappop(self._heap)
            self._inflight -= 1

            # TTL 检查（到达晚于当前 t 的不会进来；这里处理“准备好但超时未取”的情况）
            if (t - at) > self.ttl_s:

                self.dropped_ttl += 1

                if link_key is not None:
                    self.dropped_ttl_by_link[link_key] = self.dropped_ttl_by_link.get(link_key, 0) + 1

                continue

            ready.setdefault(rx, []).append(msg)

            if link_key is not None:
                self.popped_by_link[link_key] = self.popped_by_link.get(link_key, 0) + 1
                
        return ready
