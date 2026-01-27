class SpooferInterface:
    def __init__(self, limits: LimitParams, rng):
        self.rng=rng; self.lim=limits
        self.budget = 0.0
    def reset(self, episode_budget: float):
        self.budget = episode_budget
    def apply(self, raw_msg: RawTxMsg, region_cell: RegionBiasCell) -> Tuple[RawTxMsg, List[AttackFlag], float]:
        # 从对抗agent的动作缓冲读取，或按脚本规则执行
        # 返回：修改后的msg、flags、消耗的cost
        # TODO: 实现 offset/delay/silence/mirror/id_swap 等，并裁剪到 lim
        return raw_msg, [], 0.0
