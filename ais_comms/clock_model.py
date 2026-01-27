class ClockModel:
    def __init__(self, params: ClockParams, rng):
        self.delta0 = rng.normal(0, params.sigma_delta0)   # 初始偏移
        self.drift  = rng.normal(0, params.sigma_drift)    # 漂移率
    def local_time(self, global_t: Ts) -> Ts:
        return global_t + self.delta0 + self.drift * global_t
