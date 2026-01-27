class ErrorInjector:
    def __init__(self, cfg: ErrorParams, field_cfg: FieldErrorParams, rng):
        self.cfg = cfg; self.fcfg = field_cfg; self.rng = rng
        self.bias_e = 0.0; self.bias_n = 0.0  # 慢变偏置（东北向）

    def update_bias(self):
        # AR(1): b_{t+1} = rho * b_t + v
        v_e = self.rng.normal(0, self.cfg.bias_sigma_step)
        v_n = self.rng.normal(0, self.cfg.bias_sigma_step)
        self.bias_e = self.cfg.bias_rho * self.bias_e + v_e
        self.bias_n = self.cfg.bias_rho * self.bias_n + v_n

    def position_noise(self, lat, lon, region_cell: RegionBiasCell) -> Tuple[float,float,List[AttackFlag]]:
        # 高频 + 慢变 + 区域偏置 + 偶发spike
        flags = []
        de = self.rng.normal(0, self.cfg.pos_sigma_hf)
        dn = self.rng.normal(0, self.cfg.pos_sigma_hf)
        de += self.bias_e + region_cell.dpos_e
        dn += self.bias_n + region_cell.dpos_n
        if self.rng.rand() < max(self.cfg.spike_prob, region_cell.spike_prob):
            spike = self.rng.uniform(*self.cfg.spike_range_m)
            # 随机方向
            theta = self.rng.uniform(0, 2*3.14159)
            de += spike * np.cos(theta); dn += spike * np.sin(theta)
            flags.append(AttackFlag.OFFSET)
        return (de, dn, flags)

    def field_corrupt(self, raw: RawTxMsg) -> Dict[str,bool]:
        miss = {}
        # 例：draft偶发缺失或极值
        if self.rng.rand() < self.fcfg.draft_miss_p:
            raw.draft = 0.0; miss["draft"]=True
        # MMSI字符错
        if self.rng.rand() < self.fcfg.mmsi_charerr_p:
            raw.mmsi = self._perturb_mmsi(raw.mmsi)
            miss["mmsi"]=True
        # 其他字段同理……
        return miss

    def _perturb_mmsi(self, m:int)->int: ...
