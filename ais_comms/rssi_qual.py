class RSSIandQual:
    def __init__(self, cfg: RSSIQualParams):
        self.cfg = cfg

    def rssi_proxy(self, rng, dist_m: float, region_cell: RegionBiasCell) -> float:
        s = self.cfg.s0 - 20*np.log10(max(1.0, dist_m))
        s += rng.normal(0, self.cfg.fading_sigma + region_cell.fading_sigma)
        return s

    def qual(self, rssi: float, age: float, field_anom: float) -> float:
        # 0..1
        score = (self.cfg.w_rssi * self._norm_rssi(rssi)
                + self.cfg.w_age  * self._sig_age(age)
                + self.cfg.w_field* (1 - field_anom))
        return 1/(1+np.exp(-score))

    def _norm_rssi(self, s): ...
    def _sig_age(self, age): ...
