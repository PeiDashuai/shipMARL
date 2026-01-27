from dataclasses import dataclass

@dataclass
class SpawnConfig:
    N: int = 2
    spawn_area: float = 240.0
    spawn_margin: float = 12.0
    spawn_min_sep: float = 40.0
    spawn_goal_min_sep: float = 60.0
    spawn_len: float = 160.0
    spawn_retry: int = 80
    spawn_dir_jitter_deg: float = 6.0
    collide_thr: float = 12.0
    v_min: float = 0.1
    v_max: float = 2.0
    mode: str = "random_fixedlen"  # or "circle_center"
