import math
import numpy as np

def wrap_to_pi(x: float) -> float:
    return (x + math.pi) % (2 * math.pi) - math.pi

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)
