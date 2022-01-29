import numpy as np


def det2d(fst: np.ndarray, snd: np.ndarray) -> float:
    return fst[0] * snd[1] - fst[1] * snd[0]
