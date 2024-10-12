# !/usr/bin/python3

import numpy as np


class SimplePController:
    def __init__(self, p_gain: float):
        self._p_gain = p_gain

    def step(self, current_x: np.ndarray,
             target_x: np.ndarray):
        return (target_x - current_x) / np.linalg.norm(target_x - current_x)
