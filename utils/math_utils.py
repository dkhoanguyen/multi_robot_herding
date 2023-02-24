#!/usr/bin/python3

import numpy as np


def angle_between(v1: np.ndarray, v2: np.ndarray):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
