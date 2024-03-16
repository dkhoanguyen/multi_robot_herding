# !/usr/bin/python3

import numpy as np


def double_integrator_dynamic(x_t: np.ndarray, v_t: np.ndarray,
                              u_t: np.ndarray, dt=0.1):
    v_t1 = v_t + u_t * dt
    x_t1 = x_t + v_t1 * dt
    return x_t1, v_t1
