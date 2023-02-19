#!/usr/bin/python3

import math

import numpy as np
from numpy.linalg import norm

from spatialmath import *
from spatialmath.base import *

from math_utils import *

class Controller():
    def __init__(self):
        self._v_ref = 0
        self._w_ref = 0

        self._v_err = 0.005

    def calculate_reference_v(self, start_pose: np.ndarray, end_pose: np.ndarray):
        dist = norm(start_pose[0:1] - end_pose[0:1])
        theta = angle_between(start_pose[0:1], end_pose[0:1])

        if theta != 0.0:
            self._v_ref = 0.0
        else:
            self._v_ref = dist

        self._w_ref = theta

    def get_reference_v(self):
        return self._v_ref, self._w_ref
