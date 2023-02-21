#!/usr/bin/python3

import math

import numpy as np
from numpy.linalg import norm

from spatialmath import *
from spatialmath.base import *

from math_utils import *

class Controller():
    def __init__(self, 
                 max_v: float = 0.1,
                 max_w: float = 1.5):
        self._v_ref = 0
        self._w_ref = 0

        self._max_v = max_v
        self._max_w = max_w

        self._v_err = 0.01
        self._w_err = 0.01

        self._ld = 0.1

        self._at_target = False

    def calculate_reference_v(self, start_pose: np.ndarray, end_pose: np.ndarray):
        w_start_T = SE2(start_pose)
        w_end_T = SE2(end_pose)

        start_end_T = w_start_T.inv() @ w_end_T

        if start_end_T.t[0] <= self._v_err:
            self._at_target = True
        else:
            self._at_target = False
        
        if not self._at_target:
            self._v_ref = self._max_v
            yt = start_end_T.t[1]
            ld2 = self._ld ** 2    
            self._w_ref = np.min(np.array([2 * self._v_ref / ld2 * yt, self._max_w]))
        else:
            self._v_ref = 0.0
            self._w_ref = 0.0
        print(w_start_T)
        print(w_end_T)
        print("v_ref: ", self._v_ref)
        print("w_ref: ", self._w_ref)
        print("===")
        
    def at_target(self):
        return self._at_target

    def get_reference_v(self):
        return self._v_ref, self._w_ref
