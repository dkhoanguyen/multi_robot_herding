#!/usr/bin/python3

import math

import numpy as np
from numpy.linalg import norm

from spatialmath import *
from spatialmath.base import *

import pymunk

from kinematic_model import DifferentialDrive
from animation_handle import DDAnimationHandle

from math_utils import *
class DifferentialDriveRobot():
    def __init__(self,
                 pose: np.ndarray
                 ) -> None:
        super().__init__()
        self._pose = pose
        self._wheel_base_length = 10
        self._wheel_radius = 5
        self._wheel_width = 5

        self._kinematic_model = DifferentialDrive(
            self._wheel_base_length, self._wheel_radius)

        self._animation_handle = DDAnimationHandle(pose)

        #
        self._w_p_T = SE2(pose)

    @property
    def animation(self):
        return self._animation_handle

    def _get_pose_in_world(self, b_pose: np.ndarray):
        b_wp_T = SE2(b_pose)
        w_wp_T = self._w_p_T @ b_wp_T
        return np.array([w_wp_T.t[0], w_wp_T.t[1], w_wp_T.theta()])

    def _get_pose_in_body(self, w_pose: np.ndarray):
        w_wp_T = SE2(w_pose)
        b_wp_T = self._w_p_T.inv() @ w_wp_T
        return np.array([b_wp_T.t[0], b_wp_T.t[1], b_wp_T.theta()])

    def _get_reference_velocities(self, start_pose: np.ndarray, end_pose: np.ndarray):
        linear = norm(end_pose[0:1] - start_pose[0:1])
        angular = angle_between(start_pose[0:1], end_pose[0:1])
        if angular != 0.0:
            linear = 0.0
        return linear, angular
