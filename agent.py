#!/usr/bin/python3

import numpy as np
from kinematic_model import KinematicModel, DifferentialDrive
from animation_handle import AnimationHandle, DDAnimationHandle


class Agent(object):
    def __init__(self,
                 pose: np.ndarray) -> None:
        self._pose = np.ndarray([3, 1])
        self._kinematic_model: KinematicModel = None
        self._animation_handle = None

    @property
    def pose(self):
        return self._pose


class DifferentialDriveRobot(Agent):
    def __init__(self,
                 pose: np.ndarray,
                 ) -> None:
        super().__init__(pose)
        self._wheel_base = 10
        self._wheel_radius = 5
        self._base_radius = 5
        self._kinematic_model = DifferentialDrive(
            pose, self._wheel_base, self._wheel_radius)

        self._animation_handle = DDAnimationHandle(
            pose, self._base_radius, self._wheel_base, 5)

