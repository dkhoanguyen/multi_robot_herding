#!/usr/bin/python3

import numpy as np
from abc import ABC, abstractmethod


class KinematicModel(ABC):
    def __init__(self) -> None:
        self._pose = np.identity(3)
        self._forward_matrix = np.zeros([3, 1])
        self._inverse_matrix = np.zeros([3, 1])

    @abstractmethod
    def forward_kinematic(self, wheel_velocities: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_kinematic(self, body_velocities: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def compute_kinematic_matrices(self):
        pass


class DifferentialDrive(KinematicModel):
    def __init__(self,
                 pose: np.ndarray,
                 wheel_base: float,
                 wheel_radius: float):
        super().__init__()
        self._pose = pose
        self._wheel_base = wheel_base
        self._wheel_radius = wheel_radius

        self.compute_kinematic_matrices()

    def forward_kinematic(self, wheel_velocities: np.ndarray) -> np.ndarray:
        w_l = wheel_velocities[0]
        w_r = wheel_velocities[1]
        return np.array([
            0.5 * self._wheel_radius * (w_l + w_r),
            0,
            (w_r - w_l) * self._wheel_radius / self._wheel_base
        ])

    def inverse_kinematic(self, body_velocities: np.ndarray) -> np.ndarray:
        v = body_velocities[0]
        w = body_velocities[2]

        return np.array([
            (v - w * self._wheel_base / 2) / self._wheel_radius,
            (v + w * self._wheel_base / 2) / self._wheel_radius
        ])

    def compute_kinematic_matrices(self):
        return