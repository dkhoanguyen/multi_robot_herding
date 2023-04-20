# !/usr/bin/python3

import pygame
import numpy as np
from spatialmath import SE2
from spatialmath.base import *

from entity.entity import Autonomous, Entity
from app.utils import *

class Shepherd(Autonomous):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 local_perception: float,
                 local_boundary: float,
                 mass: float,
                 min_v: float,
                 max_v: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='leader-boid.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)

        self._local_perception = local_perception
        self._local_boundary = local_boundary

        self._r = 80
        self._consensus_r = 200

        self._consensus_point = np.zeros((2,))

    def follow_mouse(self):
        mouse_pose = pygame.mouse.get_pos()
        self.move_to_pose(np.array(mouse_pose))

    def display(self, screen: pygame.Surface, debug=False):
        pygame.draw.circle(screen, pygame.Color(
            'white'), center=self._pose, radius=self._r, width=3)
        # pygame.draw.circle(screen, pygame.Color(
        #     'white'), center=self._consensus_point, radius=10, width=3)
        return super().display(screen, debug)

    def in_entity_radius(self, qi: np.ndarray, r: float):
        # Project entity posit
        return np.linalg.norm(self._pose - qi) <= r + self._r

    def induce_delta_agent(self, alpha_agent: Entity):
        qi = alpha_agent.pose.reshape((2, 1))
        pi = alpha_agent.velocity.reshape((2, 1))
        yk = self._pose.reshape((2, 1))
        d = np.linalg.norm(qi - yk)

        if d < self._r:
            mu = d / np.linalg.norm(qi - yk)
            ak = (qi - yk)/np.linalg.norm(qi - yk)
            P = np.eye(2) - ak @ ak.transpose()

            qik = mu * qi + (1 - mu) * yk
            pik = mu * P @ pi
            return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)
        else:
            mu = self._r / np.linalg.norm(qi - yk)
            ak = (qi - yk)/np.linalg.norm(qi - yk)
            P = np.eye(2) - ak @ ak.transpose()

            qik = mu * qi + (1 - mu) * yk
            pik = mu * P @ pi
            return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)

    # Immitate how herds should be moving away from shepherds
    def induce_consesus_point(self, r=400):
        angle = -self._heading
        consensus_point = transl2(
            self._pose) @ trot2(angle) @ transl2(np.array([r, 0]))

        # Save this value for visualisation purpose only
        self._consensus_point = consensus_point[0:2, 2]
        return self._consensus_point