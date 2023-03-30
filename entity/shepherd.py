# !/usr/bin/python3
import pygame
import numpy as np

from entity.entity import Autonomous, Entity
from utils.math_utils import *
from app import params


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

        self._r = 100

    def follow_mouse(self):
        mouse_pose = pygame.mouse.get_pos()
        self.move_to_pose(np.array(mouse_pose))

    def display(self, screen: pygame.Surface, debug=False):
        pygame.draw.circle(screen, pygame.Color(
            'white'), center=self._pose, radius=self._r, width=3)
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
