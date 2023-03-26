# !/usr/bin/python3
import pygame
import numpy as np

from entity.entity import Autonomous
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

    def follow_mouse(self):
        mouse_pose = pygame.mouse.get_pos()
        self.move_to_pose(np.array(mouse_pose))

    def wander(self, other_predators: list):
        rands = 2 * np.random.rand(len(other_predators)) - 1
        cos = np.cos([b.wandering_angle for b in other_predators])
        sin = np.sin([b.wandering_angle for b in other_predators])

        shepherd: Shepherd
        for i, shepherd in enumerate(other_predators):
            if self == shepherd:
                nvel = normalize(self.velocity)
                # calculate circle center
                circle_center = nvel * params.WANDER_DIST
                # calculate displacement force
                c, s = cos[i], sin[i]
                displacement = np.dot(
                    np.array([[c, -s], [s, c]]), nvel * params.WANDER_RADIUS)
                self.steer(circle_center + displacement,
                           alt_max=params.BOID_MAX_FORCE * 1.5)
                self.wandering_angle += params.WANDER_ANGLE * rands[i]

    def remain_in_screen(self):
        if self.pose[0] > params.SCREEN_WIDTH - params.BOX_MARGIN:
            self.steer(np.array([-params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE * 1.5)
        if self.pose[0] < params.BOX_MARGIN:
            self.steer(np.array([params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE * 1.5)
        if self.pose[1] < params.BOX_MARGIN:
            self.steer(np.array([0., params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE * 1.5)
        if self.pose[1] > params.SCREEN_HEIGHT - params.BOX_MARGIN:
            self.steer(np.array([0., -params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE * 1.5)
