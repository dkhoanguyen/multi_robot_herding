# !/usr/bin/python3
import math

from spatialmath.base import *

import numpy as np
from src.multi_robot_herding.utils import params, utils
from src.multi_robot_herding.behavior.behavior import Behavior
from src.multi_robot_herding.entity.herd import Herd
from src.multi_robot_herding.entity.shepherd import Shepherd
from src.multi_robot_herding.entity.obstacle import Obstacle


class Orbit(Behavior):
    AU = 149.6e6 * 1000
    # G = 6.67428e-11
    G = 667.4
    SCALE = 250 / AU  # 1AU = 100 pixels
    TIMESTEP = 3600*24  # 1 day

    def __init__(self):
        super().__init__()
        self._entity = None
        self._orbit = []
        self._enter_orbit = False

    def add_entity(self, entity: Shepherd):
        self._entity = entity

    def update(self, *args, **kwargs):
        # center = np.array([500, 500])
        # force = self._attraction(center)

        # self._entity._plot_velocity = True

        # # print(f"force: {force}")
        # self._entity.velocity = self._entity.velocity + force / 2 * 0.05
        # self._entity.pose = self._entity.pose + self._entity.velocity * 0.05
        # self._orbit.append(self._entity.pose.copy())

        # self._vis_entity.orbit = self._orbit

        # self._entity._rotate_image(self._entity.velocity)
        # self._entity.reset_steering()
        self._entity.follow_mouse()
        self._entity.update()

    def _attraction(self, pose: np.ndarray):
        pose_entity = pose - self._entity.pose
        distance = np.linalg.norm(pose_entity)
        entity_mass = 2
        other_mass = 385

        if distance > 175.5 and not self._enter_orbit:
            return np.zeros(2)
        else:
            if not self._enter_orbit:
                self._enter_orbit = True
        force = self.G * entity_mass * other_mass / distance ** 2
        theta = math.atan2(pose_entity[1], pose_entity[0])

        return np.array([force * math.cos(theta), force * math.sin(theta)])
