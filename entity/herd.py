# !/usr/bin/python3
import time
import pygame
import numpy as np

from entity.entity import Entity, Autonomous
from app.utils import *


class Herd(Autonomous):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 local_perception: float,
                 local_boundary: float,
                 personal_space: float,
                 mass: float,
                 min_v: float,
                 max_v: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='normal-boid.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)

        self._local_perception = local_perception
        self._local_boundary = local_boundary
        self._personal_space = personal_space

    @property
    def local_perception(self):
        return self._local_perception

    @property
    def local_boundary(self):
        return self._local_boundary

    @property
    def personal_space(self):
        return self._personal_space

    def in_local_perception(self, entity: Entity):
        pass
