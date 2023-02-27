# !/usr/bin/python3

import math
import numpy as np

from behavior.entity import Entity, EntityType


class Obstacle(Entity):
    def __init__(self,
                 pose: np.ndarray,
                 radius: float):
        super().__init__(pose=pose, velocity=np.zeros(2), type=EntityType.OBSTACLE)
        self._radius = radius

    def apply_behaviour(self, entities: list):
        return None
