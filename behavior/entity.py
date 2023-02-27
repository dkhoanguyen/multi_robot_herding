# !/usr/bin/python3

from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

class EntityType(Enum):
    AUTONOMOUS = 1
    ROBOT = 2
    OBSTACLE = 3

class Entity(ABC):
    def __init__(self, 
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 type: EntityType = EntityType.OBSTACLE):
        self._type = type
        self._pose = pose
        if type == EntityType.OBSTACLE:
            self._velocity = np.zeros(2)
        else:
            self._velocity = velocity # velocity vector

    @property
    def type(self):
        return self._type

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose

    @property
    def velocity(self):
        return self._velocity

    @abstractmethod
    def apply_behaviour(self,
                        entities: list,
                        dt: float) -> np.ndarray:
        pass
