# !/usr/bin/python3
import pygame
import numpy as np

from abc import abstractmethod
from entity.entity import Entity
from utils.math_utils import *


class Obstacle(Entity):

    @abstractmethod
    def in_entity_radius(self, qi: np.ndarray, r: float):
        pass

    @abstractmethod
    def induce_beta_agent(self, alpha_agent: Entity):
        pass


class Hyperplane(Obstacle):
    def __init__(self, ak: np.ndarray,
                 yk: np.ndarray):
        self._ak = ak.reshape((2, 1))
        self._yk = yk.reshape((2, 1))
        self._P = np.eye(2) - self._ak @ self._ak.transpose()

    def in_entity_radius(self, qi: np.ndarray, r: float):
        # Project entity position onto the plane
        projected_q = self._P @ qi + (np.eye(2) - self._P) @ self._yk
        return np.linalg.norm(projected_q - qi) <= r

    def induce_beta_agent(self, alpha_agent: Entity) -> np.ndarray:
        qi = alpha_agent.pose.reshape((2, 1))
        pi = alpha_agent.velocity.reshape((2, 1))

        qik = self._P @ qi + (np.eye(2) - self._P) @ self._yk
        pik = self._P @ pi
        return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)


class Sphere(Obstacle):
    pass
