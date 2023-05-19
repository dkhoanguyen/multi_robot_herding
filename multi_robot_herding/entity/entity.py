# !/usr/bin/python3
import pygame
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from spatialmath.base import *

from multi_robot_herding.utils import utils


class EntityType(Enum):
    AUTONOMOUS = 1
    ROBOT = 2
    OBSTACLE = 3


class DynamicType(Enum):
    Static = "static"
    SingleIntegrator = "single_integrator"
    DoubleIntegrator = "double_integrator"


class Entity(pygame.sprite.Sprite, ABC):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 image_path: str,
                 mass: float,
                 type: EntityType = EntityType.OBSTACLE):
        super().__init__()
        if pose is None:
            pose = np.zeros(2)
        if velocity is None:
            velocity = np.zeros(2)
        self._image_path = "/Users/khoanguyen/Projects/research/multi_robot_herding/assets/img/" + image_path
        self.base_image = pygame.image.load(self._image_path)
        self.rect = self.base_image.get_rect()
        self.image = self.base_image

        self._pose = pose
        self._velocity = velocity
        self._pre_velocity = velocity
        self._acceleration = np.zeros(2)

        angle = -np.rad2deg(np.angle(velocity[0] + 1j * velocity[1]))
        self._heading = np.deg2rad(angle)

        # Behavior
        self._behaviors = {}

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = pose
        self.rect.center = tuple(pose)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._pre_velocity = self._velocity.copy()
        self._velocity = velocity
        self._acceleration = self._velocity - self._pre_velocity

    @property
    def acceleration(self):
        return self._acceleration

    @abstractmethod
    def update(self, *args, **kwargs):
        pass

    def _rotate_image(self, vector: np.ndarray):
        """Rotate base image using the velocity and assign to image."""
        angle = -np.rad2deg(np.angle(vector[0] + 1j * vector[1]))
        self._heading = np.deg2rad(angle)
        self.image = pygame.transform.rotate(self.base_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def display(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)

    # Behaviors
    def add_behavior(self, behavior: dict):
        self._behaviors.update(behavior)


class Autonomous(Entity):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 image_path: str,
                 mass: float,
                 min_v: float,
                 max_v: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path=image_path,
            mass=mass,
            type=EntityType.AUTONOMOUS)

        # self._heading =
        self._steering = np.zeros(2)
        self._wandering_angle = utils.randrange(-np.pi, np.pi)

        self._min_v = min_v
        self._max_v = max_v

        self._speed = 0.5 * self._max_v
        self._pre_vel = np.array([0, 0])
        self._at_pose = False
        self._plot_velocity = False

        # Additional params for testing
        self._force = np.zeros(2)
        self._force_mag = 0
        self._plot_force = False
        self._plot_force_mag = False

        self._type = DynamicType.SingleIntegrator

    def display(self, screen: pygame.Surface, debug=False):
        super().display(screen)
        if self._plot_velocity:
            pygame.draw.line(
                screen, pygame.Color("yellow"),
                tuple(self.pose), tuple(self.pose + self.velocity))
        if self._plot_force:
            pygame.draw.line(
                screen, pygame.Color("white"),
                tuple(self.pose), tuple(self.pose + 75 * self._force))
        if self._plot_force_mag:
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=self._pose, radius=5, width=3)
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=self._pose, radius=self._force_mag, width=3)
