# !/usr/bin/python3

import pygame
from utils.math_utils import *
from enum import Enum
from app import assets
from app import utils
from app import params

import numpy as np


class EntityType(Enum):
    AUTONOMOUS = 1
    ROBOT = 2
    OBSTACLE = 3


class Entity(pygame.sprite.Sprite):

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
        self._image_path = image_path
        self.base_image, self.rect = assets.image_with_rect(self._image_path)
        self.image = self.base_image

        self.type = type
        self.mass = mass
        self._pose = pose
        self._velocity = velocity

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
        self._velocity = velocity
        self._rotate_image()

    def _rotate_image(self):
        """Rotate base image using the velocity and assign to image."""
        angle = -np.rad2deg(np.angle(self.velocity[0] + 1j * self.velocity[1]))
        self.image = pygame.transform.rotate(self.base_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)

    def display(self, screen: pygame.Surface):
        screen.blit(self.image, self.rect)


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

        self._steering = np.zeros(2)
        self._wandering_angle = utils.randrange(-np.pi, np.pi)

        self._min_v = min_v
        self._max_v = max_v

        self._speed = 0.5 * self._max_v

    @property
    def wandering_angle(self):
        return self._wandering_angle

    @wandering_angle.setter
    def wandering_angle(self, value):
        self._wandering_angle = value

    @property
    def speed(self):
        return self._speed

    @speed.setter
    def speed(self, value):
        self._speed = value

    def steer(self, force, alt_max):
        self._steering += utils.truncate(force / self.mass, alt_max)

    def update(self):
        self.velocity = utils.truncate(
            self.velocity + self._steering, self._speed)
        self.pose = self.pose + self.velocity

    def display(self, screen: pygame.Surface, debug=False):
        super().display(screen)
        if debug:
            pygame.draw.line(
                screen, pygame.Color("red"),
                tuple(self.pose), tuple(self.pose + 2 * self.velocity))
            pygame.draw.line(
                screen, pygame.Color("blue"), tuple(self.pose),
                tuple(self.pose + 30 * self._steering))

    def reset_steering(self):
        self._steering = np.zeros(2)
