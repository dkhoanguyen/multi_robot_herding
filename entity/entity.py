# !/usr/bin/python3

import pymunk
import pygame
from enum import Enum
from app import params
from app.utils import *

from spatialmath import SE2
from spatialmath.base import *

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
        self._image_path = "/Users/khoanguyen/Projects/research/mrp_sims/assets/img/" + image_path
        self.base_image = pygame.image.load(self._image_path)
        self.rect = self.base_image.get_rect()
        self.image = self.base_image

        self.type = type
        self.mass = mass
        self._pose = pose
        self._velocity = velocity
        self._state = np.hstack((pose, velocity))

        angle = -np.rad2deg(np.angle(velocity[0] + 1j * velocity[1]))
        self._heading = np.deg2rad(angle)

        self._pymunk_addables = {}

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

    @property
    def heading(self):
        return self._heading

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def get_pymunk_addables(self):
        return self._pymunk_addables

    def _rotate_image(self, vector: np.ndarray):
        """Rotate base image using the velocity and assign to image."""
        angle = -np.rad2deg(np.angle(vector[0] + 1j * vector[1]))
        self._heading = np.deg2rad(angle)
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

        # self._heading =
        self._steering = np.zeros(2)
        self._wandering_angle = randrange(-np.pi, np.pi)

        self._min_v = min_v
        self._max_v = max_v

        self._speed = 0.5 * self._max_v
        self._pre_vel = np.array([0, 0])
        self._at_pose = False

        # Additional params for testing
        self._force = np.zeros(2)
        self._force_mag = 0
        self._plot_force = False
        self._plot_force_mag = False

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

    @property
    def at_pose(self):
        return self._at_pose

    def steer(self, force, alt_max):
        self._steering += truncate(force / self.mass, alt_max)

    def update(self):
        if self._speed == 0.0 and \
                not np.array_equal(self._velocity, np.array([0.0, 0.0])):
            self._pre_vel = self._velocity.copy()
            self._rotate_image(self._pre_vel)
            self._velocity = truncate(
                self._velocity + self._steering, self._speed)

        elif self._speed != 0.0:
            self._velocity = truncate(
                self._velocity + self._steering, self._speed)
            self._rotate_image(self._velocity)

        self.pose = self.pose + self._velocity

    def display(self, screen: pygame.Surface, debug=False):
        super().display(screen)
        # pygame.draw.line(
        #     screen, pygame.Color("yellow"),
        #     tuple(self.pose), tuple(self.pose + 2 * self.velocity))
        if self._plot_force:
            pygame.draw.line(
                screen, pygame.Color("white"),
                tuple(self.pose), tuple(self.pose + 200 * self._force))
        if self._plot_force_mag:
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=self._pose, radius=5, width=3)
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=self._pose, radius=self._force_mag, width=3)
        self.reset_steering()

    def reset_steering(self):
        self._steering = np.zeros(2)

    # Higher level control
    def move_to_pose(self, pose: np.ndarray):
        force = pose - self.pose
        desired_speed = norm(force)
        if desired_speed <= 1.0:
            desired_speed = 0
            self._at_pose = True
        else:
            self._at_pose = False
        self._speed = np.clip(desired_speed, self._min_v, self._max_v)
        self.steer(force - self.velocity,
                   alt_max=params.BOID_MAX_FORCE)

    def follow_velocity(self, velocity: np.ndarray):
        v = 100
        w = 1
        dt = 0.05

        self._heading += w * dt
        self.pose = self.pose + np.array([math.cos(self._heading),
                                          math.sin(self._heading)]) * v * dt
        self.velocity = np.array([math.cos(self._heading),
                                  math.sin(self._heading)]) * v * dt
        angle = np.rad2deg(-self._heading)

        self.image = pygame.transform.rotate(self.base_image, angle)
        self.rect = self.image.get_rect(center=self.rect.center)
        print(self.pose)
