#!/usr/bin/python3

import math

import pygame
import numpy as np
from numpy.linalg import inv

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from spatialmath.base import *


class AnimationHandle(pygame.sprite.Sprite):

    def __init__(self,
                 pose: np.ndarray):
        super().__init__()
        self._pose = pose
        self._w_p_T = transl2(pose[0], pose[1]) * trot2(pose[2])

        # "Visual" body
        self._body = pymunk.Body()
        self._body.position = Vec2d(
            pose[0],
            pose[1]
        )
        self._body.angle = pose[2]

        # Physic body
        self._control_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self._control_body.position = Vec2d(
            pose[0],
            pose[1]
        )
        self._control_body.angle = pose[2]

        # Variables for shape representation
        self._shape_body = pymunk.Body()
        self._shape_body.position = Vec2d(
            pose[0],
            pose[1]
        )
        self._shape_body.angle = pose[2]
        self._shape = None

        self._pivot = None
        self._gear = None

    def update(self, dt: float):
        pass


class DDAnimationHandle(AnimationHandle):
    def __init__(self,
                 pose: np.ndarray,
                 base_radius: int,
                 wheel_base: int,
                 wheel_width: int):
        super().__init__(pose)
        robot_w = wheel_width + 2 * base_radius
        robot_l = wheel_base if wheel_base >= 2 * base_radius \
            else 2 * base_radius

        sprite_width = robot_w
        sprite_height = robot_l

        # Draw the shape
        self._shape = pymunk.Poly.create_box(self._shape_body, (20, 20), 1.0)
        self._shape.mass = 1
        self._shape.friction = 0.7

        self._pivot = pymunk.PivotJoint(
            self._control_body, self._body, (0, 0), (0, 0))
        self._pivot.max_bias = 0
        self._pivot.max_force = 10000

        self._gear = pymunk.GearJoint(self._control_body, self._body, 0.0, 1.0)
        self._gear.error_bias = 0
        self._gear.max_bias = 1.2
        self._gear.max_force = 50000

    def update(self, dt: float):
        pass
