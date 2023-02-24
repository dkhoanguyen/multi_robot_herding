#!/usr/bin/python3

import pygame
import numpy as np

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from spatialmath.base import *


class AnimationHandle(pygame.sprite.Sprite):

    def __init__(self,
                 pose: np.ndarray):
        super().__init__()
        self._pose = pose

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
        self._shape = None

        self._pivot = pymunk.PivotJoint(
            self._control_body, self._body, (0, 0), (0, 0))
        self._pivot.max_bias = 0
        self._pivot.max_force = 10000

        self._gear = pymunk.GearJoint(self._control_body, self._body, 0.0, 1.0)
        self._gear.error_bias = 0
        self._gear.max_bias = 1.2
        self._gear.max_force = 50000

    def update(self, dt: float):
        print("Hey")

    def set_pose(self, pose):
        self._pose = pose


class DDAnimationHandle(AnimationHandle):
    def __init__(self,
                 pose: np.ndarray):
        super().__init__(pose)

        # Draw the shape
        self._shape = pymunk.Poly.create_box(self._body, (30, 15), 1.0)
        self._shape.mass = 1
        self._shape.friction = 0.7

    def update(self, dt: float):
        self._control_body.angle = self._pose[2]
        self._control_body.position = self._pose[0],self._pose[1]
        self._body.position = self._pose[0],self._pose[1]
