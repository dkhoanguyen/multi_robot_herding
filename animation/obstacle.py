#!/usr/bin/python3

import math
import numpy as np

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
from spatialmath.base import *

from animation.animation_handle import AnimationHandle


class ObstacleHandle(AnimationHandle):
    def __init__(self,
                 pose: np.ndarray,
                 radius: float):
        super().__init__(pose)
        self._body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self._body.position = Vec2d(
            pose[0],
            pose[1]
        )
        self._bodies_dict['body'] = self._body
        self._shape = pymunk.Circle(self._body, radius, Vec2d(0,0))
        self._shape.friction = 1
        self._bodies_dict['shape'] = self._shape
