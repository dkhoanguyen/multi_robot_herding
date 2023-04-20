# !/usr/bin/python3

import pymunk
import pygame
from enum import Enum
from app import params
from app.utils import *

from spatialmath import SE2
from spatialmath.base import *

import numpy as np

from entity.entity import Entity


class VisualisationEntity(Entity):
    def __init__(self):
        super(VisualisationEntity, self).__init__(
            pose=np.array([0, 0]),
            velocity=np.array([0, 0]),
            image_path='leader-boid.png',
            mass=0,
            type=0)
        self.boundaries = None
        self.orbit = None

    def vis_boundary(self, screen):
        if self.boundaries is None:
            return
        for idx in range(self.boundaries.shape[0] - 1):
            pygame.draw.line(screen, pygame.Color("white"), tuple(
                self.boundaries[idx, :]), tuple(self.boundaries[idx + 1, :]))
        pygame.draw.line(screen, pygame.Color("white"), tuple(
            self.boundaries[self.boundaries.shape[0] - 1, :]), tuple(self.boundaries[0, :]))

    def vis_orbit(self, screen):
        if self.orbit is None:
            return
        if len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                # print(point)
                # x = x * self.SCALE + WIDTH / 2
                # y = y * self.SCALE + HEIGHT / 2
                updated_points.append(tuple(point))

            pygame.draw.lines(screen, pygame.Color("white"), False, updated_points, 2)
        pygame.draw.circle(screen, pygame.Color(
                'white'), center=(500,500), radius=5, width=3)

    def display(self, screen: pygame.Surface, debug=False):
        super().display(screen)
        self.vis_boundary(screen)
        self.vis_orbit(screen)
