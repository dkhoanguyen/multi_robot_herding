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
        super(VisualisationEntity, self).__init__()

    def display(self, screen: pygame.Surface, debug=False):
        super().display(screen)