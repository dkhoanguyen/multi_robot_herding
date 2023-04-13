# !/usr/bin/python3
import math
import time

import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Obstacle

class FormationControl(Behavior):
    def __init__(self):
        self._radius = 50
        self._k = 0.01
        self._kd = 1
        self._l = 12.5
        self._herd_mean = np.zeros(2)
        self._herds = []
        self._shepherds = []

    def _generate_formation(self, center: np.ndarray, radius: float):
        pass