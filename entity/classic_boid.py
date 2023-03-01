# !/usr/bin/python3
import pygame
import numpy as np

from entity.entity import Autonomous
from utils.math_utils import *
from app import params
from app import utils


class ClassicBoid(Autonomous):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 alighment_weight: float,
                 cohesion_weight: float,
                 separation_weight: float,
                 local_perception: float,
                 local_boundary: float,
                 mass: float,
                 min_v: float,
                 max_v: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='normal-boid.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)

        self._alighment_weight = alighment_weight
        self._cohesion_weight = cohesion_weight
        self._separation_weight = separation_weight

        self._local_perception = local_perception
        self._local_boundary = local_boundary

    # Implement interaction behaviours here in the future
