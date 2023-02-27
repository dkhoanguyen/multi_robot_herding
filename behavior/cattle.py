# !/usr/bin/python3

import math
import numpy as np
from boid import ClassicBoid


class CattleBehaviour(ClassicBoid):
    def __init__(self,
                 pose: np.ndarray,
                 alighment_weight: float,
                 cohesion_weight: float,
                 separation_weight: float,
                 local_perception: float,
                 local_boundary: float):
        super().__init__(pose, alighment_weight, cohesion_weight,
                         separation_weight, local_perception, local_boundary)
        
    def velocity_matching(self, entities: list):
        pass

    def apply_behavior(self, entities: list):
        super().apply_behavior(entities)
