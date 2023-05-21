# !/usr/bin/python3

import pygame
import numpy as np
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils

class DecentralisedFormation(DecentralisedBehavior):
    def __init__(self):
        super().__init__()
        
    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray):
        # Control signal
        u = np.zeros((1, 2))
        all_shepherd_states = np.vstack((state, other_states))