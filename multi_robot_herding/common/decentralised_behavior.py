# !/usr/bin/python3

import numpy as np
from multi_robot_herding.common.behavior import Behavior

from multi_robot_herding.entity.entity import Autonomous
from multi_robot_herding.entity.herd import Herd


class DecentralisedBehavior(Behavior):
    def __init__(self):
        super(DecentralisedBehavior, self).__init__()
        # State of this shepherd
        self._state = None
        # States of other shepherds
        self._other_states = None
        # States of herds
        self._herd_states = None

    def set_shepherd_state(self, state: np.ndarray):
        self._state = state

    def set_other_shepherd_states(self, other_states: np.ndarray):
        self._other_states = other_states

    def set_herd_states(self, herd_states: np.ndarray):
        self._herd_states = herd_states