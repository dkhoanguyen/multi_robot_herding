# !/usr/bin/python3

import numpy as np
from multi_robot_herding.common.behavior import Behavior

from multi_robot_herding.entity.entity import Autonomous
from multi_robot_herding.entity.herd import Herd


class DecentralisedBehavior(Behavior):
    def __init__(self):
        super(DecentralisedBehavior, self).__init__()
        self._herds = []

    def add_herd(self, herd: Herd):
        self._herds.append(herd)

    def extract_agent_states(self, agents: list):
        agent: Autonomous
        states = np.array([]).reshape((0, 4))
        for agent in agents:
            # Grab and put all poses into a matrix
            states = np.vstack(
                (states, np.hstack((agent.pose, agent.velocity, agent.acceleration))))
        return agents