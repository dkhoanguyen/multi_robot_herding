# !/usr/bin/python3

import numpy as np
from multi_robot_herding.common.behavior import Behavior

from multi_robot_herding.entity.entity import Autonomous
from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.obstacle import Obstacle


class DecentralisedBehavior(Behavior):
    def __init__(self):
        super(DecentralisedBehavior, self).__init__()