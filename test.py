#!/usr/bin/python3

import numpy as np

from environment import Environment
from agent import DifferentialDriveRobot

from spatialmath.base import *


def main():
    env = Environment()
    # Create agents
    pose = np.array([100, 100, 0])
    robot = DifferentialDriveRobot(pose)
    robots_animation = [robot.animation]

    env.update_robots(robots_animation)    

    while True:
        env.spin_once()


if __name__ == '__main__':
    main()
