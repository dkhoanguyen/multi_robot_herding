#!/usr/bin/python3

import numpy as np

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d


from agent import Agent, DifferentialDriveRobot
from animation_handle import AnimationHandle
from kinematic_model import KinematicModel, DifferentialDrive

from spatialmath.base import *
from environment import Environment

def main():
    # Create agents
    pose = np.array([100,100,0])
    robot = DifferentialDriveRobot(pose)
    robots = [robot]
    
    env = Environment()
    env.update_robots(robots)

    while True:
        env.spin_once()

if __name__ == '__main__':
    main()