#!/usr/bin/python3

import time

import numpy as np

from environment import Environment

from animation_handle import DDAnimationHandle
from kinematic_model import DifferentialDrive

from boid import Boid

from spatialmath.base import *


def rand(low, high):
    return np.random.uniform(low=low, high=high, size=1)[0]


def main():

    # Environment initialization
    width = 700
    height = 700

    env = Environment(width, height)

     # Create agents
    poses = np.array([300, 100, 0])

    sample_time = 1/60
    num_agents = 3
    flock = [Boid(np.array([rand(10, 690), rand(10, 600), rand(-3.14, 3.14)]),
                 np.array([rand(9, 50), 0, rand(-50, 50)])) for _ in range (num_agents)]
    
    for boid in flock:
        env.add(boid.animation)

    while True:
        for boid in flock:
            boid.apply_behavior(flock, sample_time)
        env.visualise()


if __name__ == '__main__':
    main()
