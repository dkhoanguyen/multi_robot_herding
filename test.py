#!/usr/bin/python3

import time

import numpy as np

from environment import Environment
from agent import DifferentialDriveRobot
from simple_controller import Controller

from animation_handle import DDAnimationHandle
from kinematic_model import DifferentialDrive

from spatialmath.base import *


def main():
    # Create agents
    poses = np.array([[300, 100, 1.57]])
    waypoint = np.array([200,200, 0])

    sample_time = 1/60
    sim_time_s = 20

    # Robot 
    wheel_base_length = 10
    wheel_radius = 5
    wheel_width = 5
    
    robot = DifferentialDrive(wheel_base_length, wheel_radius)
    robot_animation = DDAnimationHandle(poses[0])

    env = Environment()
    env.add(robot_animation)

    for i in np.arange(1, sim_time_s/sample_time, 1):
        
        [vref, wref] = [100, 0]
        w_v = robot.inverse_kinematic(np.array([vref, 0, wref]))
        v = robot.forward_kinematic(w_v)

        vel = Environment.body_to_world(v, poses[int(i) - 1])
        new_pose = poses[int(i) - 1] + vel * sample_time
        poses = np.vstack((poses, new_pose))

        robot_animation.set_pose(new_pose)
        robot_animation.update(None, sample_time)
        env.spin_once()

if __name__ == '__main__':
    main()
