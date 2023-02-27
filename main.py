#!/usr/bin/python3

import time

import numpy as np
from spatialmath.base import *

from environment.environment import Environment
from behavior.boid import Entity, ClassicBoid
from behavior.obstacle import Obstacle
from kinematic.kinematic_model import DifferentialDrive
from animation.animation_handle import DDAnimationHandle
from animation.obstacle import ObstacleHandle


def rand(low, high):
    return np.random.uniform(low=low, high=high, size=1)[0]


def main():

    # Environment initialization
    width = 1000
    height = 1000

    env = Environment(width, height)

    sample_time = 1/60
    num_agents = 20

    # Obstacle
    num_obs = 1
    obstacles = []
    obs1 = Obstacle(np.array([rand(750, 750),
                              rand(750, 750),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs2 = Obstacle(np.array([rand(750, 750),
                              rand(600, 600),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs3 = Obstacle(np.array([rand(750, 750),
                              rand(450, 450),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs4 = Obstacle(np.array([rand(750, 750),
                              rand(300, 300),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs5 = Obstacle(np.array([rand(750, 750),
                              rand(150, 150),
                              rand(-3.14, 3.14)]),
                    45.0)

    obs6 = Obstacle(np.array([rand(800, 800),
                              rand(750, 750),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs7 = Obstacle(np.array([rand(800, 800),
                              rand(600, 600),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs8 = Obstacle(np.array([rand(800, 800),
                              rand(450, 450),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs9 = Obstacle(np.array([rand(800, 800),
                              rand(300, 300),
                              rand(-3.14, 3.14)]),
                    45.0)
    obs10 = Obstacle(np.array([rand(800, 800),
                              rand(150, 150),
                              rand(-3.14, 3.14)]),
                     45.0)

    # obstacles.append(obs1)
    # obstacles.append(obs2)
    # obstacles.append(obs3)
    # obstacles.append(obs4)
    # obstacles.append(obs5)

    # obstacles.append(obs6)
    # obstacles.append(obs7)
    # obstacles.append(obs8)
    # obstacles.append(obs9)
    # obstacles.append(obs10)

    obs_animations = []
    for obs in obstacles:
        obs_animation = ObstacleHandle(obs.pose, obs._radius)
        obs_animations.append(obs_animation)
        env.add(obs_animation)

    # Behaviours
    flock = [ClassicBoid(np.array([rand(10, 400),
                                   rand(10, 600),
                                   rand(0, 0)]),         # Pose
                         np.array([rand(50, 50),
                                   rand(0, 0)]),      # Velocity
                         1.0,                            # Alignment weight
                         1.0,                            # Cohesion weight
                         0.75,                            # Separation weight
                         200,                            # Local perception
                         200,                             # Local boundary
                         0,                              # Min vx
                         50)                           # Max vx
             for _ in range(num_agents)]

    # Animation
    flock_animations = []
    for entity in flock:
        entity_animation = DDAnimationHandle(entity.pose)
        flock_animations.append(entity_animation)
        env.add(entity_animation)

    # Kinematic Model
    wheel_base_length = 10
    wheel_radius = 5
    autonomous_entity_model = DifferentialDrive(
        wheel_base_length, wheel_radius)

    all_entities = flock + obstacles

    while True:
        entity: Entity
        for index, entity in enumerate(flock):
            ref_v = entity.apply_behaviour(all_entities, sample_time)
            ik_v = autonomous_entity_model.inverse_kinematic(ref_v)
            body_vel = autonomous_entity_model.forward_kinematic(ik_v)

            world_vel = Environment.body_to_world(body_vel, entity.pose)
            new_pose = entity.pose + world_vel * sample_time
            if new_pose[0] >= width:
                new_pose[0] = 0
            elif new_pose[0] < 0:
                new_pose[0] = width

            if new_pose[1] >= height:
                new_pose[1] = 0
            elif new_pose[1] < 0:
                new_pose[1] = height

            entity.pose = new_pose
            flock_animations[index].set_pose(new_pose)
            flock_animations[index].update(sample_time)

        env.visualise()


if __name__ == '__main__':
    main()
