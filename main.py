#!/usr/bin/python3

import pygame
import numpy as np
from app import params
import matplotlib.pyplot as plt

from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Hyperplane, Sphere
from entity.visualise_agent import VisualisationEntity

from behavior.mathematical_flock import MathematicalFlock
from behavior.formation_control import MathematicalFormation
from behavior.spiral_motion import SpiralMotion

from environment.environment import Environment


def main():
    # Create cows
    NUMBER_OF_AGENTS = 50
    cows = []
    # Cow's properties
    local_perception = 200.0
    local_boundary = 30.0
    personal_space = 60.0
    mass = 20.0
    min_v = 0.0
    max_v = 5

    agents = np.random.randint(50, 700, (NUMBER_OF_AGENTS, 2)).astype('float')

    for i in range(NUMBER_OF_AGENTS):
        angle = np.pi * (2 * np.random.rand() - 1)
        vel = max_v * np.array([np.cos(angle), np.sin(angle)])
        cow = Herd(pose=agents[i, :2],
                   velocity=vel,
                   local_perception=local_perception,
                   local_boundary=local_boundary,
                   personal_space=personal_space,
                   mass=mass,
                   min_v=min_v,
                   max_v=max_v)
        cows.append(cow)

    # Create obstacles
    obstacles = []
    # Environment boundaries
    ak = np.array([0, 1])
    yk = np.array([0, 0])

    def display_upper_boundary(screen):
        pygame.draw.rect(screen, pygame.Color(
            'slate gray'), (0, 0, params.SCREEN_SIZE[0], 10), 0)
    upper_boundary = Hyperplane(ak, yk, display_upper_boundary)
    obstacles.append(upper_boundary)

    ak = np.array([1, 0])
    yk = np.array([0, 0])

    def display_left_boundary(screen):
        pygame.draw.rect(screen,
                         pygame.Color('slate gray'),
                         (0, 0, 10, params.SCREEN_SIZE[1]), 0)
    left_boundary = Hyperplane(ak, yk, display_left_boundary)
    obstacles.append(left_boundary)

    ak = np.array([1, 0])
    yk = np.array([1279, 0])

    def display_right_boundary(screen):
        pygame.draw.rect(screen,
                         pygame.Color('slate gray'),
                         (params.SCREEN_SIZE[0] - 10, 0,
                          params.SCREEN_SIZE[0], params.SCREEN_SIZE[1]), 0)

    right_boundary = Hyperplane(ak, yk, display_right_boundary)
    obstacles.append(right_boundary)

    ak = np.array([0, 1])
    yk = np.array([0, 719])

    def display_lower_boundary(screen):
        pygame.draw.rect(screen,
                         pygame.Color('slate gray'),
                         (0, params.SCREEN_SIZE[1] - 10,
                             params.SCREEN_SIZE[0], params.SCREEN_SIZE[1]), 0)
    lower_boundary = Hyperplane(ak, yk, display_lower_boundary)
    obstacles.append(lower_boundary)

    # Spherial obstacles
    yk = np.array([1000, 100])
    Rk = 100
    circle1 = Sphere(yk, Rk)
    obstacles.append(circle1)

    yk = np.array([1000, 510])
    Rk = 100
    circle2 = Sphere(yk, Rk)
    obstacles.append(circle2)

    yk = np.array([1000, 240])
    Rk = 100
    circle3 = Sphere(yk, Rk)
    obstacles.append(circle3)

    yk = np.array([1000, 650])
    Rk = 100
    circle4 = Sphere(yk, Rk)
    obstacles.append(circle4)

    # Create shepherds
    num_shepherds = 6
    shepherds = []
    # Shepherd's properties
    local_perception = 200.0
    local_boundary = 30.0
    personal_space = 60.0
    mass = 20.0
    min_v = 0.0
    max_v = 3

    pos = np.array([600, 900])
    # pos = np.array([350, 350])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    pos = np.array([300, 900])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    pos = np.array([100, 900])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    pos = np.array([100, -100])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    pos = np.array([300, -100])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    pos = np.array([500, -100])
    angle = 0
    vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    shepherds.append(Shepherd(pose=pos,
                              velocity=vel,
                              local_perception=local_perception,
                              local_boundary=local_boundary,
                              mass=mass,
                              min_v=min_v,
                              max_v=max_v))

    # Mathematical flock
    follow_cursor = True
    sensing_range = 150
    danger_range = 2000
    initial_consensus = np.array([350, 350])
    math_flock = MathematicalFlock(
        follow_cursor=follow_cursor,
        sensing_range=sensing_range,
        danger_range=danger_range,
        initial_consensus=initial_consensus)

    # Add cows
    for cow in cows:
        # flock.add_member(cow)
        math_flock.add_herd(cow)

    # Add obstacles
    for obstacle in obstacles:
        math_flock.add_obstacle(obstacle)

    # Add shepherd
    for shepherd in shepherds:
        math_flock.add_shepherd(shepherd)

    # Mathematical formation
    math_formation = MathematicalFormation()
    math_formation.set_herd_mean(initial_consensus)
    math_formation.add_herd(math_flock)

    # Add cows
    for cow in cows:
        math_formation.add_herd(cow)

    # Add shepherd
    for shepherd in shepherds:
        math_formation.add_shepherd(shepherd)

    spiral = SpiralMotion()
    spiral.add_single_shepherd(shepherds[0])

    # Visualisation Entity
    vis_entity = VisualisationEntity()

    # Behaviours
    behaviours = []
    behaviours.append(math_flock)
    behaviours.append(math_formation)
    behaviours.append(spiral)

    # Environment
    env = Environment()

    # Add entities
    for cow in cows:
        env.add_entity(cow)
    for shepherd in shepherds:
        env.add_entity(shepherd)
    for obstacle in obstacles:
        env.add_entity(obstacle)

    # Add behavior models
    env.add_behaviour(math_flock)
    # env.add_behaviour(math_formation)
    # env.add_behaviour(spiral)


    while env.ok:
        env.run_once()
        env.render()


if __name__ == '__main__':
    main()
