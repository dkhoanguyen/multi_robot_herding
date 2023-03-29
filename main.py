import pygame
import numpy as np
from app import params
import matplotlib.pyplot as plt

from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Hyperplane, Sphere

from behavior.flock import Flock
from behavior.mathematical_flock import MathematicalFlock
from behavior.leader_follower import LeaderFollower, LeaderFollowerType
from environment.environment import Environment

def main():
    # Create cows
    NUMBER_OF_AGENTS = 100
    cows = []
    # Cow's properties
    local_perception = 200.0
    local_boundary = 30.0
    personal_space = 60.0
    mass = 20.0
    min_v = 0.0
    max_v = 2

    # rand_x = np.linspace(200, 800, num_cows)
    # rand_y = np.linspace(200, 500, num_cows)

    # # Create cow grid
    # grid_x = np.linspace(200, 800, num_cows)
    # grid_y_100 = np.ones((1, num_cows)) * 100
    # grid_100 = np.vstack((grid_x, grid_y_100))

    # grid_x = np.linspace(200, 800, num_cows)
    # grid_y_200 = np.ones((1, num_cows)) * 200
    # grid_200 = np.vstack((grid_x, grid_y_200))

    # # grid_x = np.linspace(200, 800, num_cows)
    # # grid_y_300 = np.ones((1, num_cows)) * 300
    # # grid_300 = np.vstack((grid_x, grid_y_300))

    # grid = np.hstack((grid_100, grid_200))
    # for i in range(grid.shape[1]):
    #     pos = grid.transpose()[i, :]
    #     angle = np.pi * (2 * np.random.rand() - 1)
    #     vel = vel = np.zeros((2))
    #     cow = Herd(pose=pos,
    #                 velocity=vel,
    #                 local_perception=local_perception,
    #                 local_boundary=local_boundary,
    #                 personal_space=personal_space,
    #                 mass=mass,
    #                 min_v=min_v,
    #                 max_v=max_v)
    #     cows.append(cow)

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
    yk = np.array([800,250])
    Rk = 100
    circle1 = Sphere(yk,Rk)
    obstacles.append(circle1)

    yk = np.array([800,500])
    Rk = 100
    circle2 = Sphere(yk,Rk)
    obstacles.append(circle2)

    # # Create shepherds
    # num_shepherds = 5
    # shepherds = []
    # # Shepherd's properties
    # local_perception = 200.0
    # local_boundary = 30.0
    # personal_space = 60.0
    # mass = 20.0
    # min_v = 0.0
    # max_v = 5.0

    # pos = np.array([500, 500])
    # angle = 0
    # vel = max_v * np.array([np.cos(angle), np.sin(angle)])
    # # Leader shepherds
    # leader = Shepherd(pose=pos,
    #                   velocity=vel,
    #                   local_perception=local_perception,
    #                   local_boundary=local_boundary,
    #                   mass=mass,
    #                   min_v=min_v,
    #                   max_v=max_v)

    # shepherds.append(leader)

    # # Follower shepherds
    # rand_pos = np.linspace((200), (200, 800), num_shepherds)
    # for i in range(num_shepherds):
    #     pos = rand_pos[i, :]
    #     angle = np.pi * (2 * np.random.rand() - 1)
    #     vel = np.zeros((1,2))
    #     shepherd = Shepherd(pose=pos,
    #                         velocity=vel,
    #                         local_perception=local_perception,
    #                         local_boundary=local_boundary,
    #                         mass=mass,
    #                         min_v=min_v,
    #                         max_v=max_v)
    #     shepherds.append(shepherd)

    # Create behaviors
    # Flock properties
    alignment_weight = 5.0
    cohesion_weight = 0.05
    separation_weight = 20.0
    fleeing_weight = 5.0
    flock = Flock(
        alignment_weight=alignment_weight,
        cohesion_weight=cohesion_weight,
        separation_weight=separation_weight,
        fleeing_weight=fleeing_weight)

    # Mathematical flock
    math_flock = MathematicalFlock()

    # Add cows
    for cow in cows:
        flock.add_member(cow)
        math_flock.add_herd(cow)

    # Add obstacles
    for obstacle in obstacles:
        math_flock.add_obstacle(obstacle)

    # # Add shepherd
    # for shepherd in shepherds:
    #     math_flock.add_shepherd(shepherd)

    # # Formation
    # formation = LeaderFollower(
    #     LeaderFollowerType.COLUMN,
    #     formation_weight=1.0,
    #     spacing=40.0)

    # formation.add_leader(shepherds[0])
    # for i in range(1, num_shepherds):
    #     formation.add_follower(shepherds[i])

    # Environment
    env = Environment()

    # Add entities
    for cow in cows:
        env.add_entity(cow)
    # for shepherd in shepherds:
    #     env.add_entity(shepherd)
    for obstacle in obstacles:
        env.add_entity(obstacle)

    # # Add behavior models
    # env.add_behaviour(flock)
    env.add_behaviour(math_flock)
    # # env.add_behaviour(formation)

    while True:
        env.run_once()
        env.render()


if __name__ == '__main__':
    main()
