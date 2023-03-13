import pygame
import numpy as np
from entity.classic_boid import ClassicBoid
from entity.predator import Predator
from behavior.flock import Flock
from behavior.leader_follower import LeaderFollower, LeaderFollowerType
from environment.environment import Environment


def main():
    # Create cows
    num_cows = 10
    cows = []
    # Cow's properties
    local_perception = 200.0
    local_boundary = 30.0
    personal_space = 60.0
    mass = 20.0
    min_v = 0.0
    max_v = 35

    # rand_x = np.linspace(200, 800, num_cows)
    # rand_y = np.linspace(200, 500, num_cows)

    # Create cow grid
    grid_x = np.linspace(200, 800, num_cows)
    grid_y_100 = np.ones((1, num_cows)) * 100
    grid_100 = np.vstack((grid_x, grid_y_100))

    grid_x = np.linspace(200, 800, num_cows)
    grid_y_200 = np.ones((1, num_cows)) * 200
    grid_200 = np.vstack((grid_x, grid_y_200))

    grid_x = np.linspace(200, 800, num_cows)
    grid_y_300 = np.ones((1, num_cows)) * 300
    grid_300 = np.vstack((grid_x, grid_y_300))

    grid = np.hstack((grid_100, grid_200, grid_300))
    for i in range(grid.shape[1]):
        pos = grid.transpose()[i, :]
        angle = np.pi * (2 * np.random.rand() - 1)
        vel = max_v * np.array([np.cos(angle), np.sin(angle)])
        cow = ClassicBoid(pose=pos,
                          velocity=vel,
                          local_perception=local_perception,
                          local_boundary=local_boundary,
                          personal_space=personal_space,
                          mass=mass,
                          min_v=min_v,
                          max_v=max_v)
        cows.append(cow)

    # Create shepherds
    num_shepherds = 5
    shepherds = []
    # Shepherd's properties
    local_perception = 200.0
    local_boundary = 30.0
    personal_space = 60.0
    mass = 20.0
    min_v = 0.0
    max_v = 5.0

    pos = np.array([500, 500])
    angle = 0
    vel = 5.0 * np.array([np.cos(angle), np.sin(angle)])
    # Leader shepherds
    leader = Predator(pose=pos,
                      velocity=vel,
                      local_perception=local_perception,
                      local_boundary=local_boundary,
                      mass=mass,
                      min_v=min_v,
                      max_v=max_v)

    shepherds.append(leader)

    # Follower shepherds
    rand_pos = np.linspace((200), (200, 800), num_shepherds)
    for i in range(num_shepherds):
        pos = rand_pos[i, :]
        angle = np.pi * (2 * np.random.rand() - 1)
        vel = 5.0 * np.array([np.cos(angle), np.sin(angle)])
        shepherd = Predator(pose=pos,
                            velocity=vel,
                            local_perception=local_perception,
                            local_boundary=local_boundary,
                            mass=mass,
                            min_v=min_v,
                            max_v=max_v)
        shepherds.append(shepherd)

    # Create behaviors
    # Flock properties
    alignment_weight = 1.0
    cohesion_weight = 0.1
    separation_weight = 10.0
    fleeing_weight = 10.0
    flock = Flock(
        alignment_weight=alignment_weight,
        cohesion_weight=cohesion_weight,
        separation_weight=separation_weight,
        fleeing_weight=fleeing_weight)

    # Add cows
    for cow in cows:
        flock.add_member(cow)

    # Add shepherd
    # for shepherd in shepherds:
    # flock.add_predators(shepherds)

    # Formation
    formation = LeaderFollower(
        LeaderFollowerType.COLUMN,
        formation_weight=1.0,
        spacing=40.0)

    formation.add_leader(shepherds[0])
    for i in range(1, num_shepherds):
        formation.add_follower(shepherds[i])

    # Environment
    env = Environment()

    # Add entities
    for cow in cows:
        env.add_entity(cow)
    # for shepherd in shepherds:
    #     env.add_entity(shepherd)

    # Add behavior models
    env.add_behaviour(flock)
    # env.add_behaviour(formation)
    while True:
        env.run_once()


if __name__ == '__main__':
    main()
