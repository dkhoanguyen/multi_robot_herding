#!/usr/bin/python3

import yaml

import pygame
import numpy as np
from app import params

from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Hyperplane, Sphere
from entity.visualise_agent import VisualisationEntity

from behavior.behavior import Behavior
from behavior.mathematical_flock import MathematicalFlock
from behavior.mathematical_formation import MathematicalFormation
from behavior.orbit import Orbit

from environment.environment import Environment
from environment.spawner import Spawner


def main():
    config = 'default_config.yml'
    # Read yaml and extract configuration
    with open(f'config/{config}', 'r') as file:
        config = yaml.safe_load(file)

    entities = []
    ## Entity related configuration
    entity_config = config['entity']
    # Create herds
    herd_config = entity_config['herd']
    herds = Spawner.auto_spawn_herds(herd_config)
    entities = entities + herds
    # Create obstacles
    obstacle_config = entity_config['obstacle']
    obstacles = Spawner.auto_spawn_obstacles(obstacle_config)
    entities = entities + obstacles
    # Create shepherds
    shepherd_config = entity_config['shepherd']
    shepherds = Spawner.auto_spawn_shepherds(shepherd_config)
    entities = entities + shepherds

    ## Behavior related configuration
    behavior_config = config['behavior']
    math_flock_config = behavior_config['math_flock']
    math_flock = MathematicalFlock(**math_flock_config['params'])

    # Add herds
    for herd in herds:
        math_flock.add_herd(herd)

    # Add obstacles
    for obstacle in obstacles:
        math_flock.add_obstacle(obstacle)

    # # Add shepherd
    for shepherd in shepherds:
        math_flock.add_shepherd(shepherd)

    math_formation_config = behavior_config['math_formation']
    math_formation = MathematicalFormation(**math_formation_config['params'])

    # Add herds
    for herd in herds:
        math_formation.add_herd(herd)

    # # Add shepherd
    for shepherd in shepherds:
        math_formation.add_shepherd(shepherd)

    # # Orbit
    # orbit = Orbit()
    # orbit.add_entity(shepherds[0])

    # Visualisation Entity
    vis_entity = VisualisationEntity()

    # Behaviours
    behaviors = []
    behaviors.append(math_flock)
    behaviors.append(math_formation)
    # behaviors.append(orbit)

    # Environment
    env = Environment()

    # Add entities
    for cow in herds:
        env.add_entity(cow)
    for shepherd in shepherds:
        env.add_entity(shepherd)
    for obstacle in obstacles:
        env.add_entity(obstacle)

    env.add_entity(vis_entity)

    behavior: Behavior
    for behavior in behaviors:
        behavior.set_vis_entity(vis_entity)
        env.add_behaviour(behavior)

    while env.ok:
        env.run_once()
        env.render()

if __name__ == '__main__':
    main()
