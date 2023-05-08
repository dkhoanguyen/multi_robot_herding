#!/usr/bin/python3

import yaml

from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.shepherd import Shepherd
from multi_robot_herding.entity.obstacle import Hyperplane, Sphere
from multi_robot_herding.entity.visualise_agent import VisualisationEntity

from multi_robot_herding.behavior.behavior import Behavior
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.behavior.mathematical_formation import MathematicalFormation
from multi_robot_herding.behavior.bearing_formation import BearingFormation

from multi_robot_herding.environment.environment import Environment
from multi_robot_herding.environment.spawner import Spawner


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
    math_formation = BearingFormation(**math_formation_config['params'])

    # Add herds
    for herd in herds:
        math_formation.add_herd(herd)

    # # Add shepherd
    for shepherd in shepherds:
        math_formation.add_shepherd(shepherd)

    # Behaviours
    behaviors = []
    behaviors.append(math_flock)
    behaviors.append(math_formation)

    # Environment
    env = Environment()

    # Add entities
    for cow in herds:
        env.add_entity(cow)
    for shepherd in shepherds:
        env.add_entity(shepherd)
    for obstacle in obstacles:
        env.add_entity(obstacle)

    behavior: Behavior
    for behavior in behaviors:
        env.add_behaviour(behavior)

    while env.ok:
        env.run_once()
        env.render()

if __name__ == '__main__':
    main()
