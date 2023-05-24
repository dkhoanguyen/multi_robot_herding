#!/usr/bin/python3

import yaml

from multi_robot_herding.entity.shepherd import Shepherd

from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.behavior.decentralised_approaching import DecentralisedApproaching
from multi_robot_herding.behavior.decentralised_surrounding import DecentralisedSurrounding
from multi_robot_herding.behavior.decentralised_formation import DecentralisedFormation

from multi_robot_herding.environment.environment import Environment
from multi_robot_herding.environment.spawner import Spawner


def main():
    config = 'default_config.yml'
    # Read yaml and extract configuration
    with open(f'config/{config}', 'r') as file:
        config = yaml.safe_load(file)

    entities = []
    # Entity related configuration
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

    # Shepherd behavior

    shepherd: Shepherd
    for shepherd in shepherds:
        shepherd_behaviors = {}
        dec_approach = DecentralisedApproaching()
        shepherd_behaviors["dec_approach"] = dec_approach
        dec_surround = DecentralisedSurrounding()
        shepherd_behaviors["dec_surround"] = dec_surround
        dec_formation = DecentralisedFormation(id=shepherd.id)
        shepherd_behaviors["dec_formation"] = dec_formation
        shepherd.add_behavior(shepherd_behaviors)

    # Update entities list
    entities = entities + shepherds

    # Behavior related configuration
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

    # Environment
    env = Environment()
    for entity in entities:
        env.add_entity(entity)

    env.add_behaviour(math_flock)

    while env.ok:
        env.run_once()
        env.render()


if __name__ == '__main__':
    main()
