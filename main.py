#!/usr/bin/python3

import yaml

from multi_robot_herding.entity.shepherd import Shepherd

from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.behavior.decentralised_surrounding import DecentralisedSurrounding
from multi_robot_herding.behavior.decentralised_formation import DecentralisedFormation

from multi_robot_herding.environment.environment import Environment
from multi_robot_herding.environment.spawner import Spawner


def single_iteration():
    config = 'default_config.yml'
    # Read yaml and extract configuration
    with open(f'config/{config}', 'r') as file:
        config = yaml.safe_load(file)

    entities = []
    # Component Config
    entity_config = config['entity']
    behavior_config = config['behavior']

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
    surround_config = behavior_config['surround']
    shepherd: Shepherd
    for shepherd in shepherds:
        shepherd_behaviors = {}
        dec_surround = DecentralisedSurrounding(**surround_config['params'])
        shepherd_behaviors["dec_surround"] = dec_surround
        shepherd.add_behavior(shepherd_behaviors)

    # Update entities list
    entities = entities + shepherds

    # Behavior related configuration
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
    env_config = config['sim']
    max_t = env_config['max_t']
    collect_data = env_config['collect_data']
    env = Environment(render=env_config['render'],
                      config=config)
    for entity in entities:
        env.add_entity(entity)

    start_t = 0
    env.add_behaviour(math_flock)
    while env.ok:
        env.run_once()
        env.render()
        if env.quit():
            env.save_data()
        if collect_data and start_t > max_t:
            env.save_data()
            break
        start_t += 1

if __name__ == '__main__':
    total_run = 10
    multi_thread = False
    current_run = 0
    while current_run < total_run:
        single_iteration()
        current_run += 1
