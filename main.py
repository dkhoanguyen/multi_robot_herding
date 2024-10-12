#!/usr/bin/python3

import yaml
import numpy as np

from multi_robot_herding.entity.shepherd import Shepherd
from multi_robot_herding.entity.robot import Robot

from multi_robot_herding.behavior.decentralised_cbf import DecentralisedCBF
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
# from multi_robot_herding.behavior.decentralised_surrounding import DecentralisedSurrounding

from multi_robot_herding.environment.environment import Environment
from multi_robot_herding.environment.spawner import Spawner


def main():
    max_v = 10.0
    robot1 = Robot(id=1,
                   pose=np.array([300, 300]),
                   velocity=np.array([10, 0]),
                   local_perception=100,
                   target_pose=np.array([500, 300]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)
    
    controller1 = DecentralisedCBF(target_pos=np.array([510, 300]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot1.add_behavior({"cbf": controller1})
    
    robot2 = Robot(id=2,
                   pose=np.array([500, 300]),
                   velocity=np.array([-10, 0]),
                   local_perception=100,
                   target_pose=np.array([300, 300]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller2 = DecentralisedCBF(target_pos=np.array([300, 300]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot2.add_behavior({"cbf": controller2})

    robot3 = Robot(id=3,
                   pose=np.array([400, 200]),
                   velocity=np.array([0, 1.0]),
                   local_perception=100,
                   target_pose=np.array([400, 400]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller3 = DecentralisedCBF(target_pos=np.array([400, 400]),
                                  controller_gain=np.array([1.0, 0, 0]))
    robot3.add_behavior({"cbf": controller3})

    robot4 = Robot(id=4,
                   pose=np.array([400, 400]),
                   velocity=np.array([0, -1.0]),
                   local_perception=100,
                   target_pose=np.array([400, 200]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller4 = DecentralisedCBF(target_pos=np.array([400, 200]),
                                  controller_gain=np.array([1.0, 0, 0]))
    robot4.add_behavior({"cbf": controller4})
    
    robot5 = Robot(id=5,
                   pose=np.array([320, 220]),
                   velocity=np.array([0, 1.0]),
                   local_perception=100,
                   target_pose=np.array([480, 380]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller5 = DecentralisedCBF(target_pos=np.array([480, 380]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot5.add_behavior({"cbf": controller5})
    
    robot6 = Robot(id=6,
                   pose=np.array([480, 380]),
                   velocity=np.array([0, 1.0]),
                   local_perception=100,
                   target_pose=np.array([320, 220]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller6 = DecentralisedCBF(target_pos=np.array([320, 220]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot6.add_behavior({"cbf": controller6})
    
    robot7 = Robot(id=7,
                   pose=np.array([480, 220]),
                   velocity=np.array([0, 1.0]),
                   local_perception=100,
                   target_pose=np.array([320, 380]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller7 = DecentralisedCBF(target_pos=np.array([320, 380]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot7.add_behavior({"cbf": controller7})

    env = Environment(render=True,
                      config={})
    
    robot8 = Robot(id=8,
                   pose=np.array([320, 380]),
                   velocity=np.array([0, 1.0]),
                   local_perception=100,
                   target_pose=np.array([320, 220]),
                   mass=1.0,
                   min_v=0.0,
                   max_v=max_v,
                   max_a=1.0)

    controller8 = DecentralisedCBF(target_pos=np.array([480, 220]),
                                  controller_gain=np.array([0.1, 0, 0]))
    robot8.add_behavior({"cbf": controller8})
    
    # Add a single animal

    env = Environment(render=True,
                      config={})

    env.add_entity(robot1)
    env.add_entity(robot2)
    env.add_entity(robot3)
    env.add_entity(robot4)
    env.add_entity(robot5)
    env.add_entity(robot6)
    env.add_entity(robot7)
    env.add_entity(robot8)
    
    config = 'default_config.yml'
    # Read yaml and extract configuration
    with open(f'config/{config}', 'r') as file:
        config = yaml.safe_load(file)

    herd_config = config['entity']['herd']
    herds = Spawner.auto_spawn_herds(herd_config)
    for herd in herds:
        env.add_entity(herd)

    # Behavior related configuration
    math_flock_config = config['behavior']['math_flock']
    math_flock = MathematicalFlock(**math_flock_config['params'])
    for herd in herds:
        math_flock.add_herd(herd)
    
    math_flock.add_shepherd(robot1)
    
    env.add_behaviour(math_flock)

    start_t = 0
    max_t = 3000
    while env.ok and start_t <= max_t:
        env.run_once()
        env.render()
        start_t += 1
    
    
    # env.save_data()



# def single_iteration():
#     config = 'default_config.yml'
#     # Read yaml and extract configuration
#     with open(f'config/{config}', 'r') as file:
#         config = yaml.safe_load(file)

#     entities = []
#     # Component Config
#     entity_config = config['entity']
#     behavior_config = config['behavior']

#     # Create herds
#     herd_config = entity_config['herd']
#     herds = Spawner.auto_spawn_herds(herd_config)
#     entities = entities + herds
#     # Create obstacles
#     obstacle_config = entity_config['obstacle']
#     obstacles = Spawner.auto_spawn_obstacles(obstacle_config)
#     entities = entities + obstacles
#     # Create shepherds
#     shepherd_config = entity_config['shepherd']
#     shepherds = Spawner.auto_spawn_shepherds(shepherd_config)

#     # Shepherd behavior
#     surround_config = behavior_config['surround']
#     shepherd: Shepherd
#     for shepherd in shepherds:
#         shepherd_behaviors = {}
#         dec_surround = DecentralisedSurrounding(**surround_config['params'])
#         shepherd_behaviors["dec_surround"] = dec_surround
#         shepherd.add_behavior(shepherd_behaviors)

#     # Update entities list
#     entities = entities + shepherds

#     # Behavior related configuration
#     math_flock_config = behavior_config['math_flock']
#     math_flock = MathematicalFlock(**math_flock_config['params'])

#     # Add herds
#     for herd in herds:
#         math_flock.add_herd(herd)

#     # Add obstacles
#     for obstacle in obstacles:
#         math_flock.add_obstacle(obstacle)

#     # # Add shepherd
#     for shepherd in shepherds:
#         math_flock.add_shepherd(shepherd)

#     # Environment
#     env_config = config['sim']
#     max_t = env_config['max_t']
#     collect_data = env_config['collect_data']
#     env = Environment(render=env_config['render'],
#                       config=config)
#     for entity in entities:
#         env.add_entity(entity)

#     start_t = 0
#     env.add_behaviour(math_flock)
#     while env.ok:
#         env.run_once()
#         env.render()
#         if env.quit():
#             env.save_data()
#         if collect_data and start_t > max_t:
#             env.save_data()
#             break
#         start_t += 1


if __name__ == '__main__':
    # single_iteration()
    main()
