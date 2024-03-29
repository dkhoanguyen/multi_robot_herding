#!/usr/bin/python3

import numpy as np

from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.shepherd import Shepherd
from multi_robot_herding.entity.obstacle import Hyperplane, Sphere


class Spawner(object):
    def __init__(self):
        pass

    @staticmethod
    def manual_spawn(entity, config):
        return entity(**config)

    @staticmethod
    # We might need to make this reuseable in the future
    def auto_spawn_herds(herd_config):
        herds = []
        if herd_config.pop('init_random'):
            num = herd_config.pop('num')
            x = np.random.randint(
                700, 900, (num,1)).astype('float')
            y = np.random.randint(
                300, 500, (num,1)).astype('float')
            initial_poses = np.hstack((x, y))
            
            for i in range(num):
                angle = np.pi * (2 * np.random.rand() - 1)
                vel = herd_config['max_v'] * \
                    np.array([np.cos(angle), np.sin(angle)])

                herd_config['pose'] = initial_poses[i, :]
                herd_config['velocity'] = vel
                herds.append(Spawner.manual_spawn(
                    entity=Herd, config=herd_config))
        else:
            pass
        return herds

    @staticmethod
    def auto_spawn_shepherds(shepherd_config):
        shepherds = []

        if shepherd_config.pop('init_random'):
            shepherd_config.pop('configuration')
        else:
            shepherd_config.pop('num')
            initial_configurations = shepherd_config.pop('configuration')
            for idx, configuration in enumerate(initial_configurations):
                shepherd_config.update(configuration)
                shepherd_config['pose'] = np.array(shepherd_config['pose'])
                angle = shepherd_config.pop('angle')
                shepherd_config['velocity'] = shepherd_config['max_v'] * \
                    np.array([np.cos(angle), np.sin(angle)])
                shepherd_config['id'] = idx
                shepherds.append(Spawner.manual_spawn(
                    entity=Shepherd, config=shepherd_config))
        return shepherds

    @staticmethod
    def auto_spawn_obstacles(obs_config):
        obstacles = []
        # Spherial obstacles
        sphere_config = obs_config['sphere']
        if sphere_config:
            for sphere in sphere_config:
                obstacles.append(Spawner.manual_spawn(
                    entity=Sphere, config=sphere))

        # Hyperplane obstacles
        hyperplane_config = obs_config['hyperplane']
        if hyperplane_config:
            for hyperplane in hyperplane_config:
                obstacles.append(Spawner.manual_spawn(
                    entity=Hyperplane, config=hyperplane))
        return obstacles

    @staticmethod
    def auto_spawn_behaviors(behav_config):
        behaviors = []
