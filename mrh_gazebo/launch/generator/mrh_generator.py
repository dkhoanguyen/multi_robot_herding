# !/usr/bin/python3

import os
import yaml
from launch_file_generator import *
from env_generator import *
from robot_generator import *


class MrhLaunchGenerator(object):
    def __init__(self):
        abs_path = os.path.dirname(os.path.abspath(__file__))
        launch_file_dir = abs_path.replace("/generator", "")
        config = 'default.yaml'
        # Read yaml and extract configuration
        with open(f'{abs_path}/config/{config}', 'r') as file:
            config = yaml.safe_load(file)

        entities = []
        # Entity related configuration
        entity_config = config['entity']
        # Create herds
        herd_config = entity_config['herd']
        num = herd_config.pop('num')
        x = np.random.randint(
            -1.5, 1.5, (num, 1)).astype('float')
        y = np.random.randint(
            -1.5, 1.5, (num, 1)).astype('float')
        initial_poses = np.hstack((x, y))

        self._launch_description = LaunchDescription(path=launch_file_dir,
                                                     name="mrh_gazebo")
        world_spawner = GazeboWorld(launch_pkg="gazebo_ros",
                                    launch_file_path="launch/empty_world.launch",
                                    world_pkg="turtlebot3_gazebo",
                                    world_description_file="worlds/empty.world")
        herd_spawner = Robot(description_pkg="turtlebot3_description",
                             description_file_path="urdf/turtlebot3_burger.urdf.xacro")

        # Spawn herds
        herds = []
        for i in range(num):
            yaw = np.pi * (2 * np.random.rand() - 1)
            pose = np.array([initial_poses[i, 0], initial_poses[i, 1], yaw])
            name = f"herd_{i}"
            herds.append(herd_spawner.spawn_robot(name=name, pose=pose))

        for herd in herds:
            self._launch_description.add(herd)

        world = world_spawner.spawn_world()
        self._launch_description.add(world)
        self._launch_description.write()

test = MrhLaunchGenerator()
