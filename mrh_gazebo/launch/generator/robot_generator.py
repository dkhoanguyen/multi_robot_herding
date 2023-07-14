# !/usr/bin/python3

import numpy as np
from launch_file_generator import *


class Robot(object):
    def __init__(self, description_pkg: str,
                 description_file_path: str):
        robot_description_file_path = f"$(find {description_pkg})/{description_file_path}"
        find_command = f"$(find xacro)/xacro {robot_description_file_path}"
        self._robot_description_param = Param(name="robot_description",
                                              command=find_command)

    def _model(self, name: str, pose: np.ndarray):
        x = pose[0]
        y = pose[1]
        z = 0
        yaw = pose[2]
        args = f"-urdf -model {name} -x {x} -y {y} -z {z} -Y {yaw} -param robot_description"
        node = Node(pkg="gazebo_ros",
                    exec="spawn_model",
                    name="spawn_urdf",
                    arguments=args)
        return node

    def _state_publisher(self, name: str):
        node = Node(pkg="robot_state_publisher",
                    exec="robot_state_publisher",
                    name="robot_state_publisher",
                    output="screen")
        pub_freq_param = Param(name="publish_frequency",
                               type="double",
                               value="50.0")
        tf_prefix_param = Param(name="tf_prefix",
                                value=name)
        node.add(pub_freq_param)
        node.add(tf_prefix_param)
        return node

    def _controller(self, pkg: str, exec: str, robot_name: str):
        node = Node(pkg="mrh_gazebo",
                    exec="controller",
                    name="pp_controller")
        remapped_odom = Remap(original="/odom", new=f"/{robot_name}/odom")
        remapped_cmd_vel = Remap(
            original="/cmd_vel", new=f"/{robot_name}/cmd_vel")
        remapped_path = Remap(
            original="/path", new=f"/{robot_name}/path")
        node.add(remapped_odom)
        node.add(remapped_cmd_vel)
        node.add(remapped_path)
        return node
    
    def _behavior(self, pkg: str, exec: str, robot_name:str):
        node = Node(pkg="mrh_robot",
                    exec="mrh_robot",
                    name="robot_behavior")
        remapped_odom = Remap(original="/odom", new=f"/{robot_name}/odom")
        remapped_path = Remap(
            original="/path", new=f"/{robot_name}/path")
        node.add(remapped_odom)
        node.add(remapped_path)
        return node

    def spawn_robot(self, name: str, pose: np.ndarray):
        ns_group = Group(namespace=name)
        ns_group.add(self._robot_description_param)
        ns_group.add(self._state_publisher(name=name))
        ns_group.add(self._model(name=name, pose=pose))
        ns_group.add(self._controller(pkg="mrh_gazebo",
                     exec="controller", robot_name=name))
        ns_group.add(self._behavior(pkg="mrh_robot",
                     exec="mrh_robot", robot_name=name))
        return ns_group
