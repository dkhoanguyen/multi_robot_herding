# !/usr/bin/python3

import os
from enum import Enum
import xml.etree.ElementTree as ET


class Tag(Enum):
    LAUNCH = "launch"
    NODE = "node"
    MACHINE = "machine"
    INCLUDE = "include"
    REMAP = "remap"
    ENV = "env"
    PARAM = "param"
    ROSPARAM = "rosparam"
    GROUP = "group"
    TEST = "test"
    ARG = "arg"

    def __str__(self):
        return str(self.value)


class Element(ET.Element):
    def __init__(self, tag: Tag):
        super().__init__(str(tag))

    def add(self, element: ET.Element):
        self.append(element)


class Param(Element):
    def __init__(self, name: str = None,
                 value: str = None,
                 type: str = None,
                 command: str = None):
        super().__init__(tag=Tag.PARAM)
        self.set("name", name)
        if type is not None:
            self.set("type", type)
        if value is not None:
            self.set("value", value)
        else:
            self.set("command", command)


class Arg(Element):
    def __init__(self, name: str = None,
                 default: str = None,
                 value: str = None):
        super().__init__(tag=Tag.ARG)
        self.set("name", name)
        if default is None:
            self.set("value", value)
        else:
            self.set("default", default)


class Include(Element):
    def __init__(self, file: str = None,
                 namespace: str = None):
        super().__init__(tag=Tag.INCLUDE)
        self.set("file", file)
        if namespace is not None:
            self.set("ns", namespace)


class Node(Element):
    def __init__(self, pkg: str = None,
                 exec: str = None,
                 name: str = None,
                 namespace: str = None,
                 output: str = None,
                 arguments: str = None,
                 params: Param = None):
        super().__init__(tag=Tag.NODE)
        self.set("pkg", pkg)
        self.set("type", exec)
        self.set("name", name)
        if namespace is not None:
            self.set("ns", namespace)
        if output is not None:
            self.set("output", output)
        if arguments is not None:
            self.set("args", arguments)
        if params is not None:
            self.append(params)


class LaunchDescription(object):
    def __init__(self, path=None,
                 name="test"):
        if path is None:
            self._path = os.getcwd()
        self._name = name
        self._launch = ET.Element("launch")

    def add(self, element):
        self._launch.append(element)

    def write(self):
        # Create the XML tree
        tree = ET.ElementTree(self._launch)
        tree.write(f"{self._name}.launch")


if __name__ == "__main__":
    param = Param(name="test",
                  value="test",
                  type="test",
                  command="")
    model = Arg(name="model",
                default="burger")
    x_pos = Arg(name="x_pos",
                default="-0.2")
    y_pos = Arg(name="y_pos",
                default="-0.5")
    z_pos = Arg(name="z_pos",
                default="0.0")

    empty_world_gazebo = Include(
        file="$(find gazebo_ros)/launch/empty_world.launch")
    empty_world_gazebo.add(Arg(
        name="world_name",
        value="$(find turtlebot3_gazebo)/worlds/turtlebot3_world.world",
    ))
    empty_world_gazebo.add(Arg(
        name="paused",
        value="false"
    ))
    empty_world_gazebo.add(Arg(
        name="use_sim_time",
        value="true",
    ))
    empty_world_gazebo.add(Arg(
        name="gui",
        value="true",
    ))
    empty_world_gazebo.add(Arg(
        name="headless",
        value="false",
    ))
    empty_world_gazebo.add(Arg(
        name="debug",
        value="false",
    ))

    robot_description = Param(
        name="robot_description",
        command="$(find xacro)/xacro --inorder \
        $(find turtlebot3_description)/urdf/turtlebot3_$(arg model).urdf.xacro"
    )

    node = Node(
        pkg="gazebo_ros",
        exec="spawn_model",
        name="spawn_urdf",
        arguments="-urdf -model turtlebot3_$(arg model) \
        -x $(arg x_pos) -y $(arg y_pos) -z $(arg z_pos) \
        -param robot_description"
    )

    test = LaunchDescription()
    test.add(model)
    test.add(x_pos)
    test.add(y_pos)
    test.add(z_pos)
    test.add(empty_world_gazebo)
    test.add(robot_description)
    test.add(node)
    test.write()
