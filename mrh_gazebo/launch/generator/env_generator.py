# !/usr/bin/python3

from launch_file_generator import *


class GazeboWorld(object):
    def __init__(self, launch_pkg: str,
                 launch_file_path: str,
                 world_pkg: str,
                 world_description_file: str):
        launch_file_path_str = f"$(find {launch_pkg})/{launch_file_path}"
        world_description_path_str = f"$(find {world_pkg})/{world_description_file}"
        self._gazebo_world_include = Include(file=launch_file_path_str)
        world_name_arg = Arg(name="world_name",
                             value=world_description_path_str)
        paused_arg = Arg(name="paused",
                         value="false")
        use_sim_time_arg = Arg(name="use_sim_time", value="true")
        gui_arg = Arg(name="gui", value="true")
        headless_arg = Arg(name="headless", value="false")
        debug_arg = Arg(name="debug", value="false")

        self._gazebo_world_include.add(world_name_arg)
        self._gazebo_world_include.add(paused_arg)
        self._gazebo_world_include.add(use_sim_time_arg)
        self._gazebo_world_include.add(gui_arg)
        self._gazebo_world_include.add(headless_arg)
        self._gazebo_world_include.add(debug_arg)

    def spawn_world(self):
        return self._gazebo_world_include
