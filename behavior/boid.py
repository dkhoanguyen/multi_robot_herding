# !/usr/bin/python3

import math
import numpy as np

from environment.environment import Environment
from kinematic.kinematic_model import DifferentialDrive
from animation.animation_handle import DDAnimationHandle


class Boid(object):
    def __init__(self,
                 pose: np.ndarray,
                 initial_velocity: np.ndarray,
                 map_width: int,
                 map_height: int):
        self._pose = pose

        self._max_v = 50
        self._max_w = 100000

        # Robot
        wheel_base_length = 10
        wheel_radius = 5

        self._model = DifferentialDrive(wheel_base_length, wheel_radius)
        self._animation = DDAnimationHandle(pose)

        self._initial_velocity = initial_velocity
        self._map_width = map_width
        self._map_height = map_height

        self._perception = 150

    @property
    def model(self):
        return self._model

    @property
    def animation(self):
        return self._animation

    @property
    def pose(self):
        return self._pose

    def alignment(self, flock: list):
        avg_heading = 0
        i = 0
        boid: Boid
        for boid in flock:
            if np.linalg.norm(boid.pose[0:2] - self._pose[0:2]) <= self._perception:
                avg_heading += boid.pose[2]
                i += 1

        avg_heading = avg_heading / i
        steering = avg_heading - self._pose[2]
        return steering

    def cohesion(self, flock: list):
        steering = 0
        total = 0
        center_of_mass = np.zeros(2)
        for boid in flock:
            if np.linalg.norm(boid.pose[0:2] - self._pose[0:2]) < self._perception:
                center_of_mass += boid.pose[0:2]
                total += 1
  
        if total > 0:
            center_of_mass /= total
            vec_to_com = center_of_mass - self._pose[0:2]
            steering = math.atan2(vec_to_com[1], vec_to_com[0]) - self._pose[2]

        return steering

    def separation(self, flock: list):
        steering = 0
        total = 0
        avg_vector = np.zeros(2)

        boid: Boid
        for boid in flock:
            distance = np.linalg.norm(boid.pose[0:2] - self._pose[0:2])
            if not np.array_equal(self._pose[0:2],boid.pose[0:2]) \
               and distance < 50:
                diff = self._pose[0:2] - boid.pose[0:2]
                # This is to get the unit vector
                diff /= distance
                avg_vector += diff
                total += 1

        if total > 0:
            avg_vector /= total
            steering = math.atan2(avg_vector[1], avg_vector[0]) - self._pose[2]
        return steering

    def apply_behavior(self, flock: list, sample_time: float):
        # Potential inefficiency in code if we add more behaviors in the future
        cohesion = self.cohesion(flock)
        separation = self.separation(flock)
        alignment = self.alignment(flock)

        avg_steering = separation + alignment + 0.3 * cohesion

        if avg_steering > self._max_w:
            avg_steering = (
                avg_steering/np.linalg.norm(avg_steering)) * self._max_w

        w_v = self._model.inverse_kinematic(
            np.array([self._max_v, 0, avg_steering]))
        v = self._model.forward_kinematic(w_v)

        vel = Environment.body_to_world(v, self._pose)
        poses = self._pose + vel * sample_time

        if poses[0] >= self._map_width:
            poses[0] = 0
        elif poses[0] < 0:
            poses[0] = self._map_width

        if poses[1] >= self._map_height:
            poses[1] = 0
        elif poses[1] < 0:
            poses[1] = self._map_height
        
        self._pose = poses
        self._animation.set_pose(poses)
        self._animation.update(sample_time)
