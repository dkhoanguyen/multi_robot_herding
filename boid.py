# !/usr/bin/python3

import numpy as np

from environment import Environment
from kinematic_model import DifferentialDrive
from animation_handle import DDAnimationHandle


class Boid(object):
    def __init__(self,
                 pose: np.ndarray,
                 initial_velocity: np.ndarray):
        self._pose = pose

        self._max_v = 15
        self._max_w = 100000

        # Robot
        wheel_base_length = 10
        wheel_radius = 5

        self._model = DifferentialDrive(wheel_base_length, wheel_radius)
        self._animation = DDAnimationHandle(pose)

        self._initial_velocity = initial_velocity

        self._perception = 100

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
            if np.linalg.norm(boid.pose[0:1] - self._pose[0:1]) <= self._perception:
                avg_heading += boid.pose[2]
                i += 1

        avg_heading = avg_heading / i
        steering = avg_heading - self._pose[2]
        return steering

    def cohesion(self, flock: list):
        pass

    def separation(self, flock: list):
        steering = np.zeros(2)
        total = 0
        avg_vector = np.zeros(2)

        boid: Boid
        for boid in flock:
            distance = np.linalg.norm(boid.pose[0:1] - self._pose[0:1])
            if self._pose[0:1] != boid.pose[0:1] and distance < self._perception:
                diff = self._pose[0:1] - boid.pose[0:1]
                diff /= distance
                avg_vector += diff
                total += 1

        print(avg_vector)

        # if total > 0:
        #     avg_vector /= total
        #     if np.linalg.norm(steering) > 0:
        #         avg_vector = (avg_vector / np.linalg.norm(steering)) * self.max_speed
        #     steering = avg_vector - self.velocity
        #     if np.linalg.norm(steering)> self.max_force:
        #         steering = (steering /np.linalg.norm(steering)) * self.max_force
        return steering

    def apply_behavior(self, flock: list, sample_time: float):
        # Potential inefficiency in code if we add more behaviors in the future
        alignment = self.alignment(flock)
        separation = self.separation(flock)

        avg_steering = alignment

        if avg_steering > self._max_w:
            avg_steering = (
                avg_steering/np.linalg.norm(avg_steering)) * self._max_w

        w_v = self._model.inverse_kinematic(
            np.array([self._max_v, 0, avg_steering]))
        v = self._model.forward_kinematic(w_v)

        vel = Environment.body_to_world(v, self._pose)
        poses = self._pose + vel * sample_time

        if poses[0] >= 700:
            poses[0] = 0
        elif poses[0] < 0:
            poses[0] = 700

        if poses[1] >= 700:
            poses[1] = 0
        elif poses[1] < 0:
            poses[1] = 700
        
        self._pose = poses
        self._animation.set_pose(poses)
        self._animation.update(sample_time)
