# !/usr/bin/python3

import math
import pygame
import numpy as np
from spatialmath.base import *
from collections import deque
from enum import Enum

from multi_robot_herding.utils import utils
from multi_robot_herding.entity.entity import Autonomous, DynamicType
from multi_robot_herding.entity.obstacle import Obstacle
from multi_robot_herding.behavior.controller import SimplePController


class Robot(Autonomous):
    def __init__(self,
                 id: int,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 local_perception: float,
                 target_pose: np.ndarray,
                 mass: float,
                 min_v: float,
                 max_v: float,
                 max_a: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='leader-boid.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)
        self._target_pose = target_pose

        self._max_a = max_a
        self._id = id
        self._local_perception = local_perception
        self._type = DynamicType.DoubleIntegrator

        self._controller = SimplePController(p_gain=1.0)

        self._dt = 0.1

        self._sensing_range = 1000.0

        self._font = pygame.font.SysFont("comicsans", 16)
        self._text = None

    def __str__(self):
        return "robot"

    def update(self, *args, **kwargs):
        # Behavior tree should be here
        events = kwargs["events"]
        ids = kwargs["ids"]

        all_states = kwargs["entity_states"]
        all_animal_states = all_states["herd"]
        all_robot_states = all_states["robot"]

        # Check which robot is within vision
        robot_in_range = np.empty((0, 6))
        animal_in_range = np.empty((0,6))
        total_vel_norm = np.linalg.norm(self.state[2:4])

        for idx in range(all_robot_states.shape[0]):
            d = np.linalg.norm(self.state[:2] - all_robot_states[idx, :2])
            if d > 0.0 and d <= self._sensing_range:
                robot_in_range = np.vstack(
                    (robot_in_range, all_robot_states[idx, :]))
        for idx in range(all_animal_states.shape[0]):
            d = np.linalg.norm(self.state[:2] - all_animal_states[idx, :2])
            if d > 0.0 and d <= self._sensing_range:
                animal_in_range = np.vstack(
                    (animal_in_range, all_animal_states[idx, :]))
        
        self._behavior_state = "cbf"
        u = self._behaviors[str(self._behavior_state)].update(
                state=self.state,
                other_states=robot_in_range,
                animals_states=animal_in_range)
        
        # Double integrator update
        self.pose = self.pose + self.velocity * self._dt + 0.5 * u * self._dt ** 2

        self.velocity = self.velocity + u * self._dt
        
        if np.linalg.norm(self.velocity) > self._max_v:
            self.velocity = self._max_v * utils.unit_vector(self.velocity)
        self._rotate_image(self.velocity)
        self._text = self._font.render(str(self._id), 1, pygame.Color("white"))

    def display(self, screen: pygame.Surface, debug=False):
        if self._behaviors[str(self._behavior_state)]:
            self._behaviors[str(self._behavior_state)].display(screen)

        if self._text:
            screen.blit(self._text, tuple(self.pose - np.array([20, 20])))

        # if self._plot_influence:
        #     pygame.draw.circle(screen, pygame.Color("white"),
        #                        tuple(self.pose), 200, 1)

        return super().display(screen, debug)