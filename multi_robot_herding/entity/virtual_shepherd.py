# !/usr/bin/python3

import math
import pygame
import numpy as np
from spatialmath.base import *
from collections import deque
from enum import Enum

from multi_robot_herding.utils import utils
from multi_robot_herding.entity.entity import Autonomous, Entity, DynamicType
from multi_robot_herding.entity.obstacle import Obstacle
from multi_robot_herding.entity.shepherd import State

class VirtualShepherd(Autonomous):
    def __init__(self,
                 id: int,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 local_perception: float,
                 local_boundary: float,
                 mass: float,
                 min_v: float,
                 max_v: float):
        super().__init__(
            pose=pose,
            velocity=velocity,
            image_path='robot.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)
        
        # ID for leader follower assignment
        self._id = id
        self._local_perception = local_perception
        self._local_boundary = local_boundary

        self._r = 40
        self._consensus_r = 200.0
        self._sensing_range = 200.0

        self._behavior_state = State.IDLE

        # Consensus for state transition
        self._consensus_state = {}
        self._time_horizon = 300
        self._total_velocity_norm = deque(maxlen=self._time_horizon)

        # Display
        self._font = pygame.font.SysFont("comicsans", 16)
        self._text = None

        self._plot_influence = False

    def __str__(self):
        return "virtual"

    @property
    def id(self):
        return self._id

    @property
    def consensus_state(self):
        return self._consensus_state

    def update(self, *args, **kwargs):
        # Behavior tree should be here
        events = kwargs["events"]
        ids = kwargs["ids"]

        all_states = kwargs["entity_states"]
        all_herd_states = all_states["herd"]
        all_shepherd_states = all_states["shepherd"]
        all_virtual_states = all_states["virtual"]

        # Consensus state
        all_consensus_states = kwargs["consensus_states"]

        # Check which shepherd is within vision
        shepherd_in_range = np.empty((0, 6))
        total_vel_norm = np.linalg.norm(self.state[2:4])

        for idx in range(all_shepherd_states.shape[0]):
            d = np.linalg.norm(self.state[:2] - all_shepherd_states[idx, :2])
            if d > 0.0:
                shepherd_in_range = np.vstack(
                    (shepherd_in_range, all_shepherd_states[idx, :]))
                total_vel_norm += np.linalg.norm(all_shepherd_states[idx, 2:4])
        for idx in range(all_virtual_states.shape[0]):
            d = np.linalg.norm(self.state[:2] - all_virtual_states[idx, :2])
            if d > 0.0:
                shepherd_in_range = np.vstack(
                    (shepherd_in_range, all_virtual_states[idx, :]))
                total_vel_norm += np.linalg.norm(all_virtual_states[idx, 2:4])
        self._total_velocity_norm.append(total_vel_norm)

        # if self._behavior_state == State.IDLE:
        #     self._behavior_state = State.SURROUND

        # # elif self._behavior_state == State.SURROUND:
        # #     if str(State.SURROUND) in self._behaviors.keys():
        # #         if self._behaviors[str(State.SURROUND)].transition(
        # #                 state=self.state,
        # #                 other_states=shepherd_in_range,
        # #                 herd_states=all_herd_states,
        # #                 consensus_states=all_consensus_states):
        # #             self._behavior_state = State.SURROUND
        u = np.zeros(2)

        # if self._behaviors[str(self._behavior_state)]:
        #     u = self._behaviors[str(self._behavior_state)].update(
        #         state=self.state,
        #         other_states=shepherd_in_range,
        #         herd_states=all_herd_states,
        #         obstacles=self._static_obstacles,
        #         consensus_states=all_consensus_states,
        #         output_consensus_state=self._consensus_state)

        if self._type == DynamicType.SingleIntegrator:
            if np.linalg.norm(u) > self._max_v:
                u = self._max_v * utils.unit_vector(u)

            qdot = u.reshape((u.size,))
            self.velocity = qdot
            self.pose = self.pose + self.velocity * 0.2

        self._rotate_image(self.velocity)
        self._text = self._font.render(str(self._id), 1, pygame.Color("white"))

    def display(self, screen: pygame.Surface, debug=False):
        # if self._behaviors[str(self._behavior_state)]:
        #     self._behaviors[str(self._behavior_state)].display(screen)

        if self._text:
            screen.blit(self._text, tuple(self.pose - np.array([20, 20])))

        if self._plot_influence:
            pygame.draw.circle(screen, pygame.Color("white"),
                               tuple(self.pose), 200, 1)

        return super().display(screen, debug)

    def in_entity_radius(self, qi: np.ndarray, r: float) -> bool:
        # Project entity posit
        return np.linalg.norm(self._pose - qi) <= (r + self._r)

    def induce_delta_agent(self, alpha_agent: Entity):
        qi = alpha_agent.pose.reshape((2, 1))
        pi = alpha_agent.velocity.reshape((2, 1))
        yk = self._pose.reshape((2, 1))
        d = np.linalg.norm(qi - yk)

        if d < self._r:
            mu = d / np.linalg.norm(qi - yk)
            ak = (qi - yk)/np.linalg.norm(qi - yk)
            P = np.eye(2) - ak @ ak.transpose()

            qik = mu * qi + (1 - mu) * yk
            pik = mu * P @ pi
            return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)
        else:
            mu = self._r / np.linalg.norm(qi - yk)
            ak = (qi - yk)/np.linalg.norm(qi - yk)
            P = np.eye(2) - ak @ ak.transpose()

            qik = mu * qi + (1 - mu) * yk
            pik = mu * P @ pi
            return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)