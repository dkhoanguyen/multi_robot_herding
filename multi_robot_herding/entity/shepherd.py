# !/usr/bin/python3

import math
import pygame
import numpy as np
from spatialmath.base import *
from collections import deque
from enum import Enum

from multi_robot_herding.entity.entity import Autonomous, Entity, DynamicType
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior


class State(Enum):
    IDLE = "dec_idle"
    APPROACH = "dec_approach"
    SURROUND = "dec_surround"
    FORMATION = "dec_formation"

    def __str__(self):
        return f"{self.value}"


class Shepherd(Autonomous):
    def __init__(self,
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
            image_path='leader-boid.png',
            mass=mass,
            min_v=min_v,
            max_v=max_v)

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

    def __str__(self):
        return "shepherd"
    
    @property
    def consensus_state(self):
        return self._consensus_state

    def update(self, *args, **kwargs):
        # Behavior tree should be here
        events = kwargs["events"]
        all_states = kwargs["entity_states"]
        all_herd_states = all_states["herd"]
        all_shepherd_states = all_states["shepherd"]

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
        self._total_velocity_norm.append(total_vel_norm)

        # Update consensus
        self._consensus_state["formation_stable"] = self._formation_stable()

        formation_stable_consensus = True
        for consensus in all_consensus_states:
            if not consensus["formation_stable"]:
                formation_stable_consensus = False
                break
        # if formation_stable_consensus:
        #     print(formation_stable_consensus)
        
        if self._behavior_state == State.IDLE:
            self._behavior_state = State.APPROACH
        elif self._behavior_state == State.APPROACH and self._surround_herd(all_herd_states):
            self._behavior_state = State.SURROUND
        elif self._behavior_state == State.SURROUND and not self._surround_herd(all_herd_states):
            self._behavior_state = State.APPROACH

        u = np.zeros(2)
        if formation_stable_consensus:
            self._behaviors["dec_surround"].set_distance_to_target(80)

        if self._behaviors[str(self._behavior_state)]:
            u = self._behaviors[str(self._behavior_state)].update(
                state=self.state,
                other_states=shepherd_in_range,
                herd_states=all_herd_states)

        if self._type == DynamicType.SingleIntegrator:
            qdot = u.reshape((u.size,))
            self.velocity = qdot
            self.pose = self.pose + self.velocity * 0.2

        self._rotate_image(self.velocity)

    def display(self, screen: pygame.Surface, debug=False):
        if self._behaviors[str(self._behavior_state)]:
            self._behaviors[str(self._behavior_state)].display(screen)
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

    # Immitate how herds should be moving away from shepherds
    def induce_consesus_point(self, r=400):
        angle = -self._heading
        consensus_point = transl2(
            self._pose) @ trot2(angle) @ transl2(np.array([r, 0]))

        # Save this value for visualisation purpose only
        self._consensus_point = consensus_point[0:2, 2]
        return self._consensus_point

    def _surround_herd(self, herd_states):
        herd_mean = np.sum(
            herd_states[:, :2], axis=0) / herd_states.shape[0]

        d_to_herd_mean = np.linalg.norm(
            herd_states[:, :2] - herd_mean, axis=1)
        herd_radius = np.max(d_to_herd_mean)

        if np.linalg.norm(self.pose - herd_mean) <= (self._sensing_range + herd_radius):
            return True
        
        # for i in range(herd_states.shape[0]):
        #     d = np.linalg.norm(self.pose - herd_states[i,:2])
        #     if d <= self._sensing_range:
        #         return True
        return False

    def _formation_stable(self):
        if len(self._total_velocity_norm) != self._time_horizon:
            return False
        y = np.array(self._total_velocity_norm)
        x = np.linspace(0, self._time_horizon,
                        self._time_horizon, endpoint=False)
        coeff, err, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        if math.sqrt(err) >= 20:
            return False
        poly = np.poly1d(coeff)
        polyder = np.polyder(poly)
        cond = np.abs(np.round(float(polyder.coef[0]), 2))
        return not bool(cond)