# !/usr/bin/python3

import math
import pygame
import numpy as np
from spatialmath.base import *
from collections import deque
from enum import Enum

from multi_robot_herding.utils import utils
from multi_robot_herding.entity.entity import Autonomous, Entity, DynamicType


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

        if self._behavior_state == State.IDLE:
            self._behavior_state = State.APPROACH

        elif self._behavior_state == State.APPROACH:
            if self._behaviors[str(State.SURROUND)]:
                if self._behaviors[str(State.SURROUND)].transition(
                        state=self.state,
                        other_states=shepherd_in_range,
                        herd_states=all_herd_states,
                        consensus_states=all_consensus_states):
                    self._behavior_state = State.SURROUND

        elif self._behavior_state == State.SURROUND:
            if self._behaviors[str(State.APPROACH)]:
                if self._behaviors[str(State.APPROACH)].transition(
                        state=self.state,
                        other_states=shepherd_in_range,
                        herd_states=all_herd_states,
                        consensus_states=all_consensus_states):
                    self._behavior_state = State.APPROACH

        u = np.zeros(2)

        if self._behaviors[str(self._behavior_state)]:
            u = self._behaviors[str(self._behavior_state)].update(
                state=self.state,
                other_states=shepherd_in_range,
                herd_states=all_herd_states,
                consensus_states=all_consensus_states)

        if self._type == DynamicType.SingleIntegrator:
            if np.linalg.norm(u) > 20:
                u = 20 * utils.unit_vector(u)

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

    def _updated_formation_stable(self, herd_states: np.ndarray,
                                  shepherd_states: np.ndarray):
        herd_density = self._herd_density(herd_states=herd_states,
                                          shepherd_states=shepherd_states)

    def _get_delta_adjacency_vector(self, herd_state: np.ndarray,
                                    shepherd_states: np.ndarray, r: float) -> np.ndarray:
        adj_vector = []
        for i in range(shepherd_states.shape[0]):
            adj_vector.append(np.linalg.norm(
                herd_state[:2] - shepherd_states[i, :2]) <= r)
        return np.array(adj_vector, dtype=np.bool8)

    def _get_alpha_adjacency_matrix(self, agent_states: np.ndarray,
                                    r: float) -> np.ndarray:
        adj_matrix = np.array(
            [np.linalg.norm(agent_states[i, :2]-agent_states[:, :2], axis=-1) <= r
             for i in range(agent_states.shape[0])])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _density(self, si: np.ndarray, sj: np.ndarray, k: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * np.linalg.norm(sij))) * \
                utils.unit_vector(sij)
            w_sum += w
        return w_sum

    def _calc_density(self, idx: int,
                      neighbors_idxs: np.ndarray,
                      herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        density = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = herd_states[neighbors_idxs, :2]
            density = self._density(si=qi, sj=qj, k=0.375)
        return density

    def _herd_density(self, herd_states: np.ndarray,
                      shepherd_states: np.ndarray):
        herd_densities = np.zeros((herd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states,
                                                                  r=40)
        for idx in range(herd_states.shape[0]):
            # Herd internal density
            neighbor_idxs = alpha_adjacency_matrix[idx]
            density = self._calc_density(
                idx=idx, neighbors_idxs=neighbor_idxs,
                herd_states=herd_states)
            herd_densities[idx] += density

            # Herd shepherd density
            delta_adj_vec = self._get_delta_adjacency_vector(
                herd_state=herd_states[idx, :2],
                shepherd_states=shepherd_states,
                r=110)
            qi = herd_states[idx, :2]
            qj = shepherd_states[delta_adj_vec, :2]
            density = self._density(si=qi, sj=qj, k=0.375)
            herd_densities[idx] += density
        return herd_densities
