# !/usr/bin/python3

import pygame
import numpy as np
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedSurrounding(DecentralisedBehavior):
    def __init__(self, cs: float = 100.0,
                 co: float = 2.78,
                 edge_k: float = 0.1075,
                 distance_to_target: float = 200.0,
                 interagent_spacing: float = 200.0):
        super().__init__()
        self._cs = cs
        self._co = co
        self._edge_k = edge_k
        self._distance_to_target = distance_to_target
        self._interagent_spacing = interagent_spacing

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray):
        # Control signal
        u = np.zeros((1, 2))
        all_shepherd_states = np.vstack((state, other_states))

        delta_adjacency_vector = self._get_delta_adjacency_vector(
            herd_states,
            state,
            r=self._distance_to_target)

        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            all_shepherd_states,
            r=self._interagent_spacing)

        total_ps = 0
        di = state[:2]

        neighbor_herd_idxs = delta_adjacency_vector
        ps = np.zeros(2)
        if sum(neighbor_herd_idxs) > 0:
            sj = herd_states[neighbor_herd_idxs, :2]
            closest_herd = np.argmin(np.linalg.norm(di - sj, axis=1))

            ps = self._edge_following(
                si=di, sj=sj[closest_herd, :2].reshape((1, 2)), k=self._edge_k,
                stabilised_range=self._distance_to_target,
                encircle_gain=self._cs)

            total_ps += np.linalg.norm(ps)

        po = np.zeros(2)
        neighbor_shepherd_idxs = alpha_adjacency_matrix[0]
        if sum(neighbor_shepherd_idxs) > 0:
            dj = all_shepherd_states[neighbor_shepherd_idxs, :2]

            po = self._collision_avoidance_term(
                gain=self._co,
                qi=di, qj=dj,
                r=self._interagent_spacing)

        u = ps + po
        if np.linalg.norm(u) > 15:
            u = 15 * utils.unit_vector(u)
        return u

    def display(self, screen: pygame.Surface):
        return super().display(screen)

    # Inter-robot Interaction Control
    def _collision_avoidance_term(self, gain: float, qi: np.ndarray,
                                  qj: np.ndarray, r: float):
        n_ij = utils.MathUtils.sigma_norm_grad(qj - qi)
        return gain * np.sum(utils.MathUtils.phi_alpha(
            utils.MathUtils.sigma_norm(qj-qi),
            r=r,
            d=r)*n_ij, axis=0)

    def _get_alpha_adjacency_matrix(self, agent_states: np.ndarray,
                                    r: float) -> np.ndarray:
        adj_matrix = np.array(
            [np.linalg.norm(agent_states[i, :2]-agent_states[:, :2], axis=-1) <= r
             for i in range(len(agent_states))])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _local_crowd_horizon(self, si: np.ndarray, sj: np.ndarray, k: float, r: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * (np.linalg.norm(sij) - r))) * \
                utils.unit_vector(sij)
            w_sum += w
        return w_sum

    def _edge_following(self, si: np.ndarray, sj: np.ndarray,
                        k: float, stabilised_range: float,
                        encircle_gain: float):
        # TODO: might need to rework this potential function
        local_crowd_horizon = self._local_crowd_horizon(
            si=si, sj=sj, k=k, r=stabilised_range)
        return -encircle_gain * \
            (1/np.linalg.norm(local_crowd_horizon)) * \
            utils.unit_vector(local_crowd_horizon)

    def _get_delta_adjacency_vector(self, herd_states: np.ndarray,
                                    shepherd_state: np.ndarray, r: float) -> np.ndarray:
        adj_vector = []
        for i in range(herd_states.shape[0]):
            adj_vector.append(np.linalg.norm(
                shepherd_state[:2] - herd_states[i, :2]) <= r)
        return np.array(adj_vector, dtype=np.bool8)

    def _get_alpha_adjacency_matrix(self, agent_states: np.ndarray,
                                    r: float) -> np.ndarray:
        adj_matrix = np.array(
            [np.linalg.norm(agent_states[i, :2]-agent_states[:, :2], axis=-1) <= r
             for i in range(agent_states.shape[0])])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix