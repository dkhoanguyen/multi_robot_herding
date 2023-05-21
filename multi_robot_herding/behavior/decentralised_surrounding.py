# !/usr/bin/python3
import math
import pygame
import numpy as np
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedSurrounding(DecentralisedBehavior):
    def __init__(self, cs: float = 100.0,
                 co: float = 75.0,
                 edge_k: float = 0.375,
                 distance_to_target: float = 200.0,
                 interagent_spacing: float = 200.0):
        super().__init__()
        self._cs = cs
        self._co = co
        self._edge_k = edge_k
        self._distance_to_target = distance_to_target
        self._interagent_spacing = interagent_spacing

        self._pose = np.zeros(2)
        self._force = np.zeros(2)

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray):
        # Control signal
        self._pose = state[:2]
        u = np.zeros((1, 2))
        all_shepherd_states = np.vstack((state, other_states))

        herd_density = self._herd_density(herd_states=herd_states)
        filtered_herd_density_idx = np.where(
            np.linalg.norm(herd_density, axis=1) >= 0.05)
        filter_herd_states = herd_states[filtered_herd_density_idx[0], :]

        delta_adjacency_vector = self._get_delta_adjacency_vector(
            filter_herd_states,
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
            sj = filter_herd_states[neighbor_herd_idxs, :2]

            ps = self._edge_following(
                si=di, sj=sj, k=self._edge_k,
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
        pygame.draw.line(
            screen, pygame.Color("white"),
            tuple(self._pose), tuple(self._pose + 75 * self._force))
        return super().display(screen)

    # Inter-robot Interaction Control
    def _collision_avoidance_term(self, gain: float, qi: np.ndarray,
                                  qj: np.ndarray, r: float):
        # n_ij = utils.MathUtils.sigma_norm_grad(qj - qi)
        # return gain * np.sum(utils.MathUtils.phi_alpha(
        #     utils.MathUtils.sigma_norm(qj-qi),
        #     r=r,
        #     d=r)*n_ij, axis=0)
        w = self._agent_spacing_term(qi=qi, qj=qj, r_star=r, delta_r=0.1*r)
        return gain * utils.unit_vector(w)

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
            w_sum += 1/(np.linalg.norm(sij) - r) * w
        return w_sum

    def _edge_following(self, si: np.ndarray, sj: np.ndarray,
                        k: float, stabilised_range: float,
                        encircle_gain: float):
        # TODO: might need to rework this potential function
        local_crowd_horizon = self._local_crowd_horizon(
            si=si, sj=sj, k=k, r=stabilised_range)
        return -encircle_gain * utils.unit_vector(local_crowd_horizon)

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

    def _herd_density(self, herd_states: np.ndarray):
        herd_densities = np.zeros((herd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states,
                                                                  r=40)
        for idx in range(herd_states.shape[0]):
            # Density
            neighbor_idxs = alpha_adjacency_matrix[idx]
            density = self._calc_density(
                idx=idx, neighbors_idxs=neighbor_idxs,
                herd_states=herd_states)
            herd_densities[idx] = density
        return herd_densities

    def _agent_spacing_term(self, qi: np.ndarray,
                            qj: np.ndarray, r_star: float, delta_r: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            qij = qj[i, :] - qi
            r = np.linalg.norm(qij)
            e = 0.01
            alpha = (1/delta_r) * math.log((1 - e)/e)
            s_in = 1 - (1/1 + math.exp(alpha * (r - (r_star - delta_r))))
            s_out = 1 - (1/1 + math.exp(-alpha * (r - (r_star + delta_r))))
            w = (s_in + s_out) * utils.unit_vector(qij)
            w_sum += 10. * w
        return w_sum
