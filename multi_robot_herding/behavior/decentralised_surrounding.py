# !/usr/bin/python3
import math
import pygame
import numpy as np
from collections import deque

from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedSurrounding(DecentralisedBehavior):
    def __init__(self, cs: float = 10.0,
                 co: float = 30.78,
                 edge_k: float = 0.125,
                 distance_to_target: float = 200.0,
                 interagent_spacing: float = 200.0,
                 skrink_distance: float = 80.0,
                 skrink_spacing: float = 100.0,
                 sensing_range: float = 300.0):
        super().__init__()
        self._cs = cs
        self._co = co
        self._edge_k = edge_k
        self._distance_to_target = distance_to_target
        self._interagent_spacing = interagent_spacing
        self._skrink_distance = skrink_distance
        self._skrink_spacing = skrink_spacing
        self._sensing_range = sensing_range

        self._pose = np.zeros(2)
        self._force = np.zeros(2)

        # Plotting
        self._herd_density_to_plot = np.empty((0, 2))

    def transition(self, state: np.ndarray,
                   other_states: np.ndarray,
                   herd_states: np.ndarray,
                   consensus_states: dict):
        herd_mean = np.sum(
            herd_states[:, :2], axis=0) / herd_states.shape[0]

        d_to_herd_mean = np.linalg.norm(
            herd_states[:, :2] - herd_mean, axis=1)
        herd_radius = np.max(d_to_herd_mean)

        if np.linalg.norm(state[:2] - herd_mean) <= (200 + herd_radius):
            return True
        return False

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray,
               consensus_states: dict):
        # Control signal
        self._pose = state[:2]
        u = np.zeros(2)
        all_shepherd_states = np.vstack((state, other_states))

        formation_stable_consensus = True
        for consensus in consensus_states:
            if not consensus["formation_stable"]:
                formation_stable_consensus = False
                break

        d_to_target = self._distance_to_target
        spacing = self._interagent_spacing
        if formation_stable_consensus:
            d_to_target = self._skrink_distance
            spacing = self._skrink_spacing

        herd_density = self._herd_density(herd_states=herd_states,
                                          shepherd_states=all_shepherd_states,
                                          r_shepherd=d_to_target)
 
        total_density = np.sum(np.linalg.norm(herd_density, axis=1))
        filtered_herd_density_idx = np.where(
            np.linalg.norm(herd_density, axis=1) >= 35)
        filter_herd_states = herd_states[filtered_herd_density_idx[0], :]
        self._herd_density_to_plot = filter_herd_states[:,:2]

        delta_adjacency_vector = self._get_delta_adjacency_vector(
            filter_herd_states,
            state,
            r=self._sensing_range)

        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            all_shepherd_states,
            r=d_to_target)

        total_ps = 0
        di = state[:2]

        neighbor_herd_idxs = delta_adjacency_vector
        ps = np.zeros(2)
        if sum(neighbor_herd_idxs) > 0:
            sj = filter_herd_states[:, :2]

            ps = self._sigmoid_edge_following(
                qi=di,
                qj=sj,
                d=spacing,
                bound=20
            )
            total_ps = np.linalg.norm(ps)
        po = np.zeros(2)
        neighbor_shepherd_idxs = alpha_adjacency_matrix[0]
        if sum(neighbor_shepherd_idxs) > 0:
            dj = all_shepherd_states[neighbor_shepherd_idxs, :2]

            po = self._collision_avoidance_term(
                gain=self._co,
                qi=di, qj=dj,
                r=spacing) * 0.03

        u = ps + po
        self._force = u
        return u

    def display(self, screen: pygame.Surface):
        pygame.draw.line(
            screen, pygame.Color("white"),
            tuple(self._pose), tuple(self._pose + 1 * self._force))

        for i in range(self._herd_density_to_plot.shape[0]):
            pygame.draw.circle(screen, pygame.Color("white"),
                               tuple(self._herd_density_to_plot[i, :2]), 10, 2)
        return super().display(screen)

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
            w = (1/(1 + k * (np.linalg.norm(sij)))) * \
                utils.unit_vector(sij)
            w = sij
            w_sum += w
        return w_sum

    def _calc_density(self, idx: int,
                      neighbors_idxs: np.ndarray,
                      herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        density = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = herd_states[neighbors_idxs, :2]
            density = self._density(si=qi, sj=qj, k=0.175)
        return density

    def _herd_density(self, herd_states: np.ndarray,
                      shepherd_states: np.ndarray,
                      r_shepherd: float):
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

            # # Herd shepherd density
            # delta_adj_vec = self._get_delta_adjacency_vector(
            #     shepherd_state=herd_states[idx, :2],
            #     herd_states=shepherd_states,
            #     r=r_shepherd)

            # qi = herd_states[idx, :2]
            # qj = shepherd_states[delta_adj_vec, :2]
            # density = self._density(si=qi, sj=qj, k=0.375)
            # herd_densities[idx] += density
        return herd_densities

    def _calc_group_objective_control(self, target: np.ndarray,
                                      c1: float, c2: float,
                                      qi: np.ndarray, pi: np.ndarray):
        def calc_group_objective_term(
                c1: float, c2: float,
                pos: np.ndarray, qi: np.ndarray, pi: np.ndarray):
            return -c1 * utils.MathUtils.sigma_1(qi - pos) - c2 * (pi)
        u_gamma = calc_group_objective_term(
            c1=c1,
            c2=c2,
            pos=target,
            qi=qi,
            pi=pi)
        return u_gamma

    def _sigmoid_edge_following(self, qi: np.ndarray, qj: np.ndarray, d: float,
                                bound: float):
        def custom_sigmoid(qi: np.ndarray, qj: np.ndarray, d: float,
                           bound: float):
            k = 0.1
            qij = qi - qj
            rij = np.linalg.norm(qij)
            smoothed_rij_d = rij - d
            sigma = (bound/(1 + np.exp(-smoothed_rij_d - k*d))) - \
                (bound/(1 + np.exp(smoothed_rij_d - k*d)))
            sigma = np.round(sigma, 3)
            return sigma

        def local_crowd_horizon(qi: np.ndarray, qj: np.ndarray,
                                r: float,
                                k: float = 0.375):
            qij = qi - qj
            return (1/(1 + k * np.linalg.norm(qij)))

        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = utils.unit_vector(qj[i, :] - qi)
            gain = local_crowd_horizon(qi=qi, qj=qj[i, :], r=d)
            sigma = custom_sigmoid(qi=qi, qj=qj[i, :], d=d, bound=bound)
            u_sum += gain * sigma * uij * 12
        return u_sum
