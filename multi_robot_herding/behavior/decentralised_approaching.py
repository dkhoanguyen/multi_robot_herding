# !/usr/bin/python3

import numpy as np
import pygame
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedApproaching(DecentralisedBehavior):
    def __init__(self, co: float = 2.78,
                 interagent_spacing: float = 40.0):
        self._co = co
        self._interagent_spacing = interagent_spacing

        # Const parameters
        self._c1 = 5
        self._c2 = 0.2 * np.sqrt(self._c1)

    def update(self, state: np.ndarray,
                     other_states: np.ndarray,
                     herd_states: np.ndarray):
        u = np.zeros((1, 2))
        all_shepherd_states = np.vstack((state, other_states))

        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            all_shepherd_states,
            r=self._interagent_spacing)

        herd_mean = np.sum(
            herd_states[:, :2], axis=0) / herd_states.shape[0]

        di = state[:2]
        d_dot_i = state[2:4]

        po = np.zeros((1, 2))
        neighbor_shepherd_idxs = alpha_adjacency_matrix[0]
        if sum(neighbor_shepherd_idxs) > 0:
            dj = all_shepherd_states[neighbor_shepherd_idxs, :2]

            po = self._collision_avoidance_term(
                gain=self._co,
                qi=di, qj=dj,
                r=self._interagent_spacing)

        # Move toward herd mean
        p_gamma = self._calc_group_objective_control(
            target=herd_mean,
            c1=self._c1,
            c2=self._c2,
            qi=di, pi=d_dot_i)

        u = po + p_gamma
        return u
    
    def display(self, screen: pygame.Surface):
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
             for i in range(agent_states.shape[0])])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

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
