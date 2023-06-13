# !/usr/bin/python3

import math
import pygame
import numpy as np
from collections import deque

from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedSurrounding(DecentralisedBehavior):
    def __init__(self, cs: float = 10.0,
                 co: float = 30.78 * 0.075,
                 edge_k: float = 0.125,
                 distance_to_target: float = 125.0,
                 interagent_spacing: float = 150.0,
                 skrink_distance: float = 140.0,
                 skrink_spacing: float = 160.0,
                 sensing_range: float = 700.0):
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
        self._force_ps = np.zeros(2)
        self._force_po = np.zeros(2)
        self._force_u = np.zeros(2)

        self._stopped = False

        # Plotting
        self._herd_density_to_plot = np.empty((0, 2))

        self._time_horizon = 200
        self._total_energy = deque(maxlen=self._time_horizon)
        self._stablised_energy = deque(maxlen=10)

        self._triggered = False
        self._voronoi = None

        self._plot_force = True
        self._plot_range = False

    def transition(self, state: np.ndarray,
                   other_states: np.ndarray,
                   herd_states: np.ndarray,
                   consensus_states: dict):

        for idx in range(herd_states.shape[0]):
            if np.linalg.norm(state[:2] - herd_states[idx, :2]) <= self._sensing_range:
                return True
        return False

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray,
               obstacles: list,
               consensus_states: dict,
               raw_states: np.ndarray):
        # Control signal
        self._pose = state[:2]
        u = np.zeros(2)
        all_shepherd_states = np.vstack((state, other_states))

        d_to_target = self._distance_to_target
        spacing = self._interagent_spacing
        # if formation_stable and not self._triggered:
        #     self._distance_to_target = self._skrink_distance
        #     self._interagent_spacing = self._skrink_spacing
        #     self._triggered = True

        herd_density = self._herd_density(herd_states=herd_states,
                                          shepherd_states=all_shepherd_states,
                                          r_shepherd=d_to_target)
        # density = np.linalg.norm(herd_density, axis=1)
        total_density = np.sum(np.linalg.norm(herd_density, axis=1))
        self._total_energy.append(total_density)

        formation_stable = self._formation_stable()
        # if self._stopped or formation_stable:
        #     self._stopped = True
        #     return u

        filter_herd_states = herd_states[:, :]
        self._herd_density_to_plot = filter_herd_states

        delta_adjacency_vector = self._get_delta_adjacency_vector(
            filter_herd_states,
            state,
            r=self._sensing_range)

        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            all_shepherd_states,
            r=self._sensing_range)

        di = state[:2]
        d_dot_i = state[2:4]

        neighbor_herd_idxs = delta_adjacency_vector
        ps = np.zeros(2)
        if sum(neighbor_herd_idxs) > 0:
            sj = filter_herd_states[neighbor_herd_idxs, :2]
            s_dot_j = filter_herd_states[neighbor_herd_idxs, 2:4]
            ps = self._potential_edge_following(qi=di,
                                                qj=sj,
                                                pi=d_dot_i,
                                                pj=s_dot_j,
                                                d=d_to_target,
                                                d_dead=10,
                                                gain=2)
        self._force_ps = ps

        po = np.zeros(2)
        neighbor_shepherd_idxs = alpha_adjacency_matrix[0]

        if sum(neighbor_shepherd_idxs) > 0:
            dj = all_shepherd_states[neighbor_shepherd_idxs, :2]
            d_dot_j = all_shepherd_states[neighbor_shepherd_idxs, 2:4]
            po = self._collision_avoidance_term(
                gain=0.7,
                qi=di, qj=dj,
                pi=d_dot_i,
                pj=d_dot_j,
                r=spacing)
        self._force_po = po

        # Obstacle avoidance term
        beta_adj_vec = self._get_beta_adjacency_vector(state=state,
                                                       obstacles=obstacles,
                                                       r=self._sensing_range)
        p_avoid = np.zeros((2,))
        if sum(beta_adj_vec) > 0:
            p_avoid = self._obstacle_avoidance(qi=di, pi=d_dot_i,
                                               beta_adj_vec=beta_adj_vec,
                                               obstacle_list=obstacles,
                                               r=30,
                                               gain=1)
        p_target = np.zeros((2,))
        if self._stopped or formation_stable:
            self._stopped = True
            p_target = -(10/total_density)*(di - np.array([1200,350]))

        self._force_u = p_target
        u = 15 * ps + 6 * po + p_avoid + p_target
        return u

    def display(self, screen: pygame.Surface):
        if self._plot_force:
            pygame.draw.line(
                screen, pygame.Color("white"),
                tuple(self._pose), tuple(self._pose + 5 * (self._force_po)))
            pygame.draw.line(
                screen, pygame.Color("yellow"),
                tuple(self._pose), tuple(self._pose + 5 * (self._force_ps)))
            pygame.draw.line(
                screen, pygame.Color("grey"),
                tuple(self._pose), tuple(self._pose + 5 * (self._force_u)))

        if self._plot_range:
            pygame.draw.circle(screen, pygame.Color("white"),
                            tuple(self._pose), self._distance_to_target, 1)
            # pygame.draw.circle(screen, pygame.Color("yellow"),
            #                    tuple(self._pose), self._sensing_range, 1)

        return super().display(screen)

    def _collision_avoidance_term(self, gain: float,
                                  qi: np.ndarray, qj: np.ndarray,
                                  pi: np.ndarray, pj: np.ndarray,
                                  r: float):

        def custom_potential_function(qi: np.ndarray, qj: np.ndarray,
                                      d: float):
            qij = qi - qj
            rij = np.linalg.norm(qij)
            smoothed_rij_d = rij - d
            c = 12
            m = 10

            fx = gain*(1 - np.exp(-smoothed_rij_d/c))
            gx = np.tanh(smoothed_rij_d/m)

            # Potential function
            px = -fx*(gx**2)
            return px

        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = qi - qj[i, :]
            p = custom_potential_function(
                qi=qi, qj=qj[i, :], d=r)
            u_sum += p * utils.unit_vector(uij) + 0.0 * pj[i, :]
        return u_sum / qj.shape[0]

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

    def _density(self, si: np.ndarray, sj: np.ndarray, k: float, d: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = np.abs(np.linalg.norm(sij) - d) * utils.unit_vector(sij)
            w_sum += w
        return w_sum

    def _calc_density(self, idx: int,
                      neighbors_idxs: np.ndarray,
                      herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        density = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = herd_states[neighbors_idxs, :2]
            density = self._density(si=qi, sj=qj, k=0.375, d=0)
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

            # Herd shepherd density
            delta_adj_vec = self._get_delta_adjacency_vector(
                shepherd_state=herd_states[idx, :2],
                herd_states=shepherd_states,
                r=self._sensing_range)

            qi = herd_states[idx, :2]
            qj = shepherd_states[delta_adj_vec, :2]
            density = self._density(si=qi, sj=qj, k=0.375, d=r_shepherd)
            herd_densities[idx] += density
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
                                bound: float, k: float):
        def custom_sigmoid(qi: np.ndarray, qj: np.ndarray, d: float,
                           bound: float):
            qij = qi - qj
            rij = np.linalg.norm(qij)
            smoothed_rij_d = rij - d
            sigma = (bound/(1 + np.exp(-smoothed_rij_d))) - \
                (bound/(0.1 + np.exp(smoothed_rij_d)))
            # sigma = np.round(sigma, 3)
            return sigma

        def local_crowd_horizon(qi: np.ndarray, qj: np.ndarray,
                                r: float,
                                k: float = 0.5):
            qij = qi - qj
            return (1/(1 + k * ((np.linalg.norm(qij)))))
            # return np.linalg.norm(qij)

        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = qj[i, :] - qi
            gain = local_crowd_horizon(qi=qi, qj=qj[i, :], r=d, k=k)
            # gain = 1
            sigma = custom_sigmoid(qi=qi, qj=qj[i, :], d=d, bound=bound)
            # sigma = custom_exponential(qi=qi, qj=qj[i, :], d=d, bound=bound)
            # print(f"sigma = {sigma}")
            u_sum += gain * sigma * utils.unit_vector(uij) * 3
        return u_sum

    def _potential_edge_following(self, qi: np.ndarray, qj: np.ndarray,
                                  pi: np.ndarray, pj: np.ndarray,
                                  d: float,
                                  d_dead: float,
                                  gain: float):
        def custom_potential(qi: np.ndarray, qj: np.ndarray,
                             d: float,
                             d_dead: float,
                             gain: float):
            qij = qi - qj
            rij = np.linalg.norm(qij)
            smoothed_rij_d = rij - d
            c = 10
            m = 20

            fx = gain*(1 - np.exp(-smoothed_rij_d/c))
            gx = np.tanh(smoothed_rij_d/m)

            # Potential function
            px = -fx*(gx**2)
            return px

        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = qi - qj[i, :]
            p = custom_potential(
                qi=qi, qj=qj[i, :], d=d, d_dead=d_dead, gain=gain)
            u_sum += (p * utils.unit_vector(uij) + 0.08 * (pj[i, :]))
        return u_sum / qj.shape[0]

    def _formation_stable(self):
        if len(self._total_energy) != self._time_horizon:
            return False
        y = np.array(self._total_energy)
        x = np.linspace(0, self._time_horizon,
                        self._time_horizon, endpoint=False)
        coeff, err, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        if math.sqrt(err) >= 1200:
            return False
        poly = np.poly1d(coeff)
        polyder = np.polyder(poly)
        cond = np.abs(np.round(float(polyder.coef[0]), 1))
        return not bool(cond)

    def _get_beta_adjacency_vector(self, state: np.ndarray,
                                   obstacles: list, r: float) -> np.ndarray:
        adj_vec = []
        for obstacle in obstacles:
            adj_vec.append(obstacle.in_entity_radius(state[:2], r=r))
        return np.array(adj_vec)

    def _obstacle_avoidance(self, qi: np.ndarray,
                            pi: np.ndarray,
                            beta_adj_vec: np.ndarray,
                            obstacle_list: list,
                            r: float,
                            gain: float):
        def custom_potential_function(qi: np.ndarray, qj: np.ndarray,
                                      d: float):
            qij = qi - qj
            rij = np.linalg.norm(qij)
            smoothed_rij_d = rij - d
            c = 6
            m = 10

            fx = gain*(- np.exp(-smoothed_rij_d/c))
            gx = np.tanh(smoothed_rij_d/m)

            # Potential function
            px = -fx*(gx**2)
            return px

        obs_in_radius = np.where(beta_adj_vec)
        beta_agents = np.array([]).reshape((0, 4))

        u_sum = np.zeros(2).astype(np.float64)
        for obs_idx in obs_in_radius[0]:
            beta_agent = obstacle_list[obs_idx].induce_beta_agent(
                qi, pi)
            qj = beta_agent[:2]
            qij = qi - qj
            p = custom_potential_function(
                qi=qi, qj=qj, d=r)
            u_sum += p * utils.unit_vector(qij) * 5
        return u_sum
