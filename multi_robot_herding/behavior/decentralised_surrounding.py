# !/usr/bin/python3

import math
import pygame
import numpy as np
from collections import deque

from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class PotentialFunc(object):
    @staticmethod
    def proposed_func(xi: np.ndarray, xj: np.ndarray,
                      d: float,
                      attract: bool = True,
                      repulse: bool = True,
                      c: float = 10,
                      m: float = 10):
        a = int(attract)
        r = int(repulse)

        xij = xi - xj
        xij_norm = np.linalg.norm(xij)

        n_xij_d = xij_norm - d
        fx = a - r * np.exp(-n_xij_d/c)
        gx = np.tanh(n_xij_d/m)

        # Potential function
        px = -fx*(gx**2)
        return px

    @staticmethod
    def density_func(xi: np.ndarray, xj: np.ndarray,
                     k: float):
        pass


class DecentralisedSurrounding(DecentralisedBehavior):
    def __init__(self,
                 potential_func: dict,
                 Cs: float = 30.0,
                 Cr: float = 2.0,
                 Cv: float = 1.2,
                 Co: float = 1.0,
                 distance_to_target: float = 125.0,
                 interagent_spacing: float = 150.0,
                 obstacle_range: float = 40.0,
                 sensing_range: float = 700.0):
        super().__init__()

        self._potential_func = potential_func
        self._Cs = Cs
        self._Cr = Cr
        self._Cv = Cv
        self._Co = Co

        self._distance_to_target = distance_to_target
        self._interagent_spacing = interagent_spacing
        self._obstacle_range = obstacle_range
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

        self._plot_force = False
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
               consensus_states: dict):
        # Control signal
        self._pose = state[:2]
        u = np.zeros(2)
        all_shepherd_states = np.vstack((state, other_states))

        d_to_target = self._distance_to_target
        spacing = self._interagent_spacing

        # herd_density = self._herd_density(herd_states=herd_states,
        #                                   shepherd_states=all_shepherd_states,
        #                                   r_shepherd=d_to_target)
        # total_density = np.sum(np.linalg.norm(herd_density, axis=1))
        # self._total_energy.append(total_density)

        delta_adjacency_vector = self._get_delta_adjacency_vector(
            herd_states,
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
            sj = herd_states[neighbor_herd_idxs, :2]
            s_dot_j = herd_states[neighbor_herd_idxs, 2:4]
            ps = self._potential_edge_following(qi=di,
                                                qj=sj,
                                                d=d_to_target,
                                                gain=self._Cs)
        self._force_ps = ps

        po = np.zeros(2)
        neighbor_shepherd_idxs = alpha_adjacency_matrix[0]

        if sum(neighbor_shepherd_idxs) > 0:
            dj = all_shepherd_states[neighbor_shepherd_idxs, :2]
            po = self._collision_avoidance_term(
                gain=self._Cr,
                qi=di, qj=dj,
                d=spacing)
        self._force_po = po

        # Velocity consensus
        pv = np.zeros((2,))
        if sum(neighbor_herd_idxs) > 0:
            sj = herd_states[neighbor_herd_idxs, :2]
            s_dot_j = herd_states[neighbor_herd_idxs, 2:4]
            pv = self._velocity_consensus(pj=s_dot_j,
                                          gain=self._Cv)

        # Obstacle avoidance term
        beta_adj_vec = self._get_beta_adjacency_vector(state=state,
                                                       obstacles=obstacles,
                                                       r=self._sensing_range)
        p_avoid = np.zeros((2,))
        if sum(beta_adj_vec) > 0:
            p_avoid = self._obstacle_avoidance(qi=di, pi=d_dot_i,
                                               beta_adj_vec=beta_adj_vec,
                                               obstacle_list=obstacles,
                                               d=self._obstacle_range,
                                               gain=self._Co)
        u = ps + po + pv + p_avoid
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
            pygame.draw.circle(screen, pygame.Color("yellow"),
                               tuple(self._pose), self._sensing_range, 1)

        return super().display(screen)

    def _collision_avoidance_term(self, gain: float,
                                  qi: np.ndarray, qj: np.ndarray,
                                  d: float):

        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = qi - qj[i, :]
            func_input = self._potential_func['col_avoid']
            func_input.update({'xi': qi, 'xj': qj[i, :], 'd': d})
            p = PotentialFunc.proposed_func(**func_input)
            u_sum += gain * p * utils.unit_vector(uij)
        return u_sum / qj.shape[0]

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

    def _potential_edge_following(self, qi: np.ndarray, qj: np.ndarray,
                                  d: float,
                                  gain: float):
        u_sum = np.zeros(2).astype(np.float64)
        for i in range(qj.shape[0]):
            uij = qi - qj[i, :]
            func_input = self._potential_func['edge_follow']
            func_input.update({'xi': qi, 'xj': qj[i, :], 'd': d})
            p = PotentialFunc.proposed_func(**func_input)
            u_sum += gain * p * utils.unit_vector(uij)
        return u_sum / qj.shape[0]

    def _velocity_consensus(self, pj: np.ndarray, gain: float):
        u_sum = np.zeros(2).astype(np.float64)
        for i in range(pj.shape[0]):
            u_sum += pj[i, :]
        return gain * u_sum / pj.shape[0]

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
                            d: float,
                            gain: float):
        obs_in_radius = np.where(beta_adj_vec)
        u_sum = np.zeros(2).astype(np.float64)
        for obs_idx in obs_in_radius[0]:
            beta_agent = obstacle_list[obs_idx].induce_beta_agent(
                qi, pi)
            qj = beta_agent[:2]
            qij = qi - qj
            func_input = self._potential_func['obs_avoid']
            func_input.update({'xi': qi, 'xj': qj, 'd': d})
            p = PotentialFunc.proposed_func(**func_input)
            u_sum += gain * p * utils.unit_vector(qij)
        return u_sum

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
