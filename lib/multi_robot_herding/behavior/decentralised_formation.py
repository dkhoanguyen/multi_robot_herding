# !/usr/bin/python3
import math
import pygame
import numpy as np
from collections import deque

from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.utils import utils


class DecentralisedFormation(DecentralisedBehavior):
    def __init__(self,
                 id: int):
        super().__init__()

        # Params
        self._id = id

        self._time_horizon = 200
        self._total_energy = deque(maxlen=self._time_horizon)
        self._stablised_energy = deque(maxlen=10)

        self._formation_generated = False
        self._p_star = np.empty((0, 2))
        self._plotting_p_star = np.empty((0, 4))
        self._g_star = np.empty((0, 2))
        self._trigger_formation = False
        self._formation = np.empty((0, 2))

        self._centroid_p_star = np.zeros((1, 2))

        self._error_pose_ij = None
        self._init_error_pose = False

    def transition(self, state: np.ndarray,
                   other_states: np.ndarray,
                   herd_states: np.ndarray,
                   consensus_states: dict):
        all_shepherd_states = np.vstack((state, other_states))
        herd_density = self._herd_density(herd_states=herd_states,
                                          shepherd_states=all_shepherd_states,
                                          r_shepherd=10)
        total_density = np.sum(np.linalg.norm(herd_density, axis=1))
        self._total_energy.append(total_density)

        formation_stable = self._formation_stable()

        if formation_stable:
            self._stablised_energy.append(
                sum(self._total_energy)/len(self._total_energy))
        if len(self._stablised_energy) == self._stablised_energy.maxlen:
            return True
        return False
        # events = pygame.event.get()
        # for event in events:
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_DOWN:
        #             return True

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               herd_states: np.ndarray,
               obstacles: list,
               consensus_states: dict,
               output_consensus_state: dict):
        u = np.zeros(2)
        all_shepherd_states = np.vstack((state, other_states))

        if not self._formation_generated:
            ret, p_star = self._generate_formation(raw_states[:, :2])
            if ret:
                self._g_star = p_star
                self._formation_generated = True

        u = self._maneuver_formation(
            shepherd_states=raw_states, formation=self._g_star)
        return u

    def display(self, screen: pygame.Surface):
        return super().display(screen)

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
            # # Herd internal density
            # neighbor_idxs = alpha_adjacency_matrix[idx]
            # density = self._calc_density(
            #     idx=idx, neighbors_idxs=neighbor_idxs,
            #     herd_states=herd_states)
            # herd_densities[idx] += density

            # Herd shepherd density
            delta_adj_vec = self._get_delta_adjacency_vector(
                shepherd_state=herd_states[idx, :2],
                herd_states=shepherd_states,
                r=250)

            qi = herd_states[idx, :2]
            qj = shepherd_states[delta_adj_vec, :2]
            density = self._density(si=qi, sj=qj, k=0.375, d=r_shepherd)
            herd_densities[idx] += density
        return herd_densities

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

    def _formation_stable(self):
        if len(self._total_energy) != self._time_horizon:
            return False
        y = np.array(self._total_energy)
        x = np.linspace(0, self._time_horizon,
                        self._time_horizon, endpoint=False)
        coeff, err, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        if math.sqrt(err) >= 110:
            return False
        poly = np.poly1d(coeff)
        polyder = np.polyder(poly)
        cond = np.abs(np.round(float(polyder.coef[0]), 1))
        return not bool(cond)

    # Formation
    def _generate_formation(self, states: np.ndarray):
        p_star = states
        start_end = self._calc_start_end(
            np.ones((states.shape[0], states.shape[0])), global_rigid=True)
        H_bar = self._calc_H_bar(start=start_end[0, :], end=start_end[1, :])
        g_star, e_norm = self._calc_g(p_star, start_end[0, :], start_end[1, :])
        diagP = self._calc_diagPg(g_star, e_norm)
        Rd = diagP @ H_bar
        n = states.shape[0]
        if np.linalg.matrix_rank(Rd) != (2*n - 3):
            return False, np.empty((0, 2))
        self._p_star = p_star
        return True, g_star

    def _enforce_formation(self, states: np.ndarray):
        u = np.zeros((states.shape[0], 2))
        p = states
        p_star = self._g_star

        start_end = self._calc_start_end(
            np.ones((len(self._shepherds), len(self._shepherds))), global_rigid=True)
        H_bar = self._calc_H_bar(start=start_end[0, :], end=start_end[1, :])

        g, e_norm = self._calc_g(p, start_end[0, :], start_end[1, :])
        diagP = self._calc_diagPg(g, e_norm)
        flatten_p_star = np.ravel(p_star)
        u = H_bar.transpose() @ diagP @ flatten_p_star
        u = np.reshape(u, (int(u.size/2), 2))
        return u

    def _maneuver_formation(self, shepherd_states: np.ndarray,
                            formation: np.ndarray):
        u = np.zeros((shepherd_states.shape[0], 2))
        if formation.shape[0] == 0:
            return u
        p_star = self._p_star
        p = shepherd_states[:, :2]

        start_end = self._calc_start_end(
            np.ones((shepherd_states.shape[0], shepherd_states.shape[0])), global_rigid=True)
        g, _ = self._calc_g(p_star, start_end[0, :], start_end[1, :])

        if not self._init_error_pose:
            self._error_pose_ij = np.zeros(
                (1, 2, start_end.shape[1], start_end.shape[1]))
            self._init_error_pose = True

        # Decentralise control
        leader_idx = [0, 1]
        all_total_Pg_ij = np.zeros(
            (2, 2, shepherd_states.shape[0]))
        all_Pg_ij = np.zeros((2, 2, start_end.shape[1], start_end.shape[1]))

        # Centroid
        centroid_p_star = np.zeros((1, 2))
        for idx in range(shepherd_states.shape[0]):
            centroid_p_star += shepherd_states[idx, :2]
        centroid_p_star = centroid_p_star/shepherd_states.shape[0]

        # Scale
        scale = 0
        for idx in range(shepherd_states.shape[0]):
            scale += np.linalg.norm(
                shepherd_states[idx, :2] - centroid_p_star) ** 2
        scale = np.sqrt(scale / shepherd_states.shape[0])

        self._plotting_p_star = np.empty((0, 4))
        total_bearing_error = 0
        for idx in range(start_end.shape[1]):
            pair_ij = start_end[:, idx]
            g_star, e_norm_star = self._calc_g(
                p_star, [pair_ij[0]], [pair_ij[1]])
            g, _ = self._calc_g(p, [pair_ij[0]], [pair_ij[1]])
            total_bearing_error += np.round(np.linalg.norm(g - g_star), 3)
            if start_end[0, idx] in leader_idx:
                continue
            Pg_ij = self._calc_diagPg(g_star, e_norm_star)
            all_total_Pg_ij[:, :, pair_ij[0]] += Pg_ij
            all_Pg_ij[:, :, pair_ij[0], pair_ij[1]] = Pg_ij

        kp = 0.04
        kd = 0.0
        kv = 0.07
        ki = 0.085

        dt = 0.2
        test_list = [0, 1, 2, 3, 4, 5]
        target = np.array(pygame.mouse.get_pos())
        target = np.array([700,350])
        vc = -0.5 * utils.unit_vector(centroid_p_star - target)

        for i in range(shepherd_states.shape[0]):
            if i in leader_idx:
                v = np.array([1,0])
                u[i] = u[i] + vc
                continue

            ui = np.zeros((2, 1))
            Ki = all_total_Pg_ij[:, :, i]

            for j in test_list:
                if i == j:
                    continue

                Pg_ij = all_Pg_ij[:, :, i, j]
                pi = shepherd_states[i, :2]
                pj = shepherd_states[j, :2]
                vi = shepherd_states[i, 2:4]
                vj = shepherd_states[j, 2:4]
                vj_dot = shepherd_states[j, 4:]

                # Integral term
                self._error_pose_ij[:, :, i, j] += (pi - pj) * dt
                integral_term = ki * self._error_pose_ij[:, :, i, j]
                derivative_term = kd * (pi - pj) / dt

                ui += Pg_ij @ (kp * (pi - pj) + kv *
                               (vi - vj) - vj_dot + integral_term + derivative_term).reshape((2, 1))
            ui = -np.linalg.inv(Ki) @ ui
            u[i] = ui.reshape((1, 2))
        return u[self._id]

    # Private
    def _calc_start_end(self, adj_matrix: np.ndarray,
                        global_rigid: bool = True) -> np.ndarray:
        start = []
        end = []

        # Fully connected graph
        if global_rigid:
            adj_matrix = np.ones_like(adj_matrix)
            np.fill_diagonal(adj_matrix, 0)

        for i in range(len(adj_matrix)):
            start_i = []
            end_i = []
            for j in range(len(adj_matrix[i, :])):
                if adj_matrix[i, j] == 0:
                    continue
                start_i.append(i)
                if adj_matrix[i, j] and j not in end_i:
                    end_i.append(j)
            start = start + start_i
            end = end + end_i
        return np.array([start, end])

    def _calc_H(self, start: np.ndarray, end: np.ndarray):
        n = max(start) + 1
        m = len(start)
        H = np.zeros((m, n))
        for i in range(m):
            H[i, start[i]] = -1
            H[i, end[i]] = 1
        return H

    def _calc_H_bar(self, start: np.ndarray, end: np.ndarray):
        n = max(start) + 1
        m = len(start)
        H = np.zeros((m, n))
        for i in range(m):
            H[i, start[i]] = -1
            H[i, end[i]] = 1
        H_bar = np.kron(H, np.eye(2))
        return H_bar

    def _calc_g(self, p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g_vec = np.empty((0, p.shape[1]))
        g_norm_vec = []
        for i in range(len(start)):
            g = utils.unit_vector(p[end[i]] - p[start[i]])
            g_vec = np.vstack((g_vec, g))
            g_norm_vec.append(np.linalg.norm(p[end[i]] - p[start[i]]))
        return g_vec, np.array(g_norm_vec)

    def _calc_diagPg(self, g: np.ndarray, e_norm: np.ndarray):
        Pg = np.empty((0, g.shape[1]))
        for i in range(g.shape[0]):
            gk = g[i, :]
            Pgk = utils.MathUtils.orthogonal_projection_matrix(gk) / e_norm[i]
            Pg = np.vstack((Pg, Pgk))
        n = Pg.shape[0]
        total = Pg.shape[0] / Pg.shape[1]
        diagPg = np.eye(n)
        for element in range(int(total)):
            m = Pg.shape[1]
            diagPg[m*element:m*element+2, m*element:m*element +
                   2] = Pg[m*element:m*element+2, :]
        return diagPg

    def _calc_Rd(self, p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g, e_norm = self._calc_g(p, start, end)
        diagPg = self._calc_diagPg(g, e_norm)
        H_bar = self._calc_H_bar(start, end)
        Rd = diagPg @ H_bar
        return Rd

    def _calc_BL(self, p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g, e_norm = self._calc_g(p, start, end)
        diagPg = self._calc_diagPg(g, e_norm)
        H_bar = self._calc_H_bar(start, end)
        return H_bar.transpose() @ diagPg @ H_bar
