# !/usr/bin/python3

import pygame
import numpy as np

from multi_robot_herding.utils import utils


class FormationUtils():

    @staticmethod
    def calc_bearing(adj_matrix: np.ndarray,
                     global_rigid: bool = True):
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

    @staticmethod
    def calc_H(start: np.ndarray, end: np.ndarray):
        n = max(start) + 1
        m = len(start)
        H = np.zeros((m, n))
        for i in range(m):
            H[i, start[i]] = -1
            H[i, end[i]] = 1
        return H

    @staticmethod
    def calc_H_bar(start: np.ndarray, end: np.ndarray):
        n = max(start) + 1
        m = len(start)
        H = np.zeros((m, n))
        for i in range(m):
            H[i, start[i]] = -1
            H[i, end[i]] = 1
        H_bar = np.kron(H, np.eye(2))
        return H_bar

    @staticmethod
    def calc_g(p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g_vec = np.empty((0, p.shape[1]))
        g_norm_vec = []
        for i in range(len(start)):
            g = utils.unit_vector(p[end[i]] - p[start[i]])
            g_vec = np.vstack((g_vec, g))
            g_norm_vec.append(np.linalg.norm(p[end[i]] - p[start[i]]))
        return g_vec, np.array(g_norm_vec)

    @staticmethod
    def calc_diagPg(g: np.ndarray, e_norm: np.ndarray):
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

    @staticmethod
    def calc_Rd(p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g, e_norm = FormationUtils.calc_g(p, start, end)
        diagPg = FormationUtils.calc_diagPg(g, e_norm)
        H_bar = FormationUtils.calc_H_bar(start, end)
        Rd = diagPg @ H_bar
        return Rd

    @staticmethod
    def calc_BL(p: np.ndarray, start: np.ndarray, end: np.ndarray):
        g, e_norm = FormationUtils.calc_g(p, start, end)
        diagPg = FormationUtils.calc_diagPg(g, e_norm)
        H_bar = FormationUtils.calc_H_bar(start, end)
        return H_bar.transpose() @ diagPg @ H_bar


class FormationController():
    def __init__(self, kp: float = 6.5, kd: float = 0.1,
                 kv: float = 1.0, ki: float = 1.0, dt: float = 0.2):
        self._p_star = None
        self._g_star = None

        self._init_error_pose = False
        self._error_pose_ij = None

        # Parameters
        self._kp = kp
        self._kd = kd
        self._kv = kv
        self._ki = ki
        self._dt = dt

    def set_formation(self, formation_states: np.ndarray):

        p_star = formation_states
        start_end = FormationUtils.calc_bearing(
            np.ones((formation_states.shape[0], formation_states.shape[0])),
            global_rigid=True)
        H_bar = FormationUtils.calc_H_bar(
            start=start_end[0, :], end=start_end[1, :])
        g_star, e_norm = FormationUtils.calc_g(
            p_star, start_end[0, :], start_end[1, :])
        diagP = FormationUtils.calc_diagPg(g_star, e_norm)
        Rd = diagP @ H_bar
        n = formation_states.shape[0]
        if np.linalg.matrix_rank(Rd) != (2*n - 3):
            return False
        self._p_star = p_star
        self._g_star = g_star
        return True

    def enforce_formation(self, states: np.ndarray):
        u = np.zeros((states.shape[0], 2))
        p = states
        p_star = self._g_star

        start_end = FormationUtils.calc_bearing(
            np.ones((self._p_star.shape[0], self._p_star.shape[0])),
            global_rigid=True)
        H_bar = FormationUtils.calc_H_bar(
            start=start_end[0, :], end=start_end[1, :])

        g, e_norm = FormationUtils.calc_g(
            p, start_end[0, :], start_end[1, :])
        diagP = FormationUtils.calc_diagPg(g, e_norm)
        flatten_p_star = np.ravel(p_star)
        u = H_bar.transpose() @ diagP @ flatten_p_star
        u = np.reshape(u, (int(u.size/2), 2))
        return u

    def maneuver_formation(self, states: np.ndarray):
        u = np.zeros((states.shape[0], 2))
        p_star = self._p_star
        p = states[:, :2]

        start_end = FormationUtils.calc_bearing(
            np.ones((states.shape[0], states.shape[0])), global_rigid=True)
        g, _ = FormationUtils.calc_g(p_star, start_end[0, :], start_end[1, :])

        if not self._init_error_pose:
            self._error_pose_ij = np.zeros(
                (1, 2, start_end.shape[1], start_end.shape[1]))
            self._init_error_pose = True

        # Decentralise control
        leader_idx = [0, 1]
        all_total_Pg_ij = np.zeros(
            (2, 2, states.shape[0]))
        all_Pg_ij = np.zeros((2, 2, start_end.shape[1], start_end.shape[1]))

        # Centroid
        centroid_p_star = np.zeros((1, 2))
        for idx in range(states.shape[0]):
            centroid_p_star += states[idx, :2]
        centroid_p_star = centroid_p_star/states.shape[0]

        # Scale
        scale = 0
        for idx in range(states.shape[0]):
            scale += np.linalg.norm(
                states[idx, :2] - centroid_p_star) ** 2
        scale = np.sqrt(scale / states.shape[0])

        total_bearing_error = 0
        for idx in range(start_end.shape[1]):
            pair_ij = start_end[:, idx]
            g_star, e_norm_star = FormationUtils.calc_g(
                p_star, [pair_ij[0]], [pair_ij[1]])
            g, _ = FormationUtils.calc_g(p, [pair_ij[0]], [pair_ij[1]])
            total_bearing_error += np.round(np.linalg.norm(g - g_star), 4)
            if start_end[0, idx] in leader_idx:
                continue
            Pg_ij = FormationUtils.calc_diagPg(g_star, e_norm_star)
            all_total_Pg_ij[:, :, pair_ij[0]] += Pg_ij
            all_Pg_ij[:, :, pair_ij[0], pair_ij[1]] = Pg_ij

        for i in range(states.shape[0]):
            if i in leader_idx:
                target = np.array(pygame.mouse.get_pos())
                vc = -5 * utils.unit_vector(centroid_p_star - target)
                u[i] = u[i] + vc
                continue

            ui = np.zeros((2, 1))
            Ki = all_total_Pg_ij[:, :, i]
            for j in range(start_end.shape[0]):
                if i == j:
                    continue
                Pg_ij = all_Pg_ij[:, :, i, j]
                pi = states[i, :2]
                pj = states[j, :2]
                vi = states[i, 2:4]
                vj = states[j, 2:4]
                vj_dot = states[j, 4:]

                # Integral term
                self._error_pose_ij[:, :, i, j] += (pi - pj) * self._dt
                integral_term = self._ki * self._error_pose_ij[:, :, i, j]
                derivative_term = self._kd * (pi - pj) / self._dt

                ui += Pg_ij @ (self._kp * (pi - pj) + self._kv *
                               (vi - vj) - vj_dot + integral_term + derivative_term).reshape((2, 1))

            ui = -np.linalg.inv(Ki) @ ui
            u[i] = ui.reshape((1, 2))
        return u
