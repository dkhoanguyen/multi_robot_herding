# !/usr/bin/python3
import math
import time

import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Obstacle

from scipy.spatial import ConvexHull


class MathUtils():

    EPSILON = 0.1
    H = 0.2
    A, B = 5, 5
    C = np.abs(A-B)/np.sqrt(4*A*B)  # phi

    R = 40
    D = 40

    @staticmethod
    def sigma_1(z):
        return z / np.sqrt(1 + z**2)

    @staticmethod
    def sigma_norm(z, e=EPSILON):
        return (np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2) - 1) / e

    @staticmethod
    def sigma_norm_grad(z, e=EPSILON):
        return z/np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2)

    @staticmethod
    def bump_function(z, h=H):
        ph = np.zeros_like(z)
        ph[z <= 1] = (1 + np.cos(np.pi * (z[z <= 1] - h)/(1 - h)))/2
        ph[z < h] = 1
        ph[z < 0] = 0
        return ph

    @staticmethod
    def phi(z, a=A, b=B, c=C):
        return ((a + b) * MathUtils.sigma_1(z + c) + (a - b)) / 2

    @staticmethod
    def phi_alpha(z, r=R, d=D):
        r_alpha = MathUtils.sigma_norm([r])
        d_alpha = MathUtils.sigma_norm([d])
        return MathUtils.bump_function(z/r_alpha) * MathUtils.phi(z-d_alpha)

    @staticmethod
    def normalise(v, pre_computed=None):
        n = pre_computed if pre_computed is not None else math.sqrt(
            v[0]**2 + v[1]**2)
        if n < 1e-13:
            return np.zeros(2)
        else:
            return np.array(v) / n


class MathematicalFlock(Behavior):
    C1_alpha = 3
    C2_alpha = 2 * np.sqrt(C1_alpha)
    C1_beta = 20
    C2_beta = 2 * np.sqrt(C1_beta)
    C1_gamma = 5
    C2_gamma = 0.2 * np.sqrt(C1_gamma)

    ALPHA_RANGE = 40
    ALPHA_DISTANCE = 40
    ALPHA_ERROR = 5
    BETA_RANGE = 30
    BETA_DISTANCE = 30

    def __init__(self, follow_cursor: bool,
                 sensing_range: float,
                 danger_range: float,
                 initial_consensus: np.ndarray):
        super().__init__()
        self._herds = []
        self._shepherds = []
        self._obstacles = []

        self._sample_t = 0
        self._pause_agents = np.zeros(1)

        self._follow_cursor = follow_cursor
        self._sensing_range = sensing_range
        self._danger_range = danger_range
        self._consensus_pose = np.array(initial_consensus)

        self._start_time = time.time()
        self._stop = False

        # For control
        self._mass = 0
        self._flocking_condition = 1

    # Herd
    def add_herd(self, herd: Herd):
        self._herds.append(herd)

    # Shepherd
    def add_shepherd(self, shepherd: Shepherd):
        self._shepherds.append(shepherd)

    # Obstacle
    def add_obstacle(self, obstacle: Obstacle):
        self._obstacles.append(obstacle)

    def set_consensus(self, consensus: np.ndarray):
        self._consensus = consensus

    def get_herd_mean(self):
        herd: Herd
        herd_states = np.array([]).reshape((0, 2))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            herd_states = np.vstack(
                (herd_states, herd.pose))
        return np.sum(herd_states, axis=0) / herd_states.shape[0]

    def update(self, dt: float):
        self._flocking(dt)
        # self._edge_following()

    def _remain_in_screen(self, herd: Herd):
        if herd.pose[0] > params.SCREEN_WIDTH - 700:
            herd.steer(np.array([-params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[0] < params.BOX_MARGIN:
            herd.steer(np.array([params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[1] < params.BOX_MARGIN:
            herd.steer(np.array([0., params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[1] > params.SCREEN_HEIGHT - params.BOX_MARGIN:
            herd.steer(np.array([0., -params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)

    # Mathematical model of flocking
    def _flocking(self, *args, **kwargs):
        events = self._get_events(args)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and not self._stop:
                    self._stop = True
                if event.key == pygame.K_UP and self._stop:
                    self._stop = False

        herd: Herd
        herd_states = np.array([]).reshape((0, 4))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            herd_states = np.vstack(
                (herd_states, np.hstack((herd.pose, herd.velocity))))

        u = np.zeros((len(self._herds), 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states,
                                                                  r=self._sensing_range)
        lower_tril = np.tril(alpha_adjacency_matrix, -1)
        epsilon = sum(sum(lower_tril))

        beta_adjacency_matrix = self._get_beta_adjacency_matrix(herd_states,
                                                                self._obstacles,
                                                                r=self._sensing_range)
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(herd_states,
                                                                  self._shepherds,
                                                                  r=self._sensing_range)

        total_pairwise_sum = 0
        for idx, herd in enumerate(self._herds):
            # Plotting config
            herd._plot_force = False
            herd._plot_force_mag = False

            qi = herd_states[idx, :2]
            pi = herd_states[idx, 2:]

            # Alpha agent
            u_alpha = np.zeros(2)
            neighbor_idxs = alpha_adjacency_matrix[idx]

            if sum(neighbor_idxs) > 0:
                qj = herd_states[neighbor_idxs, :2]
                pj = herd_states[neighbor_idxs, 2:]
                
                pairwise_potential_mag = (1/(1 + sum(neighbor_idxs))) * self._pairwise_potential_mag(
                    qi, qj, 30)
                total_pairwise_sum += pairwise_potential_mag

                density = self._density(qi, qj, 0.375)

                herd._force = density * 5
                herd._plot_force = True
                herd._force_mag = pairwise_potential_mag / 200

                alpha_grad = self._gradient_term(
                    c=MathematicalFlock.C2_alpha, qi=qi, qj=qj,
                    r=MathematicalFlock.ALPHA_RANGE,
                    d=MathematicalFlock.ALPHA_DISTANCE)

                alpha_consensus = self._velocity_consensus_term(
                    c=MathematicalFlock.C2_alpha,
                    qi=qi, qj=qj,
                    pi=pi, pj=pj,
                    r=MathematicalFlock.ALPHA_RANGE)
                u_alpha = alpha_grad + alpha_consensus

            # Beta agent
            u_beta = np.zeros(2)
            obstacle_idxs = beta_adjacency_matrix[idx]
            if sum(obstacle_idxs) > 0:
                # Create beta agent
                obs_in_radius = np.where(beta_adjacency_matrix[idx] > 0)
                beta_agents = np.array([]).reshape((0, 4))
                for obs_idx in obs_in_radius[0]:
                    beta_agent = self._obstacles[obs_idx].induce_beta_agent(
                        herd)
                    beta_agents = np.vstack((beta_agents, beta_agent))

                qik = beta_agents[:, :2]
                pik = beta_agents[:, 2:]
                beta_grad = self._gradient_term(
                    c=MathematicalFlock.C2_beta, qi=qi, qj=qik,
                    r=MathematicalFlock.BETA_RANGE,
                    d=MathematicalFlock.BETA_DISTANCE)

                beta_consensus = self._velocity_consensus_term(
                    c=MathematicalFlock.C2_beta,
                    qi=qi, qj=qik,
                    pi=pi, pj=pik,
                    r=MathematicalFlock.BETA_RANGE)
                u_beta = beta_grad + beta_consensus

            # Gamma agent
            target = self._consensus_pose
            if self._follow_cursor:
                target = np.array(pygame.mouse.get_pos())

            u_gamma = self._group_objective_term(
                c1=MathematicalFlock.C1_gamma,
                c2=MathematicalFlock.C2_gamma,
                pos=target,
                qi=qi,
                pi=pi)

            # Delta agent (shepherd)
            u_delta = np.zeros(2)
            delta_idxs = delta_adjacency_matrix[idx]
            if sum(delta_idxs) > 0:
                # Create delta_agent
                delta_in_radius = np.where(delta_adjacency_matrix[idx] > 0)
                delta_agents = np.array([]).reshape((0, 4))
                for del_idx in delta_in_radius[0]:
                    delta_agent = self._shepherds[del_idx].induce_delta_agent(
                        herd)
                    delta_agents = np.vstack((delta_agents, delta_agent))

                qid = delta_agents[:, :2]
                pid = delta_agents[:, 2:]
                delta_grad = self._gradient_term(
                    c=MathematicalFlock.C2_beta, qi=qi, qj=qid,
                    r=MathematicalFlock.BETA_RANGE,
                    d=MathematicalFlock.BETA_DISTANCE)
                delta_consensus = self._velocity_consensus_term(
                    c=MathematicalFlock.C2_beta,
                    qi=qi, qj=qid,
                    pi=pi, pj=pid,
                    r=MathematicalFlock.BETA_RANGE)
                u_delta = delta_grad + delta_consensus

            u_delta += self._predator_avoidance_term(
                si=qi, r=self._danger_range, k=50000)

            # u_delta = 0
            # Ultimate flocking model
            u[idx] = u_alpha + u_beta + \
                self._flocking_condition * (u_gamma + u_delta)

        total_pairwise_sum = total_pairwise_sum/(1 + epsilon)

        # Control for the agent
        qdot = u
        herd_states[:, 2:] += qdot * 0.1
        pdot = herd_states[:, 2:]
        herd_states[:, :2] += pdot * 0.2

        herd: Herd
        for idx, herd in enumerate(self._herds):
            # self._remain_in_screen(herd)
            herd.velocity = herd_states[idx, 2:]
            herd.pose = herd_states[idx, :2]
            herd._rotate_image(herd.velocity)
            herd.reset_steering()

    def _gradient_term(self, c: float, qi: np.ndarray, qj: np.ndarray,
                       r: float, d: float):
        n_ij = self._get_n_ij(qi, qj)
        return c * np.sum(MathUtils.phi_alpha(
            MathUtils.sigma_norm(qj-qi),
            r=r,
            d=d)*n_ij, axis=0)

    def _velocity_consensus_term(self, c: float, qi: np.ndarray,
                                 qj: np.ndarray, pi: np.ndarray,
                                 pj: np.ndarray, r: float):
        # Velocity consensus term
        a_ij = self._get_a_ij(qi, qj, r)
        return c * np.sum(a_ij*(pj-pi), axis=0)

    def _group_objective_term(self, c1: float, c2: float,
                              pos: np.ndarray, qi: np.ndarray, pi: np.ndarray):
        # Group objective term
        return -c1 * MathUtils.sigma_1(qi - pos) - c2 * (pi)

    def _predator_avoidance_term(self, si: np.ndarray, r: float, k: float):
        shepherd: Shepherd
        si_dot = np.zeros(2)
        for shepherd in self._shepherds:
            di = shepherd.pose.reshape(2)
            if np.linalg.norm(di - si) <= r:
                si_dot += -k * (di - si)/(np.linalg.norm(di - si))**3
        return si_dot

    def _get_alpha_adjacency_matrix(self, herd_states: np.ndarray, r: float) -> np.ndarray:
        adj_matrix = np.array([np.linalg.norm(herd_states[i, :2]-herd_states[:, :2], axis=-1) <= r
                               for i in range(len(herd_states))])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _get_beta_adjacency_matrix(self, agents: np.ndarray,
                                   obstacles: list, r: float) -> np.ndarray:
        adj_matrix = np.array([]).reshape((0, len(obstacles)))
        for i in range(len(agents)):
            adj_vec = []
            obstacle: Obstacle
            for obstacle in obstacles:
                adj_vec.append(obstacle.in_entity_radius(agents[i, :2], r=r))
            adj_matrix = np.vstack((adj_matrix, np.array(adj_vec)))
        return adj_matrix

    def _get_delta_adjacency_matrix(self, agents: np.ndarray,
                                    delta_agents: list, r: float) -> np.ndarray:
        adj_matrix = np.array([]).reshape((0, len(delta_agents)))
        for i in range(len(agents)):
            adj_vec = []
            delta_agent: Shepherd
            for delta_agent in delta_agents:
                adj_vec.append(
                    delta_agent.in_entity_radius(agents[i, :2], r=r))
            adj_matrix = np.vstack((adj_matrix, np.array(adj_vec)))
        return adj_matrix

    def _get_a_ij(self, q_i, q_js, range):
        r_alpha = MathUtils.sigma_norm([range])
        return MathUtils.bump_function(
            MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def _get_n_ij(self, q_i, q_js):
        return MathUtils.sigma_norm_grad(q_js - q_i)

    # Experimental function
    # Pairwise potential
    def _pairwise_potentia_vec(self, qi: np.ndarray, qj: np.ndarray, d: float):
        pw_sum = np.zeros(2)
        for i in range(qj.shape[0]):
            qji = qj[i, :] - qi
            pw = (np.linalg.norm(qj[i, :] - qi) -
                  d) ** 2 * utils.unit_vector(qji)
            pw_sum += pw
        return pw_sum

    def _pairwise_potential_mag(self, qi: np.ndarray, qj: np.ndarray, d: float):
        pw_sum = 0
        for i in range(qj.shape[0]):
            pw = (np.linalg.norm(qj[i, :] - qi) - d) ** 2
            pw_sum += pw
        return pw_sum

    # Elastic potential
    def _potential(self, qi: np.ndarray, qj: np.ndarray):
        p_sum = np.zeros(2)
        for i in range(qj.shape[0]):
            p = qi - qj[i, :]
            p_sum += p
        return p_sum

    def _inv_potential(self, qi: np.ndarray, qj: np.ndarray):
        p_sum = np.zeros(2)
        for i in range(qj.shape[0]):
            p = qj[i, :] - qi
            p_sum += p
        return p_sum

    def _density(self, si: np.ndarray, sj: np.ndarray, k: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * np.linalg.norm(sij))) * (sij / np.linalg.norm(sij))
            w_sum += w
        return w_sum
