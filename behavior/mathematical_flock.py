# !/usr/bin/python3
import math
import time

import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd


class MathematicalFlock(Behavior):
    C1_alpha = 3
    C2_alpha = 2 * np.sqrt(C1_alpha)
    C1_gamma = 5
    C2_gamma = 0.2 * np.sqrt(C1_gamma)
    RANGE = 100
    DISTANCE = 100

    class MathUtils():

        EPSILON = 0.1
        H = 0.2
        A, B = 5, 5
        C = np.abs(A-B)/np.sqrt(4*A*B)  # phi

        R = 30
        D = 30

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
            return ((a + b) * MathematicalFlock.MathUtils.sigma_1(z + c) + (a - b)) / 2

        @staticmethod
        def phi_alpha(z, r=R, d=D):
            r_alpha = MathematicalFlock.MathUtils.sigma_norm([r])
            d_alpha = MathematicalFlock.MathUtils.sigma_norm([d])
            return MathematicalFlock.MathUtils.bump_function(z/r_alpha) * MathematicalFlock.MathUtils.phi(z-d_alpha)

        @staticmethod
        def normalise(v, pre_computed=None):
            n = pre_computed if pre_computed is not None else math.sqrt(
                v[0]**2 + v[1]**2)
            if n < 1e-13:
                return np.zeros(2)
            else:
                return np.array(v) / n

    def __init__(self, range=RANGE, distance=DISTANCE):
        self._herds = []
        self._shepherds = []

        self._range = range
        self._distance = distance

        self._sample_t = time.time()

    def add_herd(self, herd: Herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd: Shepherd):
        self._shepherds.append(shepherd)

    def update(self, dt: float):
        # self._flocking(dt)
        self._pause_agents = np.random.random_integers(low=0, high=len(
            self._herds) - 1, size=(round(len(self._herds)/2),))
        herd: Herd
        for idx, herd in enumerate(self._herds):
            if idx in self._pause_agents:
                self._wander(herd)
                self._separate(herd, herd.personal_space)
                self._old_remain_in_screen(herd)
                herd.update()
            else:
                herd.velocity = np.zeros(2)
                herd.pose = herd.pose + herd.velocity
                herd.update()

    # Old basic herd behaviors
    def _wander(self, herd: Herd):
        WANDER_DIST = 4.0
        WANDER_RADIUS = 3.0
        WANDER_ANGLE = 1.0  # degrees

        rands = 2 * np.random.rand(len(self._herds)) - 1
        cos = np.cos([herd.wandering_angle for herd in self._herds])
        sin = np.sin([herd.wandering_angle for herd in self._herds])

        another_herd: Herd
        for i, another_herd in enumerate(self._herds):
            if herd == another_herd:
                nvel = MathematicalFlock.MathUtils.normalise(herd.velocity)
                # calculate circle center
                circle_center = nvel * WANDER_DIST
                # calculate displacement force
                c, s = cos[i], sin[i]
                displacement = np.dot(
                    np.array([[c, -s], [s, c]]), nvel * WANDER_RADIUS)
                herd.steer(circle_center + displacement, alt_max=10)
                herd.wandering_angle += WANDER_ANGLE * rands[i]

    def _old_remain_in_screen(self, herd: Herd):
        if herd.pose[0] > params.SCREEN_WIDTH - params.BOX_MARGIN:
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

        if herd.pose[0] > params.SCREEN_WIDTH:
            herd.pose[0] = params.SCREEN_WIDTH
        if herd.pose[0] < 0:
            herd.pose[0] = 0
        if herd.pose[1] < 0:
            herd.pose[1] = 0
        if herd.pose[1] > params.SCREEN_HEIGHT:
            herd.pose[1] = params.SCREEN_HEIGHT

    def _separate(self, herd: Herd, distance: float):
        number_of_neighbors = 0
        force = np.zeros(2)
        other_boid: Herd
        for other_boid in self._herds:
            if herd == other_boid:
                continue
            if utils.dist2(herd.pose, other_boid.pose) < distance ** 2:
                force -= other_boid.pose - herd.pose
                number_of_neighbors += 1
        if number_of_neighbors:
            force /= number_of_neighbors
        herd.steer(utils.normalize(force) * 10.0,
                   alt_max=params.BOID_MAX_FORCE)

    def _flocking(self, dt: float):
        herd: Herd
        states = np.array([]).reshape((0, 4))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            states = np.vstack((states, np.hstack((herd.pose, herd.velocity))))

        u = np.zeros((len(self._herds), 2))
        alpha_adjacency_matrix = self._get_adjacency_matrix(states)
        mouse_pose = pygame.mouse.get_pos()

        for idx, herd in enumerate(self._herds):
            qi = states[idx, :2]
            pi = states[idx, 2:]

            # Alpha agent
            u_alpha = 0
            neighbor_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_idxs) > 1:
                qj = states[neighbor_idxs, :2]
                pj = states[neighbor_idxs, 2:]

                alpha_grad = self._gradient_term(
                    c=MathematicalFlock.C2_alpha,
                    qi=qi,
                    qj=qj)

                alpha_consensus = self._velocity_consensus_term(
                    c=MathematicalFlock.C2_alpha,
                    qi=qi, qj=qj,
                    pi=pi, pj=pj)
                u_alpha = alpha_grad + alpha_consensus

            # Beta agent
            u_beta = 0

            # Gamma agent
            u_gamma = self._group_objective_term(
                c1=MathematicalFlock.C1_gamma,
                c2=MathematicalFlock.C2_gamma,
                pos=np.array(mouse_pose),
                qi=qi,
                pi=pi)

            # Ultimate flocking model
            u[idx] = u_alpha + u_beta + u_gamma

        qdot = u
        states[:, 2:] += qdot * 0.1
        pdot = states[:, 2:]
        states[:, :2] += pdot * 0.1

        herd: Herd
        for idx, herd in enumerate(self._herds):
            # herd.state = states[idx, :]
            herd.velocity = states[idx, 2:]
            herd.pose = states[idx, :2]
            herd._rotate_image(herd.velocity)

    def _gradient_term(self, c: float, qi: np.ndarray, qj: np.ndarray):
        n_ij = self._get_n_ij(qi, qj)
        return c * np.sum(MathematicalFlock.MathUtils.phi_alpha(
            MathematicalFlock.MathUtils.sigma_norm(qj-qi),
            r=MathematicalFlock.RANGE,
            d=MathematicalFlock.DISTANCE)*n_ij, axis=0)

    def _velocity_consensus_term(self, c: float, qi: np.ndarray,
                                 qj: np.ndarray, pi: np.ndarray, pj: np.ndarray):
        # Velocity consensus term
        a_ij = self._get_a_ij(qi, qj, self._range)
        return c * np.sum(a_ij*(pj-pi), axis=0)

    def _group_objective_term(self, c1: float, c2: float,
                              pos: np.ndarray, qi: np.ndarray, pi: np.ndarray):
        # Group objective term
        return -c1 * MathematicalFlock.MathUtils.sigma_1(qi - pos) - c2 * (pi)

    def _get_adjacency_matrix(self, agents: np.ndarray, r=RANGE):
        return np.array([np.linalg.norm(agents[i, :2]-agents[:, :2], axis=-1) <= r for i in range(len(agents))])

    def _get_a_ij(self, q_i, q_js, range):
        r_alpha = MathematicalFlock.MathUtils.sigma_norm([range])
        return MathematicalFlock.MathUtils.bump_function(
            MathematicalFlock.MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def _get_n_ij(self, q_i, q_js):
        return MathematicalFlock.MathUtils.sigma_norm_grad(q_js - q_i)