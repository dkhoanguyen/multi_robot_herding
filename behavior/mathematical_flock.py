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


class MathematicalFlock(Behavior):
    C1_alpha = 3
    C2_alpha = 2 * np.sqrt(C1_alpha)
    C1_beta = 20
    C2_beta = 2 * np.sqrt(C1_beta)
    C1_gamma = 5
    C2_gamma = 0.2 * np.sqrt(C1_gamma)

    ALPHA_RANGE = 40
    ALPHA_DISTANCE = 40
    BETA_RANGE = 30
    BETA_DISTANCE = 30

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

    def __init__(self, follow_cursor: bool,
                 initial_consensus: np.ndarray):
        self._herds = []
        self._shepherds = []
        self._obstacles = []

        self._sample_t = 0
        self._pause_agents = np.zeros(1)

        self._follow_cursor = follow_cursor
        self._consensus_pose = initial_consensus

        # Fleeing behavior

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
        agent_states = np.array([]).reshape((0, 2))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            agent_states = np.vstack(
                (agent_states, herd.pose))
        return np.sum(agent_states, axis=0) / agent_states.shape[0]

    def update(self, dt: float):
        self._flocking(dt)
        # if time.time() - self._sample_t > 3.0:
        #     self._pause_agents = np.random.random_integers(low=0, high=len(
        #         self._herds) - 1, size=(round(len(self._herds)/2),))
        #     self._sample_t = time.time()

        # herd: Herd
        # for idx, herd in enumerate(self._herds):
        #     if idx in self._pause_agents:
        #         herd.speed = 1
        #         self._wander(herd)
        #         self._separate(herd, herd.personal_space)
        #         self._old_remain_in_screen(herd)
        #         herd.update()
        #     else:
        #         herd.speed = 0.0
        #         herd.update()

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

    def _remain_in_screen(self, herd: Herd):
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

    def _flee(self, shepherd: Shepherd, herd: Herd):
        pred_pose = shepherd.pose
        pred_vel = shepherd.velocity
        t = int(utils.norm(pred_pose - herd.pose) / params.BOID_MAX_SPEED)
        pred_future_pose = pred_pose + t * pred_vel

        too_close = utils.dist2(herd.pose, pred_future_pose) < 200**2
        if too_close:
            steering = (utils.normalize(herd.pose - pred_future_pose) *
                        params.BOID_MAX_SPEED -
                        herd.velocity)
            herd.steer(steering, alt_max=params.BOID_MAX_FORCE)

    # Mathematical model of flocking
    def _flocking(self, dt):
        herd: Herd
        agent_states = np.array([]).reshape((0, 4))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            agent_states = np.vstack(
                (agent_states, np.hstack((herd.pose, herd.velocity))))

        u = np.zeros((len(self._herds), 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(agent_states,
                                                                  r=MathematicalFlock.ALPHA_RANGE)
        beta_adjacency_matrix = self._get_beta_adjacency_matrix(agent_states,
                                                                self._obstacles,
                                                                r=100)
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(agent_states,
                                                                  self._shepherds,
                                                                  r=2000)
        mouse_pose = pygame.mouse.get_pos()

        for idx, herd in enumerate(self._herds):
            qi = agent_states[idx, :2]
            pi = agent_states[idx, 2:]

            # Alpha agent
            u_alpha = 0
            neighbor_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_idxs) > 1:
                qj = agent_states[neighbor_idxs, :2]
                pj = agent_states[neighbor_idxs, 2:]

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
            u_beta = 0
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
            # if self._follow_cursor:
            #     target = np.array(mouse_pose)

            # shepherd_induced_consensus = self._shepherds[0].induce_consesus_point(
            # )
            # herd_mean = self.get_herd_mean()
            # # target = shepherd_induced_consensus
            # target = np.array(mouse_pose)

            u_gamma = self._group_objective_term(
                c1=MathematicalFlock.C1_gamma,
                c2=MathematicalFlock.C2_gamma,
                pos=target,
                qi=qi,
                pi=pi)

            # Delta agent (shepherd)
            u_delta = 0
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
                
            u_gamma = 0
            # Ultimate flocking model
            u[idx] = u_alpha + u_beta + u_gamma + u_delta

        qdot = u
        agent_states[:, 2:] += qdot * 0.1
        pdot = agent_states[:, 2:]
        agent_states[:, :2] += pdot * 0.2

        herd: Herd
        for idx, herd in enumerate(self._herds):
            for shepherd in self._shepherds:
                self._flee(shepherd, herd)
            # herd.state = agent_states[idx, :]
            herd.velocity = agent_states[idx, 2:] + herd._steering
            herd.pose = agent_states[idx, :2]
            herd._rotate_image(herd.velocity)
            herd.reset_steering()

    def _gradient_term(self, c: float, qi: np.ndarray, qj: np.ndarray,
                       r: float, d: float):
        n_ij = self._get_n_ij(qi, qj)
        return c * np.sum(MathematicalFlock.MathUtils.phi_alpha(
            MathematicalFlock.MathUtils.sigma_norm(qj-qi),
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
        return -c1 * MathematicalFlock.MathUtils.sigma_1(qi - pos) - c2 * (pi)

    def _get_alpha_adjacency_matrix(self, agents: np.ndarray, r: float) -> np.ndarray:
        return np.array([np.linalg.norm(agents[i, :2]-agents[:, :2], axis=-1) <= r
                         for i in range(len(agents))])

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
        r_alpha = MathematicalFlock.MathUtils.sigma_norm([range])
        return MathematicalFlock.MathUtils.bump_function(
            MathematicalFlock.MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def _get_n_ij(self, q_i, q_js):
        return MathematicalFlock.MathUtils.sigma_norm_grad(q_js - q_i)
