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
        self._consensus_pose = initial_consensus

        self._start_time = time.time()
        self._stop = False

        # For control
        self._mass = 0

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
        if herd.pose[0] > params.SCREEN_WIDTH - 600:
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
    def _flocking(self, *args, **kwargs):
        events = self._get_events(args)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and not self._stop:
                    self._stop = True
                if event.key == pygame.K_UP and self._stop:
                    self._stop = False

        herd: Herd
        agent_states = np.array([]).reshape((0, 4))
        for herd in self._herds:
            # Grab and put all poses into a matrix
            agent_states = np.vstack(
                (agent_states, np.hstack((herd.pose, herd.velocity))))

        # ch = ConvexHull(agent_states[:, :2])

        u = np.zeros((len(self._herds), 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(agent_states,
                                                                  r=self._sensing_range)
        lower_tril = np.tril(alpha_adjacency_matrix, -1)
        epsilon = sum(sum(lower_tril))

        beta_adjacency_matrix = self._get_beta_adjacency_matrix(agent_states,
                                                                self._obstacles,
                                                                r=self._sensing_range)
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(agent_states,
                                                                  self._shepherds,
                                                                  r=self._danger_range)
        force_mag_list = np.empty((0, 2))

        total_pairwise_sum = 0
        for idx, herd in enumerate(self._herds):
            # Plotting config
            herd._plot_force = False
            herd._plot_force_mag = False

            qi = agent_states[idx, :2]
            pi = agent_states[idx, 2:]

            # Alpha agent
            u_alpha = 0
            neighbor_idxs = alpha_adjacency_matrix[idx]

            pairwise_potential = self._pairwise_potential(
                qi, agent_states[:, :2],
                MathematicalFlock.ALPHA_DISTANCE)
            total_pairwise_sum += pairwise_potential

            if sum(neighbor_idxs) > 0:
                qj = agent_states[neighbor_idxs, :2]
                pj = agent_states[neighbor_idxs, 2:]

                pw_qj = qj
                shepherd: Shepherd
                for shepherd in self._shepherds:
                    pw_qj = np.vstack((pw_qj, shepherd.pose.reshape(1, 2)))

                pairwise_potential = (1/(1 + sum(neighbor_idxs))) * self._pairwise_potential(
                    qi, qj, MathematicalFlock.ALPHA_DISTANCE)

                reg_potential = (1/(1 + sum(neighbor_idxs))) * \
                    self._potential(qi, pw_qj)
                herd._force = reg_potential * 0.02
                herd._plot_force = True

                herd._plot_force_mag = False
                # print(f"Robot {idx} norm: {np.linalg.norm(reg_potential)}")
                # print(f"Robot {idx} vec: {reg_potential}")
                # print("==")

                herd._force_mag = pairwise_potential / 200

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

                # # Density function with neighbors
                # herd._force = self._get_agent_density_vector(qi, qj, 0.375)
                # herd._force_mag = 35 * \
                #     self._get_agent_density_mag(qi, qj, 0.375)
            # else:
            #     herd._force = np.zeros(2)
            #     herd._force_mag = 0
            # force_mag_list = np.vstack(
            #     (force_mag_list, np.array([herd._force_mag, idx])))

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

            # if time.time() - self._start_time < 0:
            # u_gamma = 0
            # u_delta = 0
            # Ultimate flocking model
            u[idx] = u_alpha + u_beta + u_gamma + u_delta

        total_pairwise_sum = total_pairwise_sum/(1 + epsilon)
        # print(total_pairwise_sum)
        # with open('data/gathered_flock.txt', 'a') as f:
        #     f.write(str(total_pairwise_sum) + '\n')

        # Control for the agent
        qdot = u
        agent_states[:, 2:] += qdot * 0.1
        pdot = agent_states[:, 2:]
        agent_states[:, :2] += pdot * 0.2

        herd: Herd
        for idx, herd in enumerate(self._herds):
            # for shepherd in self._shepherds:
            #     self._flee(shepherd, herd)
            self._remain_in_screen(herd)
            herd.velocity = agent_states[idx, 2:] + herd._steering
            herd.pose = agent_states[idx, :2]
            herd._rotate_image(herd.velocity)
            herd.reset_steering()

        # shepherd: Shepherd
        # for shepherd in self._shepherds:
        #     reg_potential = (1/(1 + len(self._herds))) * \
        #         self._inv_potential(shepherd.pose, agent_states[:, :2])
        #     shepherd._force = reg_potential * 0.005
        #     shepherd._plot_force = True
        #     shepherd.velocity = reg_potential * 0.008

        #     if self._stop:
        #         return
            
        #     shepherd.pose = shepherd.pose + shepherd.velocity
        #     shepherd._rotate_image(shepherd.velocity)
        #     shepherd.reset_steering()
        #     # shepherd.follow_mouse()
        #     # shepherd.update()

        # self._vis_entity.boundaries = agent_states[ch.vertices, :2]

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
        adj_matrix = np.array([np.linalg.norm(agents[i, :2]-agents[:, :2], axis=-1) <= r
                               for i in range(len(agents))])
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
        r_alpha = MathematicalFlock.MathUtils.sigma_norm([range])
        return MathematicalFlock.MathUtils.bump_function(
            MathematicalFlock.MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def _get_n_ij(self, q_i, q_js):
        return MathematicalFlock.MathUtils.sigma_norm_grad(q_js - q_i)

    # Experimental function
    def _get_all_agent_density_vectors(self, s_i: np.ndarray, s_js: np.ndarray, k: float):
        p = []
        for s_j in s_js:
            wij = 1/(1 + k * np.linalg.norm(s_i - s_j)) * \
                utils.unit_vector(s_i - s_j)
            if not sum(np.isnan(wij)):
                p.append(wij)
        return np.array(p)

    def _get_agent_density_vector(self, s_i: np.ndarray, s_js: np.ndarray, k: float):
        p = []
        for s_j in s_js:
            wij = 1/(1 + k * np.linalg.norm(s_i - s_j)) * \
                utils.unit_vector(s_i - s_j)
            if not sum(np.isnan(wij)):
                p.append(wij)
        p = np.sum(np.array(p), axis=0)
        return p

    def _get_agent_density_mag(self, s_i: np.ndarray, s_js: np.ndarray, k: float):
        p = []
        for s_j in s_js:
            wij = 1/(1 + k * np.linalg.norm(s_i - s_j)) * \
                utils.unit_vector(s_i - s_j)
            if not sum(np.isnan(wij)):
                p.append(np.linalg.norm(wij))
        p = sum(p)
        return p

    def _pairwise_potential(self, qi: np.ndarray, qj: np.ndarray, d: float):
        pw_sum = 0
        for i in range(qj.shape[0]):
            pw = (np.linalg.norm(qj[i, :] - qi) - d) ** 2
            pw_sum += pw
        return pw_sum

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
