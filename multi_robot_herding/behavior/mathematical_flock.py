# !/usr/bin/python3
import math
import networkx as nx

import pygame
import numpy as np
from multi_robot_herding.utils import params, utils
from multi_robot_herding.behavior.behavior import Behavior
from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.shepherd import Shepherd
from multi_robot_herding.entity.obstacle import Obstacle


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

        self._enable_flocking = True

        # For control
        self._mass = 0
        self._flocking_condition = 0
        self._dt = 0.2
        self._dt_sqr = 0.1

        self._boundary = {
            'x_min': 300,
            'x_max': 600,
            'y_min': 300,
            'y_max': 500,
        }

        # For visualization
        self._contour_agents = []

        # Clusters
        self._total_clusters = 0
        self._clusters = []
        self._plot_cluster = False

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

    def update(self, *args, **kwargs):
        events = self._get_events(args)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and self._enable_flocking:
                    self._enable_flocking = False
                if event.key == pygame.K_UP and not self._enable_flocking:
                    self._enable_flocking = True

        if self._enable_flocking:
            self._flocking_condition = 1
        else:
            self._flocking_condition = 0

        # self._flocking_condition = 1

        herd: Herd
        herd_states = np.array([]).reshape((0, 4))

        for herd in self._herds:
            # Grab and put all poses into a matrix
            herd_states = np.vstack(
                (herd_states, np.hstack((herd.pose, herd.velocity))))

        shepherd: Shepherd
        shepherd_states = np.array([]).reshape((0, 4))
        for shepherd in self._shepherds:
            # Grab and put all poses into a matrix
            shepherd_states = np.vstack(
                (shepherd_states, np.hstack((shepherd.pose, shepherd.velocity))))

        local_clustering = self._local_clustering(
            herd_states, shepherd_states, k=0.5)
        global_clustering = self._global_clustering(
            herd_states, shepherd_states)

        flocking = self._flocking(herd_states, shepherd_states)

        remain_in_bound_u = self._calc_remain_in_boundary_control(
            herd_states, self._boundary, k=5.0)
        
        # # Density
        # herd_density = self._herd_density(herd_states=herd_states)
        # density_mag = np.linalg.norm(herd_density,axis=1)
        # density_mag = density_mag[density_mag > 0.09]
        # print(density_mag)

        qdot = (1 - self._flocking_condition) * local_clustering + \
            flocking + self._flocking_condition * global_clustering + \
            (1 - self._flocking_condition) * remain_in_bound_u
        herd_states[:, 2:4] += qdot * self._dt_sqr
        pdot = herd_states[:, 2:4]
        herd_states[:, :2] += pdot * self._dt

        herd: Herd
        for idx, herd in enumerate(self._herds):
            # Scale velocity
            if np.linalg.norm(herd_states[idx, 2:4]) > herd._max_v:
                herd_states[idx, 2:4] = herd._max_v * \
                    utils.unit_vector(herd_states[idx, 2:4])

            herd.velocity = herd_states[idx, 2:4]
            herd.pose = herd_states[idx, :2]
            herd._rotate_image(herd.velocity)

            # herd._force = 2 * herd_density[idx,:]
            # herd._plot_force = True

    def display(self, screen: pygame.Surface):
        if self._clusters is not None and len(self._clusters) > 0 and self._plot_cluster:
            for cluster in self._clusters:
                for edge in cluster:
                    pygame.draw.line(screen, pygame.Color("white"), tuple(edge[0, :2]),
                                     tuple(edge[1, :2]))

    # Mathematical model of flocking
    def _flocking(self, herd_states: np.ndarray,
                  shepherd_states: np.ndarray) -> np.ndarray:
        u = np.zeros((herd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states,
                                                                  r=self._sensing_range)
        beta_adjacency_matrix = self._get_beta_adjacency_matrix(herd_states,
                                                                self._obstacles,
                                                                r=self._sensing_range)
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(herd_states,
                                                                  self._shepherds,
                                                                  r=self._sensing_range)

        for idx in range(herd_states.shape[0]):
            # Flocking terms
            neighbor_idxs = alpha_adjacency_matrix[idx]
            u_alpha = self._calc_flocking_control(
                idx=idx, neighbors_idxs=neighbor_idxs,
                herd_states=herd_states)

            # Obstacle avoidance term
            obstacle_idxs = beta_adjacency_matrix[idx]
            u_beta = self._calc_obstacle_avoidance_control(
                idx=idx, obstacle_idxs=obstacle_idxs,
                beta_adj_matrix=beta_adjacency_matrix,
                herd_states=herd_states)

            # Shepherd
            shepherd_idxs = delta_adjacency_matrix[idx]
            u_delta = self._calc_shepherd_interaction_control(
                idx=idx, shepherd_idxs=shepherd_idxs,
                delta_adj_matrix=delta_adjacency_matrix,
                herd_states=herd_states)

            # Ultimate flocking model
            u[idx] = u_alpha + u_beta + u_delta
        return u

    def _herd_density(self, herd_states: np.ndarray):
        herd_densities = np.zeros((herd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(herd_states,
                                                                  r=self._sensing_range)
        for idx in range(herd_states.shape[0]):
            # Density
            neighbor_idxs = alpha_adjacency_matrix[idx]
            density = self._calc_density(
                idx=idx, neighbors_idxs=neighbor_idxs,
                herd_states=herd_states)
            herd_densities[idx] = density
        return herd_densities

    def _global_clustering(self, herd_states: np.ndarray,
                           shepherd_states: np.ndarray) -> np.ndarray:
        u = np.zeros((herd_states.shape[0], 2))
        for idx in range(herd_states.shape[0]):
            qi = herd_states[idx, :2]
            pi = herd_states[idx, 2:4]

            # Group consensus term
            target = self._consensus_pose
            if self._follow_cursor:
                target = np.array(pygame.mouse.get_pos())

            u_gamma = self._calc_group_objective_control(target=target,
                                                         qi=qi, pi=pi)
            u[idx] = u_gamma
        return u

    def _local_clustering(self, herd_states: np.ndarray,
                          shepherd_states: np.ndarray,
                          k: float) -> np.ndarray:
        adj_matrix = self._get_alpha_adjacency_matrix(
            herd_states=herd_states, r=self._sensing_range * 1.)
        graph = nx.Graph(adj_matrix)

        clusters_idxs = [graph.subgraph(c).copy()
                         for c in nx.connected_components(graph)]

        self._total_clusters = len(clusters_idxs)
        self._clusters = []

        clusters = []
        cluster_indx_list = []
        for cluster_idxs in clusters_idxs:
            cluster = []
            cluster_indx = []

            for cluster_edge in cluster_idxs.edges:
                cluster.append(herd_states[cluster_edge, :])

            self._clusters.append(cluster)

            cluster_nodes = []
            if len(cluster_idxs.nodes) == 1:
                continue

            for cluster_node in cluster_idxs.nodes:
                cluster_nodes.append(herd_states[cluster_node, :])
                cluster_indx.append(cluster_node)
            clusters.append(cluster_nodes)
            cluster_indx_list.append(cluster_indx)

        # Perform local flocking with local cluster
        all_gamma = np.zeros((herd_states.shape[0], 2))
        for cluster_indx, cluster in enumerate(clusters):
            if len(clusters) == 1:
                continue

            local_cluster_states = np.empty((0, 4))
            for cluster_node in cluster:
                local_cluster_states = np.vstack(
                    (local_cluster_states, cluster_node))

            for idx in range(local_cluster_states.shape[0]):
                qi = local_cluster_states[idx, :2]
                pi = local_cluster_states[idx, 2:4]

                this_indx = cluster_indx_list[cluster_indx][idx]

                # Group consensus term
                cluster_mean = np.sum(
                    local_cluster_states[:, :2], axis=0) / local_cluster_states.shape[0]

                target = cluster_mean
                u_gamma = k * self._calc_group_objective_control(
                    target=target,
                    qi=qi, pi=pi)
                all_gamma[this_indx, :] = u_gamma
        return all_gamma

    def _calc_remain_in_boundary_control(self, herd_states: Herd, boundary: np.ndarray, k: float):
        x_min = boundary['x_min']
        x_max = boundary['x_max']
        y_min = boundary['y_min']
        y_max = boundary['y_max']

        u = np.zeros_like(herd_states[:, 2:4])
        for idx in range(herd_states.shape[0]):
            qi = herd_states[idx, :2]

            if qi[0] < x_min:
                u[idx, :] += k * np.array([1.0, 0.0])

            elif qi[0] > x_max:
                u[idx, :] += k * np.array([-1.0, 0.0])

            if qi[1] < y_min:
                u[idx, :] += k * np.array([0.0, 1.0])

            elif qi[1] > y_max:
                u[idx, :] += k * np.array([0.0, -1.0])
        return u

    def _calc_flocking_control(self, idx: int,
                               neighbors_idxs: np.ndarray,
                               herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        pi = herd_states[idx, 2:4]
        u_alpha = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = herd_states[neighbors_idxs, :2]
            pj = herd_states[neighbors_idxs, 2:4]

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
        return u_alpha

    def _calc_density(self, idx: int,
                      neighbors_idxs: np.ndarray,
                      herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        density = np.zeros(2)
        if sum(neighbors_idxs) > 0:
            qj = herd_states[neighbors_idxs, :2]
            density = self._density(si=qi, sj=qj, k=0.375)
        return density

    def _calc_obstacle_avoidance_control(self, idx: int,
                                         obstacle_idxs: np.ndarray,
                                         beta_adj_matrix: np.ndarray,
                                         herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        pi = herd_states[idx, 2:4]
        u_beta = np.zeros(2)
        if sum(obstacle_idxs) > 0:
            # Create beta agent
            obs_in_radius = np.where(beta_adj_matrix[idx] > 0)
            beta_agents = np.array([]).reshape((0, 4))
            for obs_idx in obs_in_radius[0]:
                beta_agent = self._obstacles[obs_idx].induce_beta_agent(
                    self._herds[idx])
                beta_agents = np.vstack((beta_agents, beta_agent))

            qik = beta_agents[:, :2]
            pik = beta_agents[:, 2:4]
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
        return u_beta

    def _calc_group_objective_control(self, target: np.ndarray,
                                      qi: np.ndarray, pi: np.ndarray):
        u_gamma = self._group_objective_term(
            c1=MathematicalFlock.C1_gamma,
            c2=MathematicalFlock.C2_gamma,
            pos=target,
            qi=qi,
            pi=pi)
        return u_gamma

    def _calc_shepherd_interaction_control(self, idx: int,
                                           shepherd_idxs: np.ndarray,
                                           delta_adj_matrix: np.ndarray,
                                           herd_states: np.ndarray):
        qi = herd_states[idx, :2]
        pi = herd_states[idx, 2:4]
        u_delta = np.zeros(2)
        if sum(shepherd_idxs) > 0:
            # Create delta_agent
            delta_in_radius = np.where(delta_adj_matrix[idx] > 0)
            delta_agents = np.array([]).reshape((0, 4))
            for del_idx in delta_in_radius[0]:
                delta_agent = self._shepherds[del_idx].induce_delta_agent(
                    self._herds[idx])
                delta_agents = np.vstack((delta_agents, delta_agent))

            qid = delta_agents[:, :2]
            pid = delta_agents[:, 2:4]
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
            si=qi, r=self._danger_range, k=200000)

        return u_delta

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

    # Clustering and graph theory
    def _get_herd_laplacian_matrix(self, herd_states: np.ndarray):
        # First get adjaceny matrix
        adj_matrix = self._get_alpha_adjacency_matrix(
            herd_states=herd_states, r=self._sensing_range).astype(np.float64)
        # Then get the degree matrix
        deg_matrix = np.zeros_like(adj_matrix)
        diag_sum = np.sum(adj_matrix, axis=1)
        np.fill_diagonal(deg_matrix, diag_sum)
        laplacian_matrix = deg_matrix - adj_matrix
        return laplacian_matrix
