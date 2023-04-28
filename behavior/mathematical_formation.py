# !/usr/bin/python3

import pygame
import numpy as np
from app import utils
from behavior.behavior import Behavior
from behavior.mathematical_flock import MathUtils
from entity.herd import Herd
from entity.shepherd import Shepherd

from scipy.spatial import ConvexHull


class MathematicalFormation(Behavior):
    Cs = 35
    Cr = 0

    def __init__(self):
        super().__init__()

        self._herds = []
        self._shepherds = []

        self._stop = True

        # Stuff to display
        self._plot_enforced_agent = False
        self._herd_mean = np.zeros(2)
        self._herd_radius = 150

        self._vis_boundary = False
        self._boundary_agents = []

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def update(self, *args, **kwargs):
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

        self._herd_mean = np.sum(
            herd_states[:, :2], axis=0) / herd_states.shape[0]

        d_to_herd_mean = np.linalg.norm(
            herd_states[:, :2] - self._herd_mean, axis=1)
        self._herd_radius = np.max(d_to_herd_mean)

        shepherd: Shepherd
        shepherd_states = np.array([]).reshape((0, 4))
        for shepherd in self._shepherds:
            # Grab and put all poses into a matrix
            shepherd_states = np.vstack(
                (shepherd_states, np.hstack((shepherd.pose, shepherd.velocity))))

        delta_adjacency_matrix = self._get_delta_adjacency_matrix(herd_states,
                                                                  self._shepherds,
                                                                  r=200)
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(shepherd_states,
                                                                  r=200)

        # hull = ConvexHull(herd_states[:,:2])
        # self._boundary_agents = herd_states[hull.vertices, :2]
        # self._vis_boundary = True

        p = np.zeros((len(self._shepherds), 2))
        shepherd: Shepherd
        for idx, shepherd in enumerate(self._shepherds):
            di = shepherd_states[idx, :2]
            d_dot_i = shepherd_states[idx, 2:]

            neighbor_herd_idxs = delta_adjacency_matrix[idx]
            ps = np.zeros(2)
            if sum(neighbor_herd_idxs) > 0:
                sj = herd_states[neighbor_herd_idxs, :2]
                # s_dot_j = herd_stat
                # es[neighbor_herd_idxs, 2:]
                stabilised_range = 110
                pairwise_density = self._pairwise_density(
                    si=di, sj=sj, k=0.125, r=stabilised_range)

                # print(f"Robot {idx}: {np.linalg.norm(ps)}")
                ps = -MathematicalFormation.Cs * \
                    (1/np.sqrt(np.linalg.norm(pairwise_density)**2)) * \
                    utils.unit_vector(pairwise_density)

            po = np.zeros(2)
            # Enforce virual beta agent
            agent_spacing = 200
            neighbor_shepherd_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_shepherd_idxs) > 0:
                dj = shepherd_states[neighbor_shepherd_idxs, :2]
                d_dot_j = shepherd_states[neighbor_shepherd_idxs, 2:]

                # print(50/np.linalg.norm(di - self._herd_mean))
                alpha_grad = self._gradient_term(
                    c=2 * np.sqrt(10/np.linalg.norm(di - self._herd_mean)), qi=di, qj=dj,
                    r=agent_spacing,
                    d=agent_spacing)
                po = alpha_grad

            # Enforce virual beta agent
            enforced_beta = np.zeros(2)
            enforce_agent = self._induce_enforced_beta_agent(
                yk=self._herd_mean,
                Rk=self._herd_radius,
                di=di,
                di_dot=d_dot_i)
            bt_di = enforce_agent[:2]
            bt_di_dot = enforce_agent[2:]

            beta_grad = self._gradient_term(
                c=2 * np.sqrt(20), qi=di, qj=bt_di,
                r=30,
                d=30)

            beta_consensus = self._velocity_consensus_term(
                c=2 * np.sqrt(20),
                qi=di, qj=bt_di,
                pi=d_dot_i, pj=bt_di_dot,
                r=30)

            enforced_beta = beta_grad + beta_consensus

            pg = np.zeros(2)

            # Total density p
            p[idx] = ps + po + enforced_beta

            if np.linalg.norm(p[idx]) > 15:
                p[idx] = 15 * utils.unit_vector(p[idx])

        self._plot_enforced_agent = False

        if not self._stop:
            # Control for the agent
            qdot = p
            shepherd_states[:, 2:] = qdot
            pdot = shepherd_states[:, 2:]
            shepherd_states[:, :2] += pdot * 0.15

            shepherd: Shepherd
            for idx, shepherd in enumerate(self._shepherds):
                # self._remain_in_screen(herd)
                shepherd._plot_velocity = True
                shepherd.velocity = shepherd_states[idx, 2:]
                shepherd.pose = shepherd_states[idx, :2]
                shepherd._rotate_image(shepherd.velocity)
                shepherd.reset_steering()

    def display(self, screen: pygame.Surface):
        if self._plot_enforced_agent:
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=tuple(self._herd_mean),
                radius=self._herd_radius, width=2)

        if self._vis_boundary and self._boundary_agents is not None:
            for idx in range(self._boundary_agents.shape[0] - 1):
                pygame.draw.line(screen, pygame.Color("white"), tuple(
                    self._boundary_agents[idx, :]), tuple(self._boundary_agents[idx + 1, :]))
            pygame.draw.line(screen, pygame.Color("white"), tuple(
                self._boundary_agents[self._boundary_agents.shape[0] - 1, :]),
                tuple(self._boundary_agents[0, :]))

    # Common functions with MathematicalFlock
    def _gradient_term(self, c: float, qi: np.ndarray, qj: np.ndarray,
                       r: float, d: float):
        # n_ij = utils.unit_vector(qj - qi)
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

    def _get_alpha_adjacency_matrix(self, agent_states: np.ndarray,
                                    r: float) -> np.ndarray:
        adj_matrix = np.array(
            [np.linalg.norm(agent_states[i, :2]-agent_states[:, :2], axis=-1) <= r
             for i in range(len(agent_states))])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _get_delta_adjacency_matrix(self, agents: np.ndarray,
                                    delta_agents: list, r: float) -> np.ndarray:
        adj_matrix = np.array([]).reshape((0, len(agents))).astype(np.bool8)
        delta_agent: Shepherd
        for delta_agent in delta_agents:
            adj_vec = []
            for i in range(len(agents)):
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

    def _pairwise_density(self, si: np.ndarray, sj: np.ndarray, k: float, r: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * (np.linalg.norm(sij) - r))) * \
                utils.unit_vector(sij)
            w_sum += w
        return w_sum

    def _induce_enforced_beta_agent(self, yk: np.ndarray, Rk: float,
                                    di: np.ndarray, di_dot: np.ndarray):
        yk = np.array(yk).reshape((2, 1))
        Rk = float(Rk)
        di = di.reshape((2, 1))
        di_dot = di_dot.reshape((2, 1))

        mu = Rk / np.linalg.norm(di - yk)
        ak = (di - yk)/np.linalg.norm(di - yk)
        P = np.eye(2) - ak @ ak.transpose()

        qik = mu * di + (1 - mu) * yk
        pik = mu * P @ di_dot
        return np.hstack((qik.transpose(), pik.transpose())).reshape(4,)
