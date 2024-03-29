# !/usr/bin/python3

import pygame
import numpy as np
from collections import deque
from enum import IntEnum

from multi_robot_herding.utils import utils
from multi_robot_herding.behavior.behavior import Behavior
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.behavior.mathematical_flock import MathUtils
from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.shepherd import Shepherd

class BearingFormation(Behavior):
    class State(IntEnum):
        IDLE = 0
        APPROACH = 1
        SURROUND = 2
        APPREHEND = 3
        MOVE = 4

    def __init__(self, sensing_range: float,
                 agent_spacing: float,
                 scaled_agent_spacing: float):
        super().__init__()

        self._herds = []
        self._shepherds = []

        self._cs = 100

        self._sensing_range = sensing_range
        self._default_agent_spacing = agent_spacing
        self._scaled_agent_spacing = scaled_agent_spacing

        self._time_horizon = 300
        self._total_ps_over_time = deque(maxlen=self._time_horizon)

        self._formation_generated = False
        self._p_star = np.empty((0, 2))
        self._plotting_p_star = np.empty((0, 4))
        self._g_star = np.empty((0, 2))
        self._trigger_formation = False
        self._formation = np.empty((0, 2))

        self._centroid_p_star = np.zeros((1, 2))

        self._error_pose_ij = None
        self._init_error_pose = False

        # Formation morphing state machine
        self._state = BearingFormation.State.IDLE
        self._states = []
        self._init_states = False
        self._total_bearing_error = 0
        self._shrink = False

        # Stuff to display
        self._plot_enforced_agent = False
        self._herd_mean = np.zeros(2)
        self._herd_radius = 0

        self._vis_boundary = False
        self._boundary_agents = []

        self._plot_formation = False

        self._font = pygame.font.SysFont("comicsans", 16)
        self._text_list = []

        self._formation_cluster = []

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def update(self, *args, **kwargs):
        events = self._get_events(args)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and not self._trigger_formation:
                    self._trigger_formation = True
                if event.key == pygame.K_UP and self._trigger_formation:
                    self._trigger_formation = False

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
        shepherd_states = np.array([]).reshape((0, 6))
        for shepherd in self._shepherds:
            shepherd_states = np.vstack(
                (shepherd_states, np.hstack((shepherd.pose, shepherd.velocity, shepherd.acceleration))))

        if not self._init_states:
            self._states = np.ones(
                shepherd_states.shape[0], dtype=np.int8) * BearingFormation.State.IDLE
            self._init_states = True

        # Consensus
        for idx, _ in enumerate(self._states):
            if self._states[idx] == BearingFormation.State.IDLE:
                self._states[idx] = BearingFormation.State.APPROACH
            elif self._states[idx] == BearingFormation.State.APPROACH and \
                    np.linalg.norm(shepherd_states[idx, :2] - self._herd_mean) <= (self._sensing_range + self._herd_radius):
                self._states[idx] = BearingFormation.State.SURROUND
            elif self._states[idx] == BearingFormation.State.SURROUND and self._formation_stable():
                self._states[idx] = BearingFormation.State.APPREHEND
            elif self._states[idx] == BearingFormation.State.APPREHEND and self._formation_stable():
                self._states[idx] = BearingFormation.State.MOVE

        approach_u = self._approach_herd(
            shepherd_states=shepherd_states, herd_mean=self._herd_mean, logic_states=self._states)
        surround_u = self._herd_surrounding(herd_states=herd_states,
                                            shepherd_states=shepherd_states,
                                            logic_states=self._states)

        # If consensus reaches for formatio generation
        if np.sum(self._states) == BearingFormation.State.MOVE * self._states.size:
            if not self._formation_generated:
                ret, p_star = self._generate_formation(shepherd_states[:, :2])
                if ret:
                    self._g_star = p_star
                    self._formation_generated = True

        formation_u = self._maneuver_formation(
            shepherd_states=shepherd_states,
            formation=self._g_star, logic_states=self._states)

        qdot = approach_u + surround_u + formation_u
        shepherd_states[:, 2:4] = qdot
        pdot = shepherd_states[:, 2:4]
        shepherd_states[:, :2] += pdot * 0.15

        shepherd: Shepherd
        self._text_list.clear()
        for idx, shepherd in enumerate(self._shepherds):
            shepherd._plot_velocity = True
            shepherd.velocity = shepherd_states[idx, 2:4]
            shepherd.pose = shepherd_states[idx, :2]
            shepherd._rotate_image(shepherd.velocity)

            text = self._font.render(str(idx), 1, pygame.Color("white"))
            self._text_list.append((text, shepherd.pose - np.array([20, 20])))

    def display(self, screen: pygame.Surface):
        if self._plot_enforced_agent:
            pygame.draw.circle(screen, pygame.Color(
                'white'), center=tuple(self._herd_mean),
                radius=self._herd_radius, width=2)

        pygame.draw.circle(screen, pygame.Color(
            'white'), center=tuple(self._centroid_p_star),
            radius=20, width=2)

        if self._plot_formation:
            for i in range(self._plotting_p_star.shape[0]):
                pygame.draw.line(screen, pygame.Color("white"), tuple(self._plotting_p_star[i, :2]),
                                 tuple(self._plotting_p_star[i, 2:]))

        for text in self._text_list:
            screen.blit(text[0], tuple(text[1]))

    def _find_herd(self):
        # TODO
        pass

    def _approach_herd(self, shepherd_states: np.ndarray,
                       herd_mean: np.ndarray,
                       logic_states: np.ndarray) -> np.ndarray:
        u = np.zeros((shepherd_states.shape[0], 2))
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            shepherd_states,
            r=self._default_agent_spacing)

        for idx in range(shepherd_states.shape[0]):
            if logic_states[idx] != BearingFormation.State.APPROACH:
                continue

            di = shepherd_states[idx, :2]
            d_dot_i = shepherd_states[idx, 2:4]

            agent_spacing = self._scaled_agent_spacing * self._default_agent_spacing
            po = np.zeros(2)
            neighbor_shepherd_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_shepherd_idxs) > 0:
                dj = shepherd_states[neighbor_shepherd_idxs, :2]

                po = self._collision_avoidance_term(
                    gain=0.8 * MathematicalFlock.C2_alpha,
                    qi=di, qj=dj,
                    r=agent_spacing)

            # Move toward herd mean
            p_gamma = self._calc_group_objective_control(
                target=herd_mean,
                c1=MathematicalFlock.C1_gamma,
                c2=MathematicalFlock.C2_gamma,
                qi=di, pi=d_dot_i)

            u[idx] = po + p_gamma
        return u

    def _herd_surrounding(self, herd_states: np.ndarray,
                          shepherd_states: np.ndarray,
                          logic_states: np.ndarray) -> np.ndarray:
        u = np.zeros((shepherd_states.shape[0], 2))
        delta_adjacency_matrix = self._get_delta_adjacency_matrix(
            herd_states,
            self._shepherds,
            r=self._sensing_range)

        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(
            shepherd_states,
            r=self._default_agent_spacing)

        total_ps = 0

        for idx in range(shepherd_states.shape[0]):
            if logic_states[idx] != BearingFormation.State.SURROUND and \
                    logic_states[idx] != BearingFormation.State.APPREHEND:
                continue

            di = shepherd_states[idx, :2]

            neighbor_herd_idxs = delta_adjacency_matrix[idx]
            ps = np.zeros(2)
            if sum(neighbor_herd_idxs) > 0:
                sj = herd_states[neighbor_herd_idxs, :2]

                stabilised_range = self._sensing_range
                if logic_states[idx] == BearingFormation.State.APPREHEND:
                    stabilised_range = 110

                ps = self._edge_following(
                    si=di, sj=sj, k=0.095,
                    stabilised_range=stabilised_range,
                    encircle_gain=self._cs)

                total_ps += np.linalg.norm(ps)

            po = np.zeros(2)
            agent_spacing = self._default_agent_spacing
            neighbor_shepherd_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_shepherd_idxs) > 0:
                dj = shepherd_states[neighbor_shepherd_idxs, :2]

                po = self._collision_avoidance_term(
                    gain=0.8 * MathematicalFlock.C2_alpha,
                    qi=di, qj=dj,
                    r=agent_spacing)

            u[idx] = ps + po

            if np.linalg.norm(u[idx]) > 15:
                u[idx] = 15 * utils.unit_vector(u[idx])
        self._total_ps_over_time.append(total_ps)
        return u

    def _follow_cursor(self, shepherd_states: np.ndarray):
        target = np.array(pygame.mouse.get_pos())
        u = np.zeros((shepherd_states.shape[0], 2))
        for idx in range(shepherd_states.shape[0]):
            di = shepherd_states[idx, :2]
            d_dot_i = shepherd_states[idx, 2:4]

            u[idx] = self._calc_group_objective_control(
                target=target,
                c1=1.5 * MathematicalFlock.C1_gamma,
                c2=1.5 * MathematicalFlock.C2_gamma,
                qi=di, pi=d_dot_i)
        return u

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
                            formation: np.ndarray,
                            logic_states: np.ndarray):
        u = np.zeros((shepherd_states.shape[0], 2))
        if formation.shape[0] == 0:
            return u
        p_star = self._p_star
        p = shepherd_states[:, :2]

        start_end = self._calc_start_end(
            np.ones((len(self._shepherds), len(self._shepherds))), global_rigid=True)
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
            total_bearing_error += np.round(np.linalg.norm(g - g_star), 4)
            pair = np.array([shepherd_states[pair_ij[0], :2],
                             shepherd_states[pair_ij[1], :2]])
            self._plotting_p_star = np.vstack(
                (self._plotting_p_star, pair.reshape((1, 4))))
            if start_end[0, idx] in leader_idx:
                continue
            Pg_ij = self._calc_diagPg(g_star, e_norm_star)
            all_total_Pg_ij[:, :, pair_ij[0]] += Pg_ij
            all_Pg_ij[:, :, pair_ij[0], pair_ij[1]] = Pg_ij
        # print(total_bearing_error)

        kp = 6.5
        kd = 0.05
        kv = 1
        ki = 0.82
        alpha = -0.025
        desired_length = 150
        desired_scale = 130
        dt = 0.2
        for i in range(shepherd_states.shape[0]):
            if logic_states[i] != BearingFormation.State.MOVE:
                continue
            if i in leader_idx:
                # ratio = desired_length / \
                #     np.linalg.norm(shepherd_states[i, :2] - centroid_p_star)
                # ratio = np.linalg.norm(shepherd_states[i, :2] - centroid_p_star) - desired_length
                # ratio = desired_scale/scale
                # # ratio = 0
                # # for j in range(len(leader_idx)):
                # #     if i == j:
                # #         continue
                # #     ratio += desired_length / \
                # #         np.linalg.norm(shepherd_states[i, :2] - shepherd_states[j, :2])
                # u[i] = alpha * np.round(1 - ratio, 3) * (
                #     shepherd_states[i, :2] - centroid_p_star)
                # u[i] = alpha * np.round(ratio, 1) * (
                #     shepherd_states[i, :2] - centroid_p_star)

                # # target = np.array([150, 350])
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
        return u

    # Inter-robot Interaction Control
    def _collision_avoidance_term(self, gain: float, qi: np.ndarray,
                                  qj: np.ndarray, r: float):
        n_ij = utils.MathUtils.sigma_norm_grad(qj - qi)
        return gain * np.sum(utils.MathUtils.phi_alpha(
            utils.MathUtils.sigma_norm(qj-qi),
            r=r,
            d=r)*n_ij, axis=0)

    def _velocity_consensus_term(self, gain: float, qi: np.ndarray, qj: np.ndarray,
                                 pi: np.ndarray, pj: np.ndarray, r: float):
        def calc_a_ij(q_i, q_js, range):
            r_alpha = utils.MathUtils.sigma_norm([range])
            return utils.MathUtils.bump_function(
                utils.MathUtils.sigma_norm(q_js-q_i)/r_alpha)

        a_ij = calc_a_ij(qi, qj, r)
        return gain * np.sum(a_ij*(pj-pi), axis=0)

    def _get_alpha_adjacency_matrix(self, agent_states: np.ndarray,
                                    r: float) -> np.ndarray:
        adj_matrix = np.array(
            [np.linalg.norm(agent_states[i, :2]-agent_states[:, :2], axis=-1) <= r
             for i in range(len(agent_states))])
        np.fill_diagonal(adj_matrix, False)
        return adj_matrix

    def _local_crowd_horizon(self, si: np.ndarray, sj: np.ndarray, k: float, r: float):
        w_sum = np.zeros(2).astype(np.float64)
        for i in range(sj.shape[0]):
            sij = si - sj[i, :]
            w = (1/(1 + k * (np.linalg.norm(sij) - r))) * \
                utils.unit_vector(sij)
            w_sum += w
        return w_sum

    def _edge_following(self, si: np.ndarray, sj: np.ndarray,
                        k: float, stabilised_range: float,
                        encircle_gain: float):
        # TODO: might need to rework this potential function
        local_crowd_horizon = self._local_crowd_horizon(
            si=si, sj=sj, k=k, r=stabilised_range)
        return -encircle_gain * \
            (1/np.linalg.norm(local_crowd_horizon)) * \
            utils.unit_vector(local_crowd_horizon)

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

    def _calc_group_objective_control(self, target: np.ndarray,
                                      c1: float, c2: float,
                                      qi: np.ndarray, pi: np.ndarray):
        def calc_group_objective_term(
                c1: float, c2: float,
                pos: np.ndarray, qi: np.ndarray, pi: np.ndarray):
            return -c1 * MathUtils.sigma_1(qi - pos) - c2 * (pi)
        u_gamma = calc_group_objective_term(
            c1=c1,
            c2=c2,
            pos=target,
            qi=qi,
            pi=pi)
        return u_gamma

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

    def _formation_stable(self):
        if len(self._total_ps_over_time) != self._time_horizon:
            return False
        y = np.array(self._total_ps_over_time)
        x = np.linspace(0, self._time_horizon,
                        self._time_horizon, endpoint=False)
        coeff, err, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        if np.sqrt(err) >= 20:
            return False
        poly = np.poly1d(coeff)
        polyder = np.polyder(poly)
        cond = np.abs(np.round(float(polyder.coef[0]), 2))
        return not bool(cond)
