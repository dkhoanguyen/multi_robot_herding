# !/usr/bin/python3

import pygame
import numpy as np
from collections import deque

from multi_robot_herding.utils import utils
from multi_robot_herding.behavior.behavior import Behavior
from multi_robot_herding.behavior.mathematical_flock import MathematicalFlock
from multi_robot_herding.behavior.mathematical_flock import MathUtils
from multi_robot_herding.entity.herd import Herd
from multi_robot_herding.entity.shepherd import Shepherd


class MathematicalFormation(Behavior):
    Cs = 100
    Cr = 0

    def __init__(self, sensing_range: float,
                 agent_spacing: float,
                 scaled_agent_spacing: float):
        super().__init__()

        self._sensing_range = sensing_range
        self._default_agent_spacing = agent_spacing
        self._scaled_agent_spacing = scaled_agent_spacing
        self._shrink_spacing = 150

        self._herds = []
        self._shepherds = []

        self._move_toward_herds = 1

        self._shrink = 0
        self._time_horizon = 300
        self._total_ps_over_time = deque(maxlen=self._time_horizon)

        self._follow_cursor = False

        # Formation
        self._enforce_formation = False
        self._formation_calc = False
        self._formation = None

        # Stuff to display
        self._plot_enforced_agent = False
        self._herd_mean = np.zeros(2)
        self._herd_radius = 0

        self._vis_boundary = False
        self._boundary_agents = []

        self._font = pygame.font.SysFont("comicsans", 16)
        self._text_list = []

        self._formation_cluster = []

        # # create the csv writer
        # self._file = open(f"shrink_condition_{time.time()}.csv","w")
        # self._writer = csv.writer(self._file)
        # self._write_header = True

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def update(self, *args, **kwargs):
        events = self._get_events(args)
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN and not self._enforce_formation:
                    self._enforce_formation = True
                if event.key == pygame.K_UP and self._enforce_formation:
                    self._enforce_formation = False

                if event.key == pygame.K_LEFT and not self._follow_cursor:
                    self._follow_cursor = True
                if event.key == pygame.K_RIGHT and self._follow_cursor:
                    self._follow_cursor = False

        # if self._write_header:
        #     self._write_header = False
        #     header = ["shrink_condition"]
        #     self._writer.writerow(header)

        encircle_gain = MathematicalFormation.Cs
        if self._follow_cursor:
            encircle_gain = 0

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
                                                                  r=self._sensing_range)
        alpha_adjacency_matrix = self._get_alpha_adjacency_matrix(shepherd_states,
                                                                  r=self._default_agent_spacing)

        # Enforce formation
        if self._enforce_formation:
            if not self._formation_calc:
                self._formation = self._generate_minimal_rigid_formation(
                    shepherd_states=shepherd_states)
                self._formation_calc = True

        p = np.zeros((len(self._shepherds), 2))
        shepherd: Shepherd

        total_ps = 0
        self._formation_cluster = []
        for idx, shepherd in enumerate(self._shepherds):
            di = shepherd_states[idx, :2]
            d_dot_i = shepherd_states[idx, 2:4]

            approach_herd = 1
            if np.linalg.norm(di - self._herd_mean) <= (self._sensing_range + self._herd_radius):
                approach_herd = 0

            neighbor_herd_idxs = delta_adjacency_matrix[idx]
            ps = np.zeros(2)

            if sum(neighbor_herd_idxs) > 0:
                sj = herd_states[neighbor_herd_idxs, :2]

                stabilised_range = self._sensing_range
                if self._shrink:
                    stabilised_range = 100
                local_crowd_horizon = self._local_crowd_horizon(
                    si=di, sj=sj, di=di, k=0.125, r=stabilised_range)

                ps = -encircle_gain * \
                    (1/np.linalg.norm(local_crowd_horizon)) * \
                    utils.unit_vector(local_crowd_horizon)
                total_ps += np.linalg.norm(ps)

            po = np.zeros(2)
            agent_spacing = self._default_agent_spacing
            if self._shrink:
                agent_spacing = self._shrink_spacing
            if approach_herd:
                agent_spacing = self._scaled_agent_spacing * self._default_agent_spacing

            neighbor_shepherd_idxs = alpha_adjacency_matrix[idx]
            if sum(neighbor_shepherd_idxs) > 0:
                dj = shepherd_states[neighbor_shepherd_idxs, :2]
                d_dot_j = shepherd_states[neighbor_shepherd_idxs, 2:4]

                alpha_grad = self._gradient_term(
                    c=MathematicalFlock.C2_alpha,
                    qi=di, qj=dj,
                    r=agent_spacing,
                    d=agent_spacing)
                po = alpha_grad

            # Move toward herd mean
            target = self._herd_mean
            p_gamma = self._calc_group_objective_control(
                target=target,
                c1=MathematicalFlock.C1_gamma,
                c2=MathematicalFlock.C2_gamma,
                qi=di, pi=d_dot_i)
            
            target = np.array(pygame.mouse.get_pos())
            p_follow_user = self._calc_group_objective_control(
                target=target,
                c1=1.5 * MathematicalFlock.C1_gamma,
                c2=1.5 * MathematicalFlock.C2_gamma,
                qi=di, pi=d_dot_i)

            # Total density p
            p[idx] = (1 - approach_herd) * ps + \
                po + \
                approach_herd * p_gamma + \
                int(self._follow_cursor) * p_follow_user

            if self._enforce_formation:
                pass
                # di = shepherd_states[idx, :2]
                # dist_vec = self._formation[idx]
                # po = np.zeros(2)
                # all_alpha = np.zeros(2)
                # for other_idx in range(len(dist_vec)):
                #     if dist_vec[other_idx] > 0.0:
                #         # print(other_idx)
                #         dj = shepherd_states[other_idx, :2]
                #         d_dot_j = shepherd_states[other_idx, 2:4]

                #         n_ij = self._get_n_ij(di, dj)
                #         grad = MathUtils.phi_alpha(
                #             MathUtils.sigma_norm(dj-di),
                #             r=np.round(dist_vec[other_idx]),
                #             d=np.round(dist_vec[other_idx]))*n_ij
                #         all_alpha += grad
                #         self._formation_cluster.append((di, dj))

                #         vel_consensus = self._velocity_consensus_term(
                #             c=20 * MathematicalFlock.C2_alpha,
                #             qi=di, qj=dj,
                #             pi=d_dot_i, pj=d_dot_j,
                #             r=np.round(dist_vec[other_idx]))

                # po = 10000 * MathematicalFlock.C2_alpha * all_alpha + vel_consensus
                # p[idx] = po + int(self._follow_cursor) * \
                #     p_follow_user + ps

            if not self._enforce_formation:
                if np.linalg.norm(p[idx]) > 15:
                    p[idx] = 15 * utils.unit_vector(p[idx])
            # else:
            #     if np.linalg.norm(p[idx]) > 15:
            #         p[idx] = 15 * utils.unit_vector(p[idx])

        self._plot_enforced_agent = False
        # self._writer.writerow([total_ps])
        self._total_ps_over_time.append(total_ps)
        # if self._shrink == 0 and not approach_herd:
        #     self._shrink = self._shrink_condition()
        
        if self._formation_stable():
            self._enforce_formation = True

        if not self._enforce_formation:
            qdot = p
            shepherd_states[:, 2:4] = qdot
            pdot = shepherd_states[:, 2:4]
            shepherd_states[:, :2] += pdot * 0.15
        else:
            qdot = p
            shepherd_states[:, 2:4] += qdot * 0.1
            pdot = shepherd_states[:, 2:4]
            shepherd_states[:, :2] += pdot * 0.05

        shepherd: Shepherd
        self._text_list.clear()
        for idx, shepherd in enumerate(self._shepherds):
            # self._remain_in_screen(herd)
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

        if self._vis_boundary and self._boundary_agents is not None:
            for idx in range(self._boundary_agents.shape[0] - 1):
                pygame.draw.line(screen, pygame.Color("white"), tuple(
                    self._boundary_agents[idx, :]), tuple(self._boundary_agents[idx + 1, :]))
            pygame.draw.line(screen, pygame.Color("white"), tuple(
                self._boundary_agents[self._boundary_agents.shape[0] - 1, :]),
                tuple(self._boundary_agents[0, :]))

        for text in self._text_list:
            self._screen.blit(text[0], tuple(text[1]))

        for line in self._formation_cluster:
            pygame.draw.line(screen, pygame.Color("white"), tuple(line[0]),
                             tuple(line[1]))

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

    def _calc_group_objective_control(self, target: np.ndarray,
                                      c1: float, c2: float,
                                      qi: np.ndarray, pi: np.ndarray):
        u_gamma = self._group_objective_term(
            c1=c1,
            c2=c2,
            pos=target,
            qi=qi,
            pi=pi)
        return u_gamma

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
    def _local_crowd_horizon(self, si: np.ndarray, sj: np.ndarray,
                             di: np.ndarray, k: float, r: float):
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

    def _shrink_condition(self):
        if len(self._total_ps_over_time) != self._time_horizon:
            return False
        y = np.array(self._total_ps_over_time)
        x = np.linspace(0, self._time_horizon,
                        self._time_horizon, endpoint=False)
        coeff, err, _, _, _ = np.polyfit(x, y, deg=1, full=True)
        poly = np.poly1d(coeff)
        polyder = np.polyder(poly)
        cond = np.abs(np.round(float(polyder.coef[0]), 3))
        return not bool(cond)
    
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

    def _generate_minimal_rigid_formation(self, shepherd_states: np.ndarray):
        # alpha_adj = self._get_alpha_adjacency_matrix(shepherd_states,
        #                                              r=self._shrink_spacing)
        # for row in alpha_adj:
        #     pair = np.where(row == True)[0]
        #     if len(pair) > 1:
        #         alpha_adj[pair[0], pair[1]] = True
        #         alpha_adj[pair[1], pair[0]] = True
        dist_mat = np.array(
            [np.linalg.norm(shepherd_states[i, :2]-shepherd_states[:, :2], axis=-1)
             for i in range(len(shepherd_states))])
        # formation = alpha_adj.astype(np.float64) * dist_mat
        formation = dist_mat
        return formation