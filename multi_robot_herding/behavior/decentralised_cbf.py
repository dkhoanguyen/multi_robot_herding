# !/usr/bin/python3

import time
import math
import pygame
import numpy as np
from collections import deque

from multi_robot_herding.utils.utils import *
from multi_robot_herding.common.decentralised_behavior import DecentralisedBehavior
from multi_robot_herding.behavior.controller import SimplePController

# QP solver
from qpsolvers import solve_qp
from multi_robot_herding.behavior.constraint import *


class DecentralisedCBF(DecentralisedBehavior):
    def __init__(self, target_pos: np.ndarray,
                 controller_gain: np.ndarray):
        super().__init__()

        self._target_pos = target_pos
        self._controller = SimplePController(p_gain=controller_gain[0])

        self._max_u = 5
        self._max_v = 2

        self._pose = np.zeros(2)
        self._u = np.zeros(2)

    def update(self, state: np.ndarray,
               other_states: np.ndarray,
               animals_states: np.ndarray):
        pose = state[:2]
        self._pose = pose
        velocity = state[2:4]

        # Nominal Controller
        u_nom = self._controller.step(pose, self._target_pos)
        if np.linalg.norm(u_nom) > self._max_u:
            u_nom = self._max_u * unit_vector(u_nom)
        u = u_nom
        # CBF Constraints
        ri = 30
        rj = np.ones(other_states[:,:2].shape[0]) * 30
        weight = np.ones(other_states[:,:2].shape[0]) * 0.5

        # timestep
        dt = 0.1

        xi = pose
        xj = other_states[:,:2]
        vi = unit_vector(u_nom) * 10
        # vi = velocity
        vj = other_states[:,2:4]

        # print(np.linalg.norm(xi - xj[0,:2]))

        planes = ORCA.construct_orca_planes(xi=xi, xj=xj, vi=vi, vj=vj,
                                            ri=ri, rj=rj,
                                            weight=weight,
                                            buffered_r=0.0,
                                            time_horizon=2.0)

        A = np.empty((0, 4))
        b = np.empty((0, 1))

        A_dmin, b_dmin = MinDistance.build_constraint(
            xi=xi, xj=xj, vi=velocity, vj=vj,
            ai=self._max_u, aj=0,
            d=60.0, gamma=1.0)

        A = np.vstack((A, A_dmin))
        b = np.vstack((b, b_dmin))

        A_dmax, b_dmax = MaxDistance.build_constraint(
            xi=xi, xj=xj, vi=velocity, vj=vj,
            ai=self._max_u, aj=self._max_u,
            d=300.0, gamma=1.0)

        A = np.vstack((A, A_dmax))
        b = np.vstack((b, b_dmax))

        if len(planes) > 0:
            A_orca, b_ocra = ORCA.build_constraint(planes, vi,
                                                   self._max_u, self._max_u,
                                                   1.0)
            A = np.vstack((A, A_orca,))
            b = np.vstack((b, b_ocra,))

        P = np.identity(4) * 0.5
        p_omega = 100.0
        omega_0 = 1.0
        P[2, 2] = p_omega
        P[3, 3] = 1.0
        q = -2 * np.array([u_nom[0], u_nom[1], omega_0 * p_omega, 0.0])
        UB = np.array([self._max_u, self._max_u, np.inf, np.inf])
        LB = np.array([-self._max_u, -self._max_u, -np.inf, -np.inf])

        u = solve_qp(P, q, G=A, h=b, lb=LB, ub=UB,
                     solver="cvxopt")  # osqp or cvxopt
        if u is None:
            u = np.zeros(2)
        else:
            u = u[:2]

        if np.linalg.norm(u) > self._max_u:
            u = self._max_u * unit_vector(u)
        self._u = u

        return u

    def display(self, screen: pygame.Surface):
        pygame.draw.line(
            screen, pygame.Color("yellow"),
            tuple(self._pose), tuple(self._pose + 5 * (self._u)))
        return super().display(screen)

    def transition(self, state: np.ndarray,
                   other_states: np.ndarray,
                   herd_states: np.ndarray,
                   consensus_states: dict):

        return True
