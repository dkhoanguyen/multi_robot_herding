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

    def update(self, state: np.ndarray,
               other_states: np.ndarray):
        pose = state[:2]
        velocity = state[2:4]

        if np.linalg.norm(self._target_pos - pose) <= 10:
            return np.zeros(2)

        # Nominal Controller
        u_nom = self._controller.step(pose, self._target_pos)
        if np.linalg.norm(u_nom) > self._max_u:
            u_nom = self._max_u * unit_vector(u_nom)
        u = u_nom
        # CBF Constraints
        ri = 15
        rj = 15

        # timestep
        dt = 0.05

        xi = pose
        xj = other_states[0, :2]
        vi = unit_vector(u_nom) * 10
        # vi = velocity
        vj = other_states[0, 2:4]

        plane = ORCA.construct_orca_plane(xi=xi, xj=xj, vi=vi, vj=vj,
                                          ri=ri, rj=rj,
                                          weight=0.5,
                                          buffered_r=0.0,
                                          time_horizon=2.0)
        print(np.linalg.norm(xi - xj))
        if plane is not None:
            # print("yes")
            # planes = [plane]
            # A_orca, b_ocra = ORCA.build_constraint(planes, vi, -100)
            # A_vmax, b_vmax = VelocityConstraint.build_constraint(vi, 10.0, 1.0)
            # A = np.vstack((A_orca, A_vmax))
            # b = np.vstack((b_ocra, b_vmax))
            # P = np.identity(3)
            # P[2, 2] = 1.0
            # q = np.append(-2 * u_nom, 0)
            # UB = np.array([self._max_u, self._max_u, np.inf])
            # LB = np.array([-self._max_u, -self._max_u, -np.inf])

            # u = solve_qp(P, q, G=A, h=b,lb=LB, ub=UB, solver="osqp")  # osqp or cvxopt
            # u = u[:2]

            planes = [plane]
            A_orca, b_ocra = ORCA.build_constraint(planes, vi, 1.2)
            A_vmax, b_vmax = VelocityConstraint.build_constraint(vi, 10.0, 4.0)
            A = np.vstack((A_orca, ))
            b = np.vstack((b_ocra, ))
            P = np.identity(3) * 0.5
            P[2, 2] = 500.0
            q = np.append(-2 * u_nom, -1000.0)
            UB = np.array([self._max_u, self._max_u, np.inf])
            LB = np.array([-self._max_u, -self._max_u, -np.inf])

            u = solve_qp(P, q, G=A, h=b,lb=LB, ub=UB,
                         solver="osqp")  # osqp or cvxopt

            # print(u[2])
            u = u[:2]
            if u is None:
                # print("None")
                u = u_nom

        if np.linalg.norm(u) > self._max_u:
            u = self._max_u * unit_vector(u)
        return u

    def display(self, screen: pygame.Surface):
        return super().display(screen)

    def transition(self, state: np.ndarray,
                   other_states: np.ndarray,
                   herd_states: np.ndarray,
                   consensus_states: dict):

        return True
