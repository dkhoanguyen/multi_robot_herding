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


class MathematicalFormation(Behavior):
    def __init__(self):
        self._radius = 10
        self._k = 0.01
        self._kd = 1
        self._l = 12.5
        self._herd_mean = np.zeros(2)
        self._herds = []
        self._shepherds = []

    def set_herd_mean(self, mean: np.ndarray):
        self._herd_mean = mean

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def update(self, dt):
        s_i = self._herd_mean
        ideal_phi = self._ideal_phi(s_i)

        # Step 1: Controller for p_dot
        p = self._offset(s_i, ideal_phi, l=12.5)
        p_dot = -self._k * p

        # Step 2: Ideal heading phi_star and velocity v_star
        phi_star = self._phi_star(s_i)
        v_star = self._qx(ideal_phi) * p_dot

        # Step 3: Find delta_j_star from v_star
        delta = self._delta_from_v(v_star)
        delta_j_star = self._distribute_deltas(delta)

        # Step 4: Desired d_j_star
        d_j_stars = self._calc_d_j_star(delta_j_star, phi_star)

        # Step 5: Radial controller
        # r_dot = self._radius

        # Step 6: Tracking controller
        d_dot_js = self._tracking_controller(d_j_stars)

        shepherd: Shepherd
        for idx, shepherd in enumerate(self._shepherds):
            shepherd.velocity = d_dot_js[idx]
            shepherd.pose = shepherd.pose + d_dot_js[idx] * 0.1

            shepherd._rotate_image(shepherd.velocity)
            shepherd.reset_steering()

    def _qx(self, phi: float):
        return np.array([math.cos(phi), math.sin(phi)])

    def _offset(self, s_i: np.ndarray, phi: float, l: float):
        return np.add(s_i, l * self._qx(phi))

    def _ideal_phi(self, s_i: np.ndarray):
        return math.atan2(s_i[1], s_i[0]) + np.pi

    def _phi_star(self, s_i: np.ndarray):
        return self._ideal_phi(s_i)

    def _delta_from_v(self, v: np.ndarray):
        i2 = 10000
        search_space = np.linspace(0, 2 * np.pi, i2)
        i1 = 0
        mn = np.ones((1, 2)) * np.inf
        target = np.linalg.norm(v) * self._radius ** 2

        m = len(self._shepherds)
        while (i1 < i2):
            mid = (i2 + i1) // 2
            if (mid == i1 or mid == i2):
                break
            curr = self._fx(search_space[mid], m)
            if (np.abs(curr - target) < np.abs(mn[0, 0] - target)):
                mn[0, 0] = curr
                mn[0, 1] = search_space[mid]
            if (curr < target):
                i2 = mid
            else:
                i1 = mid
        return (np.array(mn).reshape(2)[1])

    def _distribute_deltas(self, delta):
        delta_j_star = []
        m = len(self._shepherds)
        for i in range(1, m + 1):
            m = (2 * i - m - 1) / (2 * m - 2)
            d_i = delta * m
            delta_j_star.append(d_i)
        return np.array(delta_j_star)

    def _fx(self, d, m):
        return (np.sin((m * d) / (2 - 2 * m)) / np.sin(d / (2 - 2 * m)))

    def _calc_d_j_star(self, delta_j_star, phi_star):
        ideal_pos = []
        s = self._herd_mean
        for i in range(len(self._shepherds)):
            angle = delta_j_star[i] + phi_star + np.pi
            d_j_star = s + self._radius * \
                np.array([math.cos(angle), math.sin(angle)]).reshape(2)
            ideal_pos.append(d_j_star)

        return np.array(ideal_pos)

    # def radial_controller(self):
    #     r_nought = 15
    #     r_dot = (r_nought - self._radius)

    #     s_bar = self._herd_mean
    #     s_i_dots = [self.sheep_repulsion(i) for i in range(num_sheep)]
    #     s_bar_dot = sum(s_i_dots) / num_sheep

    #     for i in range(num_sheep):
    #         s_i = self.coords[i]

    #         a = (s_i - s_bar)
    #         b = (s_i_dots[i] - s_bar_dot)
    #         c = a[0] * b[0] + a[1] * b[1]
    #         r_dot += c / len(self._shepherds)

    #     return (r_dot)

    def _tracking_controller(self, d_j_stars):
        d_dot_js = []
        for i in range(len(self._shepherds)):
            d_dot_js.append(-self._kd *
                            (np.add(d_j_stars[i], - self._shepherds[i].pose)))
        return np.array(d_dot_js)
