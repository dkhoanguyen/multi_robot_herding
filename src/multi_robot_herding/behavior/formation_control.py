# !/usr/bin/python3
import math
import time

import pygame
import numpy as np
from utils import params, utils
from behavior.behavior import Behavior
from behavior.mathematical_flock import MathematicalFlock
from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Obstacle


class MathematicalFormation(Behavior):
    def __init__(self):
        super().__init__()
        self._radius = 200
        self._k = 0.01
        self._kd = 1
        self._l = 20

        self._herds = []
        self._shepherds = []

        self._flock_model = None

        self._herd_mean = np.zeros(2)
        self._herd_velocity = np.array([1, 0])
        self._herd_heading = 0

        self._start = False

    def set_herd_mean(self, mean: np.ndarray):
        self._herd_mean = mean

    def set_herd_velocity(self, velocity: np.ndarray):
        self._herd_velocity = velocity

    def set_herd_heading(self, heading: float):
        self._herd_heading = heading

    def set_flock_model(self, flock_model: MathematicalFlock):
        self._flock_model = flock_model

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def update(self, *args, **kwargs):
        mouse_pose = pygame.mouse.get_pos()
        self._herd_mean = np.array(mouse_pose)

        events = self._get_events(args)
        for event in events:
            if event.type == pygame.KEYDOWN and not self._start:
                self._start = True

        if self._start:
            s_i = self._herd_mean
            s_dot_i = self._herd_velocity
            ideal_phi = self._ideal_phi(s_dot_i)

            p = self._offset(s_i, ideal_phi, l=self._l)
            p_dot = -self._k * (p)

            thetas = self._custom_delta(ideal_phi)
            d_j_stars = self._custom_calc_dj_star(origin=s_i,
                                                  radius=self._radius,
                                                  thetas=thetas)
            shepherd: Shepherd
            for idx, shepherd in enumerate(self._shepherds):
                shepherd.move_to_pose(d_j_stars[idx])
                shepherd.update()

    def _qx(self, phi: float):
        return np.array([math.cos(phi), math.sin(phi)])

    def _offset(self, s_i: np.ndarray, phi: float, l: float):
        return np.add(s_i, l * self._qx(phi))

    def _ideal_phi(self, s_dot_i: np.ndarray):
        '''
        Get the current heading of velocity of the herd
        '''
        if np.array_equal(s_dot_i.reshape(2), np.zeros(2)):
            return self._herd_heading
        return math.atan2(s_dot_i[1], s_dot_i[0])

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

    def _custom_delta(self, ideal_angle: float):
        sliced_angle = []
        delta = 2 * np.pi / len(self._shepherds)
        angle = 0
        for i in range(len(self._shepherds)):
            if i == 0:
                angle = ideal_angle + (delta / 2)
            else:
                angle = sliced_angle[-1] + delta
                while angle > np.pi:
                    angle -= 2*np.pi
            sliced_angle.append(angle)
        return np.array(sliced_angle)

    def _custom_calc_dj_star(self, origin: np.ndarray,
                             radius: float,
                             thetas: np.ndarray):
        ideal_pos = []
        for theta in thetas:
            d_j_star = origin + radius * \
                np.array([math.cos(theta), math.sin(theta)]).reshape(2)
            ideal_pos.append(d_j_star)
        return np.array(ideal_pos)

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

    def _tracking_controller(self, d_j_stars):
        d_dot_js = []
        for i in range(len(self._shepherds)):
            d_dot_js.append(-self._kd *
                            (np.add(d_j_stars[i], - self._shepherds[i].pose)))
        return np.array(d_dot_js)
