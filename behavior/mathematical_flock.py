# !/usr/bin/python3

import numpy as np
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd


class MathUtils():

    EPSILON = 0.1
    H = 0.2
    A, B = 5, 5
    C = np.abs(A-B)/np.sqrt(4*A*B)  # phi

    R = 12
    D = 10

    @staticmethod
    def sigma_1(z):
        return z / np.sqrt(1 + z**2)

    @staticmethod
    def sigma_norm(z, e=EPSILON):
        return (np.sqrt(1 + e * np.linalg.norm(z, axis=-1, keepdims=True)**2) - 1) / e

    @staticmethod
    def sigma_norm_grad(z):
        return z/np.sqrt(1 + MathUtils.EPSILON * np.linalg.norm(z, axis=-1, keepdims=True)**2)

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


C1_alpha = 3
C2_alpha = 2 * np.sqrt(C1_alpha)
C1_gamma = 5
C2_gamma = 0.2 * np.sqrt(C1_gamma)


class MathematicalFlock(Behavior):
    def __init__(self):
        self._herds = []
        self._shepherds = []

    def add_herd(self, herd: Herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd: Shepherd):
        self._shepherds.append(shepherd)

    def update(self, dt):
        distance = 12
        poses = []
        velocities = []

        u = np.zeros((len(self._herds), 2))
        herd: Herd
        for herd in self._herds:
            # Grab and put all poses into a matrix
            poses.append(herd.pose)
            velocities.append(herd.velocity)
        poses = np.array(poses)
        velocities = np.array(velocities)

        for idx, herd in enumerate(self._herds):
            pose_i = poses[idx]
            velocity_i = poses[idx]

            if sum(np.linalg.norm(poses[idx]-poses, axis=-1) <= distance) > 1:
                pose_j = poses[idx]
                velocity_j = velocities[idx]
                n_ij = self._get_n_ij(pose_i, pose_j)

                term_1 = C2_alpha * \
                    np.sum(MathUtils.phi_alpha(
                        MathUtils.sigma_norm(pose_j - pose_i))*n_ij, axis=0)
                a_ij = self._get_a_ij(pose_i, pose_j, distance)
                term_2 = C2_alpha * \
                    np.sum(a_ij*(velocity_j-velocity_i), axis=0)
                u_alpha = term_1 + term_2
            else:
                u_alpha = 0
            u_gamma = -C1_gamma * \
                MathUtils.sigma_1(
                    pose_i - [500, 400] - C2_gamma*(velocity_i-0))
            u[idx] = u_alpha + u_gamma

            # Update control signal
            herd.velocity += u[idx] * dt
            herd.pose += herd.velocity * dt
            herd._rotate_image(herd.velocity)

            # herd.update()

    def _get_a_ij(self, q_i, q_js, range):
        r_alpha = MathUtils.sigma_norm([range])
        return MathUtils.bump_function(MathUtils.sigma_norm(q_js-q_i)/r_alpha)

    def _get_n_ij(q_i, q_js):
        return MathUtils.sigma_norm_grad(q_js - q_i)
