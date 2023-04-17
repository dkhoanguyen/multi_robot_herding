# !/usr/bin/python3
import math
import time

from spatialmath.base import *

import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd
from entity.obstacle import Obstacle


class SpiralMotion(Behavior):
    def __init__(self):
        super().__init__()
        self._herds = []
        self._shepherds = []
        self._shepherd = None

        self._current_idx = 0

    def add_herd(self, herd):
        self._herds.append(herd)

    def add_shepherd(self, shepherd):
        self._shepherds.append(shepherd)

    def add_single_shepherd(self, shepherd: Shepherd):
        self._shepherd = shepherd

    def update(self, *args, **kwargs):
        mouse_pose = pygame.mouse.get_pos()
        self._herd_mean = np.array(mouse_pose)

        events = self._get_events(args)
        next_spiral = self._create_spiral_motion(
            self._shepherd.pose, np.array(mouse_pose), 50)

        # self._shepherd.move_to_pose(next_spiral)
        self._shepherd.follow_velocity(np.array([0,0]))
        # self._shepherd.update()

    def _create_spiral_motion(self, current_pose: np.ndarray,
                              center: np.ndarray, radius: float):
        theta = np.linspace(0., 2 * np.pi, 360)
        spiral_poses = center.reshape(
            2, 1) + radius * np.array([np.cos(theta), np.sin(theta)])
        pose_center = current_pose - center
        angle_pose_center = math.atan2(pose_center[1], pose_center[0])
        
        idx = (np.abs(theta - angle_pose_center)).argmin()
        if np.abs(np.linalg.norm(pose_center) - radius) > 1:
            self._current_idx = idx
        else:
            self._current_idx = self._current_idx + 1

        if self._current_idx >= len(spiral_poses[0,:]) - 1:
            self._current_idx = 0
        return spiral_poses[:, self._current_idx]
