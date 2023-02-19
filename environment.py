#!/usr/bin/python3

import numpy as np

import pygame

import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d

from animation_handle import AnimationHandle
from kinematic_model import KinematicModel, DifferentialDrive


class Environment(object):

    SCREEN_WIDTH = 700  
    SCREEN_HEIGHT = 700

    def __init__(self):
        pygame.init()
        self._pg_screen = pygame.display.set_mode((Environment.SCREEN_WIDTH, Environment.SCREEN_HEIGHT))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._pg_screen)

        self._pm_space = pymunk.Space()
        self._pm_space.iterations = 10
        self._pm_space.sleep_time_threshold = 0.5
        pymunk.pygame_util.positive_y_is_up = True

        self._static_body = self._pm_space.static_body

        self._robots_list = None

    def update_robots(self, robots: list):
        self._robots_list = robots
        for robot in robots:
            self.update_robot(robot)

    def update_robot(self, robot: AnimationHandle):
        # Add all bodies and shape
        self._pm_space.add(robot._body)
        self._pm_space.add(robot._control_body)
        self._pm_space.add(robot._shape)

        self._pm_space.add(robot._pivot)
        self._pm_space.add(robot._gear)

    def update(self, dt):
        robot: AnimationHandle
        if self._robots_list:
            for robot in self._robots_list:
                robot.update(dt)
        self._pm_space.step(dt)

    def spin_once(self):
        # handle "global" events
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                exit()

        # draw everything
        self._pg_screen.fill(pygame.Color('white'))
        self._pm_space.debug_draw(self._draw_options)
        self.update(1/60)
        pygame.display.flip()

        self._dt = self._clock.tick(60)

    @staticmethod
    def body_to_world(body_vel: np.ndarray, pose: np.ndarray) -> np.ndarray:
        theta = pose[2]
        return np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]]) @ body_vel

    @staticmethod
    def world_to_body(world_vel: np.ndarray, pose: np.ndarray) -> np.ndarray:
        theta = pose[2]
        return np.array([
            [np.cos(theta),  np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0,              0,             1]]) @ world_vel
