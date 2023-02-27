#!/usr/bin/python3

import numpy as np

import pygame

import pymunk
import pymunk.pygame_util

from animation.animation_handle import AnimationHandle


class Environment(object):

    def __init__(self,
                 width: int = 700,
                 height: int = 700):
        pygame.init()
        self._pg_screen = pygame.display.set_mode(
            (width, height))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._pg_screen)

        self._pm_space = pymunk.Space()
        self._pm_space.iterations = 10
        self._pm_space.sleep_time_threshold = 0.5
        pymunk.pygame_util.positive_y_is_up = True

        self._static_body = self._pm_space.static_body

        self._fps = 60

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = value

    def add(self, entity: AnimationHandle):
        # Add all bodies and shape
        bodies = entity.get_bodies()
        for _, body in bodies.items():
            self._pm_space.add(body)

    def visualise(self):
        # handle "global" events
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                exit()

        # draw everything
        self._pg_screen.fill(pygame.Color('white'))
        self._pm_space.debug_draw(self._draw_options)

        # Update pymunk space
        self._pm_space.step(1/self._fps)

        # Update pygame visualisation
        pygame.display.flip()

        self._dt = self._clock.tick(self._fps)

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
