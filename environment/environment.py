#!/usr/bin/python3

import pygame
import numpy as np

from app import params, utils

from entity.entity import Entity, Autonomous
from behavior.behavior import Behavior


class Environment(object):

    def __init__(self):
        self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
        self._running = True
        self._clock = pygame.time.Clock()

        self._behaviors = []

    def add_behaviour(self, behavior: Behavior):
        self._behaviors.append(behavior)

    def update(self):
        behavior: Behavior
        motion_event, click_event = None, None
        for behavior in self._behaviors:
            behavior.update(motion_event, click_event)

    def display(self):
        behavior: Behavior
        for behavior in self._behaviors:
            behavior.display(self._screen)

    def run(self):
        while self._running:
            self._clock.tick(params.FPS)
            self._screen.fill(params.SIMULATION_BACKGROUND)
            motion_event, click_event = None, None
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.update()
            self.display()
            pygame.display.flip()
