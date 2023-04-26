#!/usr/bin/python3

import pygame
from app import params

from entity.entity import Entity
from behavior.behavior import Behavior


class Environment(object):

    def __init__(self):
        # Pygame for visualisation
        pygame.init()
        self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
        self._running = True
        self._clock = pygame.time.Clock()

        self._behaviors = []
        self._entities = []
        self._bodies = []

    @property
    def ok(self):
        return self._running

    def add_entity(self, entity: Entity):
        self._entities.append(entity)

    def add_behaviour(self, behavior: Behavior):
        behavior._set_screen(self._screen)
        self._behaviors.append(behavior)

    def update(self):
        '''
        Function to update behaviors and interaction between entities
        '''

    def display(self):
        entity: Entity
        for entity in self._entities:
            entity.display(self._screen)
        behavior: Behavior
        for behavior in self._behaviors:
            behavior.display(self._screen)

    def run_once(self):
        events = pygame.event.get()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False

        behavior: Behavior
        # motion_event, click_event = None, None
        for behavior in self._behaviors:
            behavior.update(events)

    def render(self):
        self._screen.fill(params.SIMULATION_BACKGROUND)
        self.display()
        pygame.display.flip()
        self._clock.tick(params.FPS)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
