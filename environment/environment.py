#!/usr/bin/python3

import pygame
import pymunk
import pymunk.pygame_util
from app import params

from entity.entity import Entity, Autonomous
from behavior.behavior import Behavior


class Environment(object):

    def __init__(self):
        # Pymunk for physic engine
        self._space = pymunk.Space()
        self._space.iterations = 10
        self._space.sleep_time_threshold = 0.5
        self._static_body = self._space.static_body

        # Pygame for visualisation
        pygame.init()
        self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
        self._running = True
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        self._behaviors = []
        self._entities = []

    def add_entity(self, entity: Entity):
        self._entities.append(entity)

        # Added pymunk physic elements
        addables = entity.get_pymunk_addables()
        for _, addable in addables.items():
            self._space.add(addable)

    def add_behaviour(self, behavior: Behavior):
        self._behaviors.append(behavior)

    def update(self):
        '''
        Function to update behaviors and interaction between entities
        '''
        behavior: Behavior
        motion_event, click_event = None, None
        for behavior in self._behaviors:
            behavior.update(motion_event, click_event)

    def display(self):
        entity: Entity
        for entity in self._entities:
            entity.display(self._screen)

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

    def run_once(self):
        self._clock.tick(params.FPS)
        self._screen.fill(params.SIMULATION_BACKGROUND)
        self._space.debug_draw(self._draw_options)
        motion_event, click_event = None, None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.update()
        self._space.step(1/params.FPS)
        # self.display()
        pygame.display.flip()
        # self._clock.tick(params.FPS)
