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
        # self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)

        self._behaviors = []
        self._entities = []
        self._bodies = []

        screen_width = params.SCREEN_SIZE[0]
        screen_height = params.SCREEN_SIZE[1]

        # Add boundaries
        left_segment = pymunk.Segment(self._static_body,
                                      (0.0, 0.0),
                                      (0.0, screen_height), 5.0)
        top_segment = pymunk.Segment(self._static_body,
                                     (0.0, 0.0),
                                     (screen_width, 0.0), 5.0)
        right_segment = pymunk.Segment(self._static_body,
                                       (screen_width, 0.0),
                                       (screen_width, screen_height), 5.0)
        bottom_segment = pymunk.Segment(self._static_body,
                                        (0.0, screen_height),
                                        (screen_width, screen_height), 5.0)
        self._space.add(left_segment)
        self._space.add(top_segment)
        self._space.add(right_segment)
        self._space.add(bottom_segment)

    @property
    def ok(self):
        return self._running

    def add_entity(self, entity: Entity):
        self._entities.append(entity)

        # # Added pymunk physic elements
        # addables = entity.get_pymunk_addables()
        # for key, addable in addables.items():
        #     self._space.add(addable)
        #     if key == 'body':
        #         self._bodies.append(addable)

    def add_behaviour(self, behavior: Behavior):
        self._behaviors.append(behavior)

    def update(self):
        '''
        Function to update behaviors and interaction between entities
        '''
        

        # for body in self._bodies:
        #     self._space.reindex_shapes_for_body(body)

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
