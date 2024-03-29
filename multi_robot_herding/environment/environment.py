#!/usr/bin/python3

import pygame
import numpy as np
from multi_robot_herding.utils import params

from multi_robot_herding.entity.entity import Entity
from multi_robot_herding.behavior.behavior import Behavior

class Background(object):
    def __init__(self):
        pass

class Environment(object):

    def __init__(self, multi_threaded=False):
        self._multi_threaded = multi_threaded

        # Pygame for visualisation
        pygame.init()
        self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
        self._rect = self._screen.get_rect()
        self._running = True
        self._clock = pygame.time.Clock()

        self._behaviors = []
        self._entities = []
        self._bodies = []

        self._entities = {}

        # This should be a config in the future
        entities_name = ["herd", "shepherd", "obstacle"]
        for entity_name in entities_name:
            self._entities[entity_name] = []

        self._set_static_entities = False

    @property
    def ok(self):
        return self._running

    def add_entity(self, entity: Entity):
        self._entities[entity.__str__()].append(entity)

    def add_behaviour(self, behavior: Behavior):
        self._behaviors.append(behavior)

    def update(self):
        '''
        Function to update behaviors and interaction between entities
        '''

    def display(self):
        entity: Entity
        for entities in self._entities.values():
            for entity in entities:
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

        # Set static entities
        if not self._set_static_entities:
            for shepherd in self._entities["shepherd"]:
                for obstacle in self._entities["obstacle"]:
                    shepherd.add_static_obstacle(obstacle)
            self._set_static_entities = True

        # Grab all states
        all_states = {}
        for entity_type in self._entities.keys():
            if entity_type == "obstacle":
                continue
            all_states[entity_type] = np.empty((0, 6))
            for entity in self._entities[entity_type]:
                all_states[entity_type] = np.vstack(
                    (all_states[entity_type], entity.state))

        shepherds_id = []
        for entity in self._entities["shepherd"]:
            shepherds_id.append(entity.id)
        shepherds_id = np.array(shepherds_id)

        # Consensus state
        all_consensus_states = []
        for entity in self._entities["shepherd"]:
            all_consensus_states.append(entity.consensus_state)

        # Thread for each entity
        entity: Entity
        for entity in self._entities["shepherd"]:
            entity.update(events=events,
                          ids=shepherds_id,
                          entity_states=all_states,
                          consensus_states=all_consensus_states)

    def render(self):
        self._screen.fill(params.SIMULATION_BACKGROUND)
        self.display()
        pygame.display.flip()
        self._clock.tick(params.FPS)
        pygame.display.set_caption("fps: " + str(self._clock.get_fps()))
