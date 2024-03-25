#!/usr/bin/python3

import time
import pickle
import pygame
import numpy as np

from multi_robot_herding.utils import params

from multi_robot_herding.entity.entity import Entity
from multi_robot_herding.behavior.behavior import Behavior

class Background(object):
    def __init__(self):
        pass

class Environment(object):

    def __init__(self, render,
                       config,
                       multi_threaded=False,
                       save_to_file=True,
                       save_path="data/"):
        self._multi_threaded = multi_threaded
        self._render = render

        # Pygame for visualisation
        if self._render:
            pygame.init()
            self._screen = pygame.display.set_mode(params.SCREEN_SIZE)
            self._rect = self._screen.get_rect()
            self._clock = pygame.time.Clock()
        self._running = True
        
        self._behaviors = []
        self._entities = []
        self._bodies = []

        self._entities = {}

        # This should be a config in the future
        entities_name = ["herd", "shepherd", "obstacle", "robot"]
        for entity_name in entities_name:
            self._entities[entity_name] = []

        self._set_static_entities = False
        self._save_to_file = save_to_file
        self._save_path = save_path

        self._data_to_save = {}
        self._data_to_save["configuration"] = config
        self._data_to_save["data"] = []

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
        for entity in self._entities["robot"]:
            entity.update(events=events,
                          ids=shepherds_id,
                          entity_states=all_states,
                          consensus_states=all_consensus_states)
        
        all_states.update({"ts": time.time()})
        self._data_to_save["data"].append(all_states.copy())
        
    def render(self):
        if self._render:
            self._screen.fill(params.SIMULATION_BACKGROUND)
            self.display()
            pygame.display.flip()
            self._clock.tick(params.FPS)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

    def quit(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.unicode.isalpha():
                letter = event.unicode.upper()
                if letter == 'Q':
                    self._running = False
                    return True
        return False

    def save_data(self):
        path = self._save_path + "all_states_" + str(int(time.time())) + ".pickle"
        with open(path, 'wb') as file:
            pickle.dump(self._data_to_save, file)