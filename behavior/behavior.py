# !/usr/bin/python3

import pygame
from abc import ABC, abstractmethod


class Behavior(ABC):

    def __init__(self):
        self._screen = None
        self._vis_entity = None

    @abstractmethod
    def update(self, *args, **kwargs):
        '''
        '''

    def _get_events(self, args):
        events = []
        for arg in args:
            if len(arg) == 0:
                continue
            for element in arg:
                if isinstance(element, pygame.event.Event):
                    events.append(element)

        return events
    
    def _get_mouse_pos(self, *arg):
        pass

    # Use with caution, for testing purposes
    def _set_screen(self, screen):
        self._screen = screen

    def set_vis_entity(self,vis_entity):
        self._vis_entity = vis_entity
        

