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
    @abstractmethod
    def display(self, screen: pygame.Surface):
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