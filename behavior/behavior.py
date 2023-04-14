# !/usr/bin/python3

import pygame
from abc import ABC, abstractmethod


class Behavior(ABC):

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
        

