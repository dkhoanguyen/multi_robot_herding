# !/usr/bin/python3

from abc import ABC, abstractmethod


class Behavior(ABC):

    @abstractmethod
    def update(self, *args, **kwargs):
        '''
        '''

    @abstractmethod
    def display(self, screen):
        '''
        '''
