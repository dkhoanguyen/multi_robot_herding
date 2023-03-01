# !/usr/bin/python3

from abc import ABC, abstractmethod

class Behavior(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def interact(self, *args, **kwargs):
        pass