#!/usr/bin/python3

import pygame
import numpy as np


class SpriteHandle(pygame.sprite.Sprite):
    def __init__(self,
                 pose: np.ndarray,
                 width: int,
                 height: int):
        self._pose = pose
        self._width = width
        self._height = height
        
        # image is what get's painted on the screen
        self._image = pygame.Surface((self._width, self._height))
        self._image.set_colorkey((2, 3, 4))
        self._image.fill((2, 3, 4))


