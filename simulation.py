"""Simulation classes."""
import pygame
import numpy as np
from entity.classic_boid import ClassicBoid
from entity.predator import Predator
from behavior.flock import Flock
from app import params, utils
from time import time


class Simulation:
    """Represent a simulation of a flock."""

    def __init__(self, screen):
        self.running = True
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.boids = []
        self.flock = Flock()
        self.predators = []
        self.to_update = pygame.sprite.Group()
        self.to_display = pygame.sprite.Group()

    def add_boid(self, pos):
        angle = np.pi * (2 * np.random.rand() - 1)
        vel = params.BOID_MAX_SPEED * np.array([np.cos(angle), np.sin(angle)])
        member = ClassicBoid(pose=pos,
                             velocity=vel,
                             alighment_weight=0.0,
                             cohesion_weight=0.0,
                             separation_weight=0.0,
                             local_perception=200.0,
                             local_boundary=0.0,
                             mass=20.0,
                             min_v=0.0,
                             max_v=5.0)
        self.boids.append(member)
        self.flock.add_member(member)

    def add_predator(self, pos):
        angle = np.pi * (2 * np.random.rand() - 1)
        vel = params.BOID_MAX_SPEED * np.array([np.cos(angle), np.sin(angle)])
        predator = Predator(pose=pos,
                            velocity=vel,
                            local_perception=200.0,
                            local_boundary=0.0,
                            mass=20,
                            min_v=0,
                            max_v=7.0)
        self.predators.append(predator)

    def update(self, motion_event, click_event):
        self.to_update.update(motion_event, click_event)

    def display(self):
        # for sprite in self.to_display:
        #     sprite.display(self.screen)
        self.flock.display(self.screen)
        if params.DEBUG:
            pygame.draw.polygon(
                self.screen, pygame.Color("turquoise"),
                [
                    (params.BOX_MARGIN, params.BOX_MARGIN),
                    (params.SCREEN_WIDTH - params.BOX_MARGIN,
                        params.BOX_MARGIN),
                    (params.SCREEN_WIDTH - params.BOX_MARGIN,
                        params.SCREEN_HEIGHT - params.BOX_MARGIN),
                    (params.BOX_MARGIN,
                        params.SCREEN_HEIGHT - params.BOX_MARGIN),
                ], 1)

    def init_run(self):
        # add 40 boids to the flock
        for x in range(1, 11):
            for y in range(1, 7):
                self.add_boid(utils.grid_to_px((x, y)))
        # self.add_boid(utils.grid_to_px((5, 5)))
        for x in range(5, 7):
            for y in range(5, 7):
                self.add_predator(utils.grid_to_px((x, y)))
        self.add_predator(utils.grid_to_px((5, 5)))
        self.to_display = pygame.sprite.Group(
            self.boids,
        )

    def run(self):
        self.init_run()
        dt = 0
        while self.running:
            self.clock.tick(params.FPS)
            t = time()
            motion_event, click_event = None, None
            self.screen.fill(params.SIMULATION_BACKGROUND)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return "PYGAME_QUIT"
            predator: Predator
            for predator in self.predators:
                # predator.follow_mouse()
                predator.wander(self.predators)
                predator.remain_in_screen()
                predator.update()
                predator.display(self.screen)
                predator.reset_steering()
            self.flock.update(motion_event, click_event, self.predators)
            self.display()
            pygame.display.flip()
            dt = time() - t

    def quit(self):
        self.running = False


screen = pygame.display.set_mode(params.SCREEN_SIZE)
s = Simulation(screen)
s.run()
