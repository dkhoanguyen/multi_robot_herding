# !/usr/bin/python3
import random
from time import time
import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.classic_boid import ClassicBoid
from entity.predator import Predator


class Flock(Behavior):

    def __init__(self,
                 alignment_weight: float,
                 cohesion_weight: float,
                 separation_weight: float,
                 fleeing_weight: float):
        super().__init__()
        self._alignment_weight = alignment_weight
        self._cohesion_weight = cohesion_weight
        self._separation_weight = separation_weight
        self._fleeing_weight = fleeing_weight

        self._boids = pygame.sprite.Group()
        self._predators = []
        self._grazing_time = []

    def add_member(self, boid: ClassicBoid):
        self._boids.add(boid)

    def remain_in_screen(self, boid: ClassicBoid):
        if boid.pose[0] > params.SCREEN_WIDTH - params.BOX_MARGIN:
            boid.steer(np.array([-params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if boid.pose[0] < params.BOX_MARGIN:
            boid.steer(np.array([params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if boid.pose[1] < params.BOX_MARGIN:
            boid.steer(np.array([0., params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)
        if boid.pose[1] > params.SCREEN_HEIGHT - params.BOX_MARGIN:
            boid.steer(np.array([0., -params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)

    # Basic boid behaviors
    def wander(self, boid: ClassicBoid):
        rands = 2 * np.random.rand(len(self._boids)) - 1
        cos = np.cos([b.wandering_angle for b in self._boids])
        sin = np.sin([b.wandering_angle for b in self._boids])

        other_boid: ClassicBoid
        for i, other_boid in enumerate(self._boids):
            if boid == other_boid:
                nvel = utils.normalize(boid.velocity)
                # calculate circle center
                circle_center = nvel * params.WANDER_DIST
                # calculate displacement force
                c, s = cos[i], sin[i]
                displacement = np.dot(
                    np.array([[c, -s], [s, c]]), nvel * params.WANDER_RADIUS)
                boid.steer(circle_center + displacement,
                           alt_max=params.BOID_MAX_FORCE)
                boid.wandering_angle += params.WANDER_ANGLE * rands[i]

    def separate(self, boid: ClassicBoid, distance: float):
        number_of_neighbors = 0
        force = np.zeros(2)
        other_boid: ClassicBoid
        for other_boid in self._boids:
            if boid == other_boid:
                continue
            if utils.dist2(boid.pose, other_boid.pose) < distance ** 2:
                force -= other_boid.pose - boid.pose
                number_of_neighbors += 1
        if number_of_neighbors:
            force /= number_of_neighbors
        boid.steer(utils.normalize(force) * self._separation_weight,
                   alt_max=params.BOID_MAX_FORCE)

    def align(self, boid: ClassicBoid):
        # find the neighbors
        desired = np.zeros(2)
        number_of_neighbors = 0
        other_boid: ClassicBoid
        for other_boid in self._boids:
            if boid == other_boid:
                continue
            if utils.dist2(boid.pose, other_boid.pose) < boid._local_perception ** 2:
                number_of_neighbors += 1
                desired += other_boid.velocity

        if number_of_neighbors > 0:
            boid.steer((desired / number_of_neighbors - boid.velocity) * self._alignment_weight,
                       alt_max=params.BOID_MAX_FORCE)

    def coherse(self, boid: ClassicBoid):
        center = np.zeros(2)
        number_of_neighbors = 0
        other_boid: ClassicBoid
        for other_boid in self._boids:
            distance2 = utils.dist2(boid.pose, other_boid.pose)
            if distance2 < boid._local_perception ** 2 and \
               distance2 >= boid._local_boundary ** 2:
                number_of_neighbors += 1
                center += other_boid.pose

        if number_of_neighbors > 0:
            desired = center / number_of_neighbors - boid.pose
            boid.steer((desired - boid.velocity) * self._cohesion_weight,
                       alt_max=params.BOID_MAX_FORCE / 10)

    # Interaction with predators
    def in_danger(self, boid: ClassicBoid,
                  predators: list, danger_radius: float):
        predator: Predator
        for predator in predators:
            if utils.norm(boid.pose - predator.pose) <= danger_radius:
                return True
        return False
    
    def add_predators(self, predators: list):
        self._predators = predators

    def flee(self, predator: Predator, boid: ClassicBoid):
        pred_pose = predator.pose
        pred_vel = predator.velocity
        t = int(utils.norm(pred_pose - boid.pose) / params.BOID_MAX_SPEED)
        pred_future_pose = pred_pose + t * pred_vel

        too_close = utils.dist2(boid.pose, pred_future_pose) < 400**2
        if too_close:
            steering = (utils.normalize(boid.pose - pred_future_pose) *
                        params.BOID_MAX_SPEED -
                        boid.velocity)
            boid.steer(steering * self._fleeing_weight,
                       alt_max=params.BOID_MAX_FORCE)

    def update(self, motion_event, click_event):
        boid: ClassicBoid
        predator: Predator
        for i, boid in enumerate(self._boids):
            if self.in_danger(boid, self._predators, 450):
                self.align(boid)
                self.separate(boid, boid.local_boundary)
                self.coherse(boid)
                for predator in self._predators:
                    self.flee(predator, boid)
            else:
                self.wander(boid)
                self.separate(boid, boid.personal_space)
            self.remain_in_screen(boid)

        # Questionable loop here,
        # SHould investigate whether moving update back to the above loop
        # affects the overall interactions
        for boid in self._boids:
            boid.update()
