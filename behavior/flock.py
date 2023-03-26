# !/usr/bin/python3
import random
from time import time
import pygame
import numpy as np
from app import params, utils
from behavior.behavior import Behavior
from entity.herd import Herd
from entity.shepherd import Shepherd


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

    def add_member(self, herd: Herd):
        self._boids.add(herd)

    def remain_in_screen(self, herd: Herd):
        if herd.pose[0] > params.SCREEN_WIDTH - params.BOX_MARGIN:
            herd.steer(np.array([-params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[0] < params.BOX_MARGIN:
            herd.steer(np.array([params.STEER_INSIDE, 0.]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[1] < params.BOX_MARGIN:
            herd.steer(np.array([0., params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)
        if herd.pose[1] > params.SCREEN_HEIGHT - params.BOX_MARGIN:
            herd.steer(np.array([0., -params.STEER_INSIDE]),
                       alt_max=params.BOID_MAX_FORCE)

        if herd.pose[0] > params.SCREEN_WIDTH:
            herd.pose[0] = params.SCREEN_WIDTH
        if herd.pose[0] < 0:
            herd.pose[0] = 0
        if herd.pose[1] < 0:
            herd.pose[1] = 0
        if herd.pose[1] > params.SCREEN_HEIGHT:
            herd.pose[1] = params.SCREEN_HEIGHT

    # Basic herd behaviors
    def wander(self, herd: Herd):
        rands = 2 * np.random.rand(len(self._boids)) - 1
        cos = np.cos([b.wandering_angle for b in self._boids])
        sin = np.sin([b.wandering_angle for b in self._boids])

        other_boid: Herd
        for i, other_boid in enumerate(self._boids):
            if herd == other_boid:
                nvel = utils.normalize(herd.velocity)
                # calculate circle center
                circle_center = nvel * params.WANDER_DIST
                # calculate displacement force
                c, s = cos[i], sin[i]
                displacement = np.dot(
                    np.array([[c, -s], [s, c]]), nvel * params.WANDER_RADIUS)
                herd.steer(circle_center + displacement,
                           alt_max=params.BOID_MAX_FORCE)
                herd.wandering_angle += params.WANDER_ANGLE * rands[i]

    def separate(self, herd: Herd, distance: float):
        number_of_neighbors = 0
        force = np.zeros(2)
        other_boid: Herd
        for other_boid in self._boids:
            if herd == other_boid:
                continue
            if utils.dist2(herd.pose, other_boid.pose) < distance ** 2:
                force -= other_boid.pose - herd.pose
                number_of_neighbors += 1
        if number_of_neighbors:
            force /= number_of_neighbors
        herd.steer(utils.normalize(force) * self._separation_weight,
                   alt_max=params.BOID_MAX_FORCE)

    def align(self, herd: Herd):
        # find the neighbors
        desired = np.zeros(2)
        number_of_neighbors = 0
        other_boid: Herd
        for other_boid in self._boids:
            if herd == other_boid:
                continue
            if utils.dist2(herd.pose, other_boid.pose) < herd._local_perception ** 2:
                number_of_neighbors += 1
                desired += other_boid.velocity

        if number_of_neighbors > 0:
            herd.steer((desired / number_of_neighbors - herd.velocity) * self._alignment_weight,
                       alt_max=params.BOID_MAX_FORCE)

    def coherse(self, herd: Herd):
        center = np.zeros(2)
        number_of_neighbors = 0
        other_boid: Herd
        for other_boid in self._boids:
            distance2 = utils.dist2(herd.pose, other_boid.pose)
            if distance2 < herd._local_perception ** 2 and \
               distance2 >= herd._local_boundary ** 2:
                number_of_neighbors += 1
                center += other_boid.pose

        if number_of_neighbors > 0:
            desired = center / number_of_neighbors - herd.pose
            herd.steer((desired - herd.velocity) * self._cohesion_weight,
                       alt_max=params.BOID_MAX_FORCE / 10)

    # Interaction with predators
    def in_danger(self, herd: Herd,
                  predators: list, danger_radius: float):
        shepherd: Shepherd
        for shepherd in predators:
            if utils.norm(herd.pose - shepherd.pose) <= danger_radius:
                return True
        return False

    def add_predators(self, predators: list):
        self._predators = predators

    def flee(self, shepherd: Shepherd, herd: Herd):
        pred_pose = shepherd.pose
        pred_vel = shepherd.velocity
        t = int(utils.norm(pred_pose - herd.pose) / params.BOID_MAX_SPEED)
        pred_future_pose = pred_pose + t * pred_vel

        too_close = utils.dist2(herd.pose, pred_future_pose) < 250**2
        if too_close:
            steering = (utils.normalize(herd.pose - pred_future_pose) *
                        params.BOID_MAX_SPEED -
                        herd.velocity)
            herd.steer(steering * self._fleeing_weight,
                       alt_max=params.BOID_MAX_FORCE)

    def update(self, motion_event, click_event):
        herd: Herd
        shepherd: Shepherd
        for i, herd in enumerate(self._boids):
            if self.in_danger(herd, self._predators, 400):
                self.align(herd)
                self.separate(herd, herd.local_boundary)
                self.coherse(herd)
                self.wander(herd)
                for shepherd in self._predators:
                    self.flee(shepherd, herd)
            else:
                self.wander(herd)
                self.separate(herd, herd.personal_space)
            self.remain_in_screen(herd)

        # Questionable loop here,
        # SHould investigate whether moving update back to the above loop
        # affects the overall interactions
        for herd in self._boids:
            herd.update()
