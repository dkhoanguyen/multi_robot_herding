# !/usr/bin/python3

import pygame
import numpy as np
from app import params, utils
from entity.classic_boid import ClassicBoid
from entity.predator import Predator


class Flock():

    def __init__(self):
        super().__init__()
        self._boids = pygame.sprite.Group()

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

    def separate(self, boid: ClassicBoid):
        number_of_neighbors = 0
        force = np.zeros(2)
        other_boid: ClassicBoid
        for other_boid in self._boids:
            if boid == other_boid:
                continue
            if pygame.sprite.collide_rect(boid, other_boid):
                force -= other_boid.pose - boid.pose
                number_of_neighbors += 1
        if number_of_neighbors:
            force /= number_of_neighbors
        boid.steer(utils.normalize(force) * params.MAX_SEPARATION_FORCE,
                   alt_max=params.BOID_MAX_FORCE)

    def align(self, boid: ClassicBoid):
        r2 = params.ALIGN_RADIUS * params.ALIGN_RADIUS
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
            boid.steer(desired / number_of_neighbors -
                       boid.velocity, alt_max=params.BOID_MAX_FORCE)

    # Interaction with predators
    def flee(self, predator: Predator, boid: ClassicBoid):
        pred_pose = predator.pose
        pred_vel = predator.velocity
        t = int(utils.norm(pred_pose - boid.pose) / params.BOID_MAX_SPEED)
        pred_future_pose = pred_pose + t * pred_vel

        too_close = utils.dist2(boid.pose, pred_future_pose) < params.R_FLEE**2
        if too_close:
            steering = (utils.normalize(boid.pose - pred_future_pose) *
                        params.BOID_MAX_SPEED -
                        boid.velocity)
            boid.steer(steering, alt_max=params.BOID_MAX_FORCE / 10)

    def update(self, motion_event, click_event, predators):
        boid: ClassicBoid
        predator: Predator
        for boid in self._boids:
            for predator in predators:
                self.flee(predator,boid)
            self.wander(boid)
            self.align(boid)
            self.separate(boid)
            self.remain_in_screen(boid)

        for boid in self._boids:
            boid.update()

    def display(self, screen):
        boid: ClassicBoid
        for boid in self._boids:
            boid.display(screen, debug=False)
        for boid in self._boids:
            boid.reset_steering()
