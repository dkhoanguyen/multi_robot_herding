# !/usr/bin/python3

import math
import numpy as np

from behavior.entity import Entity, EntityType


class ClassicBoid(Entity):
    def __init__(self,
                 pose: np.ndarray,
                 velocity: np.ndarray,
                 alighment_weight: float,
                 cohesion_weight: float,
                 separation_weight: float,
                 local_perception: float,
                 local_boundary: float,
                 min_vx: float,
                 max_vx: float):
        super(ClassicBoid, self).__init__(
            pose=pose,
            velocity=velocity,
            type=EntityType.AUTONOMOUS)

        self._max_steer_force = 1

        self._alighment_weight = alighment_weight
        self._cohesion_weight = cohesion_weight
        self._separation_weight = separation_weight

        self._local_perception = local_perception
        self._local_boundary = local_boundary

        self._max_vx = max_vx
        self._min_vx = min_vx
        self._max_w = 100000

    def alignment(self, entities: list):
        avg_heading = 0
        i = 0
        entity: Entity
        for entity in entities:
            # Ignore if enity is not part of the flock
            if entity.type != EntityType.AUTONOMOUS:
                continue
            if np.linalg.norm(entity.pose[0:2] - self._pose[0:2]) <= self._local_boundary:
                avg_heading += entity.pose[2]
                i += 1

        avg_heading = avg_heading / i
        steering = avg_heading - self._pose[2]
        return steering

    def cohesion(self, entities: list):
        steering = 0
        total = 0
        center_of_mass = np.zeros(2)
        entity: Entity
        for entity in entities:
            # Ignore if enity is not part of the flock
            if entity.type != EntityType.AUTONOMOUS:
                continue
            if np.linalg.norm(entity.pose[0:2] - self._pose[0:2]) < self._local_perception:
                center_of_mass += entity.pose[0:2]
                total += 1
        if total > 1:
            center_of_mass /= total
            vec_to_com = center_of_mass - self._pose[0:2]
            steering = math.atan2(vec_to_com[1], vec_to_com[0]) - self._pose[2]

        return steering

    def separation(self, entities: list):
        steering = 0
        total = 0
        avg_vector = np.zeros(2)
        slow_down = False

        entity: Entity
        for entity in entities:
            # Ignore if enity is not part of the flock
            if entity.type != EntityType.AUTONOMOUS:
                continue
            distance = np.linalg.norm(entity.pose[0:2] - self._pose[0:2])
            if not np.array_equal(self._pose[0:2], entity.pose[0:2]) \
               and distance < self._local_boundary:
                diff = self._pose[0:2] - entity.pose[0:2]
                # This is to get the unit vector
                avg_vector += diff / (distance * distance)
                total += 1
            elif distance < 40:
                slow_down = True

        if total > 0:
            avg_vector /= total
            steering = math.atan2(avg_vector[1], avg_vector[0]) - self._pose[2]
        return steering

    def collision_avoidance(self, entities: list):
        steering = 0
        total = 0
        avg_vector = np.zeros(2)
        slow_down = False
        entity: Entity
        for entity in entities:
            if entity.type != EntityType.OBSTACLE:
                continue
            distance = np.linalg.norm(entity.pose[0:2] - self._pose[0:2])
            if distance < 65:
                diff = self._pose[0:2] - entity.pose[0:2]
                # This is to get the unit vector
                diff /= distance
                avg_vector += diff
                total += 1
            # if distance < 55:
            #     slow_down = True

        if total > 0:
            avg_vector /= total
            steering = math.atan2(avg_vector[1], avg_vector[0]) - self._pose[2]
        return steering, slow_down

    def apply_behaviour(self, entities: list, dt: float):
        num_flock_mates = 0
        flock_heading = np.zeros(2)
        flock_center = np.zeros(2)
        separation_heading = np.zeros(2)
        acceleration = np.array([50.0,0.0])
        turn = 0

        entity: Entity
        for entity in entities:
            # Ignore if enity is not part of the flock
            if entity.type != EntityType.AUTONOMOUS:
                continue
            # Ignore itself
            if np.array_equal(self._pose[0:2], entity.pose[0:2]):
                continue
            distance = np.linalg.norm(entity.pose[0:2] - self._pose[0:2])
            if distance < self._local_perception:
                num_flock_mates += 1
                flock_heading += np.array([math.cos(entity.pose[2]),
                                          math.sin(entity.pose[2])])
                flock_center += entity.pose[0:2]

                if distance < self._local_boundary:
                    offset = entity.pose[0:2] - self._pose[0:2]
                    separation_heading -= offset / distance

        if num_flock_mates > 0:
            flock_heading /= num_flock_mates
            flock_center /= num_flock_mates
            offset_to_center = flock_center - self._pose[0:2]

            alignment_force = self.steer_toward(flock_heading)
            cohesion_force = self.steer_toward(offset_to_center)
            separation_force = self.steer_toward(separation_heading)

            acceleration += self._alighment_weight * alignment_force
            # acceleration += self._cohesion_weight * cohesion_force
            acceleration += self._separation_weight * separation_force
            # print("speration heading: ", separation_heading)
            self._velocity += acceleration

            turn = math.atan2(self._velocity[1], self._velocity[0])

        return np.array([self._max_vx, 0, turn])

    def steer_toward(self, vector: np.ndarray):
        if np.linalg.norm(vector) != 0:
            steer = vector / np.linalg.norm(vector) - self._velocity
        else:
            steer = np.zeros(2)
        return steer

        # alignment = self._alighment_weight * self.alignment(entities)
        # cohesion = self._cohesion_weight * self.cohesion(entities)
        # separation = self._separation_weight * self.separation(entities)
        # obs_avoidance, slow_down = self.collision_avoidance(entities)

        # avg_steering = alignment + cohesion + separation + 1 * obs_avoidance
        # if not slow_down:
        #     return np.array([self._max_vx, 0, avg_steering])
        # else:
        #     return np.array([0.1 * self._max_vx, 0, avg_steering])
