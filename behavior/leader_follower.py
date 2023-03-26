# !/usr/bin/python3

import math
from time import time
from behavior.behavior import Behavior
import numpy as np
from entity.entity import Autonomous
from entity.shepherd import Shepherd
from app import params, utils
from enum import Enum

from spatialmath.base import *


class LeaderFollowerType(Enum):
    LINE = 1
    COLUMN = 2


class LeaderFollower(Behavior):

    def __init__(self,
                 type: LeaderFollowerType,
                 formation_weight: float,
                 spacing: float):
        super().__init__()
        self._type = type
        self._formation_weight = formation_weight
        self._spacing = spacing

        self._members_list = []
        self._leader = None
        self._followers_list = []

        self._followers_poses = []

        # Control variables
        self._init_done = False

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    def add_follower(self, follower: Autonomous):
        self._followers_list.append(follower)
        self._members_list.append(follower)

    def add_leader(self, leader: Autonomous):
        self._leader = leader
        self._members_list.append(leader)

    def init_formation(self):
        '''
        Gather all followers around the leader
        Default formation to line
        '''
        # if self._init_done:
        #     return
        leader_pose = self._construct_entity_pose(self._leader)
        formation_links = self._generate_formation_link(self._type,
                                                        self._members_list,
                                                        self._spacing)
        self._followers_poses = self._generate_followers_poses(formation_links.transpose(),
                                                               leader_pose,
                                                               self._followers_list)
        self._init_done = True

    def maintain_formation(self):
        '''
        '''
        if not self._init_done:
            return

        follower: Autonomous
        for i, follower in enumerate(self._followers_list):
            target_pose = self._followers_poses[i, 0:2]
            follower.move_to_pose(target_pose)
        self._leader.follow_mouse()
        self.remain_in_screen(self._leader)

    def update(self, motion_event, click_event):
        self.init_formation()
        self.maintain_formation()

        member: Autonomous
        for member in self._members_list:
            member.update()

    def remain_in_screen(self, herd):
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

    def _construct_entity_pose(self, entity: Autonomous):
        return np.array([
            entity.pose[0],
            entity.pose[1],
            entity.heading])

    def _generate_formation_link(self,
                                 type: LeaderFollowerType,
                                 members: list,
                                 spacing: float):
        if type == LeaderFollowerType.LINE:
            # Get number of followers
            num_members = len(members)
            mid = np.round(num_members / 2)
            line_links_x = np.linspace(-mid * spacing,
                                       (mid - 1 + ((num_members) % 2)) * spacing,
                                       num_members)
            line_links_y = np.zeros(line_links_x.shape)
            line_links = np.vstack((line_links_x, line_links_y))
            return line_links

        if type == LeaderFollowerType.COLUMN:
            # Get number of followers
            num_members = len(members)
            mid = np.round(num_members / 2)
            line_links_y = np.linspace(-mid * spacing,
                                       (mid - 1 + ((num_members) % 2)) * spacing,
                                       num_members)
            line_links_x = np.zeros(line_links_y.shape)
            line_links = np.vstack((line_links_x, line_links_y))
            return line_links

    def _generate_followers_poses(self,
                                  links: list,
                                  leader_pose: np.ndarray,
                                  followers: list):
        followers_poses = np.empty((0, 3))
        mid = np.round(len(followers) / 2)
        for i, link in enumerate(links):
            if i == int(mid):
                continue
            else:
                # Perform a transformation from the leader pose to the
                # formation pose
                formation_pose = transl2(leader_pose[0:2]) @ \
                    trot2(np.pi - leader_pose[2]) @ \
                    transl2(link)
                formation_pose = np.array(
                    [formation_pose[0, 2], formation_pose[1, 2], leader_pose[2]])
                followers_poses = np.vstack(
                    (followers_poses, formation_pose))
        return followers_poses
