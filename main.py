# import pygame
# import random
# import math

# class Thingy(pygame.sprite.Sprite):
#     def __init__(self, area):
#         super().__init__()
#         self._physical_R = 15
#         self._wheel_d = 0
#         self._wheel_w = 0

#         self._robot_w = 2 * self._wheel_w + 2 * self._physical_R
#         self._robot_l = self._wheel_d if self._wheel_d >= 2 * self._physical_R \
#             else 2 * self._physical_R

#         self._surface_w = self._robot_w
#         self._surface_l = self._robot_l

#         # image is what get's painted on the screen
#         self.image = pygame.Surface((self._surface_l, self._surface_w))
#         self.image.set_colorkey((2, 3, 4))
#         self.image.fill((2, 3, 4))

#         x = self._surface_l/2
#         y = self._surface_w/2

#         pygame.draw.circle(self.image, pygame.Color(
#             'blue'), (x, y), self._physical_R, 2)
#         # pygame.draw.rect(self.image, pygame.Color('blue'),
#         #                  (x - self._wheel_d/2, y - self._physical_R - self._wheel_w, self._wheel_d, self._wheel_w), 0)
#         # pygame.draw.rect(self.image, pygame.Color('blue'),
#         #                  (x - self._wheel_d/2, y + self._physical_R, self._wheel_d, self._wheel_w), 0)
#         pygame.draw.line(self.image, pygame.Color('blue'),(x, y-1),(x + self._physical_R,y-1),2)

#         # we keep a reference to the original image
#         # since we use that for rotation to prevent distortions
#         self.original = self.image.copy()
#         # rect is used to determine the position of a sprite on the screen
#         # the Rect class also offers a lot of useful functions
#         self.rect = self.image.get_rect(center=(500, 500))
#         self.angle = 0
#         self.area = area

#     def update(self, events, dt):
#         dist = 5
#         pressed = pygame.key.get_pressed()
#         if pressed[pygame.K_UP]:
#             vx = dist

#             self.rect.move_ip(0, -5)
#         # if pressed[pygame.K_DOWN]:
#         #     self.rect.move_ip(0, 5)
#         # if pressed[pygame.K_LEFT]:
#         #     self.rect.move_ip(-5, 0)
#         # if pressed[pygame.K_RIGHT]:
#         #     self.rect.move_ip(5, 0)
#         if pressed[pygame.K_a]:
#             if self.angle < 180:
#                 self.angle += 1
#         if pressed[pygame.K_d]:
#             if self.angle > -180:
#                 self.angle -= 1

#         # let's rotate the image, but ensure that we keep the center position
#         # so it doesn't "jump around"
#         self.image = pygame.transform.rotate(self.original, self.angle)
#         self.rect = self.image.get_rect(center=self.rect.center)
#         self.rect.clamp_ip(self.area)

#     def deg2rad(self, deg):
#         return deg/180*math.pi

# def main():
#     pygame.init()
#     screen = pygame.display.set_mode((1000, 1000))
#     screen_rect = screen.get_rect()
#     clock = pygame.time.Clock()
#     sprites = pygame.sprite.Group(Thingy(screen_rect))

#     dt = 0
#     while True:
#         # nice clean main loop
#         # all game logic goes into the sprites

#         # handle "global" events
#         events = pygame.event.get()
#         for e in events:
#             if e.type == pygame.QUIT:
#                 return

#         # update all sprites
#         sprites.update(events, dt)

#         # draw everything
#         screen.fill(pygame.Color('white'))
#         sprites.draw(screen)
#         pygame.display.update()
#         dt = clock.tick(60)


# if __name__ == '__main__':
#     main()

import pymunk
from pymunk.pygame_util import *
from pymunk.vec2d import Vec2d

import pygame
from pygame.locals import *
import random

space = pymunk.Space()
b0 = space.static_body
size = w, h = 700, 300

GRAY = (220, 220, 220)
RED = (255, 0, 0)


class Circle:
    def __init__(self, pos, radius=20):
        self.body = pymunk.Body()
        self.body.position = pos
        shape = pymunk.Circle(self.body, radius)
        shape.density = 0.01
        shape.friction = 0.9
        shape.elasticity = 1
        space.add(self.body, shape)


class Box:
    def __init__(self, p0=(0, 0), p1=(w, h), d=4):
        x0, y0 = p0
        x1, y1 = p1
        ps = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for i in range(4):
            segment = pymunk.Segment(b0, ps[i], ps[(i+1) % 4], d)
            segment.elasticity = 1
            segment.friction = 1
            space.add(segment)


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.draw_options = DrawOptions(self.screen)
        self.running = True

    def run(self):
        while self.running:
            for event in pygame.event.get():
                self.do_event(event)
            self.draw()
            space.step(0.01)

        pygame.quit()

    def do_event(self, event):
        if event.type == QUIT:
            self.running = False

        elif event.type == KEYDOWN:
            if event.key in (K_q, K_ESCAPE):
                self.running = False

            if event.key == K_p:
                pygame.image.save(self.screen, 'mouse.png')
        elif event.type == MOUSEBUTTONDOWN:
            p = from_pygame(event.pos, self.screen)
            self.active_shape = None
            for s in space.shapes:
                dist, _ = s.point_query(p)
                if dist < 0:
                    self.active_shape = s

    def draw(self):
        self.screen.fill(GRAY)
        space.debug_draw(self.draw_options)
        pygame.display.update()


if __name__ == '__main__':
    Box()
    r = 25
    for i in range(9):
        x = random.randint(r, w-r)
        y = random.randint(r, h-r)
        Circle((x, y), r)
    App().run()
