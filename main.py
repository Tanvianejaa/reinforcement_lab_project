"""
main.py — Drone RL Evaluation Game
Author: Tanvi Aneja

Runs a 100-second game where three controllers compete to collect balloons.
  Human  → arrow keys
  PID    → classical control theory
  SAC    → reinforcement learning (trained with Soft Actor-Critic)

Usage:
    python main.py
"""

import os
import numpy as np
import pygame
from random import randrange
from math import sin, cos, pi, sqrt

from agents import HumanAgent, PIDAgent, SACAgent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS         = 60
WIDTH       = 800
HEIGHT      = 800
GRAVITY     = 0.08
MASS        = 1
ARM         = 25
TIME_LIMIT  = 100     # seconds
RESPAWN     = 3       # seconds
NUM_TARGETS = 150


def asset(path):
    return os.path.join(os.path.dirname(__file__), path)


# ---------------------------------------------------------------------------
# Drone state
# ---------------------------------------------------------------------------
class Drone:
    def __init__(self, name, color):
        self.name  = name
        self.color = color
        self.score = 0
        self.dead  = False
        self.respawn_timer = 0
        self.target_idx = 0
        self._reset()

    def _reset(self):
        self.x, self.xv, self.xa = 400.0, 0.0, 0.0
        self.y, self.yv, self.ya = 400.0, 0.0, 0.0
        self.angle, self.angv, self.anga = 0.0, 0.0, 0.0
        self.dead = False

    # Aliases used by SACAgent.get_obs
    @property
    def x_position(self): return self.x
    @property
    def y_position(self): return self.y
    @property
    def x_speed(self): return self.xv
    @property
    def y_speed(self): return self.yv
    @property
    def angular_speed(self): return self.angv


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
def run():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone RL — Tanvi Aneja")
    clock = pygame.time.Clock()

    # Sprites
    drone_frames = [
        pygame.transform.scale(
            pygame.image.load(asset(f"assets/balloon-flat-asset-pack/png/objects/drone-sprites/drone-{i}.png")).convert_alpha(),
            (80, 24)
        ) for i in range(1, 5)
    ]
    balloon_frames = [
        pygame.transform.scale(
            pygame.image.load(asset(f"assets/balloon-flat-asset-pack/png/balloon-sprites/red-plain/red-plain-{i}.png")).convert_alpha(),
            (30, 52)
        ) for i in range(1, 8)
    ]
    cloud1 = pygame.image.load(asset("assets/balloon-flat-asset-pack/png/background-elements/cloud-1.png")).convert_alpha()
    cloud2 = pygame.image.load(asset("assets/balloon-flat-asset-pack/png/background-elements/cloud-2.png")).convert_alpha()
    sun    = pygame.image.load(asset("assets/balloon-flat-asset-pack/png/background-elements/sun.png")).convert_alpha()
    cloud1.set_alpha(130); cloud2.set_alpha(130); sun.set_alpha(130)

    # Fonts
    bold = asset("assets/fonts/Roboto-Bold.ttf")
    reg  = asset("assets/fonts/Roboto-Regular.ttf")
    hud_font   = pygame.font.Font(bold, 18)
    score_font = pygame.font.Font(reg,  15)
    time_font  = pygame.font.Font(bold, 28)
    label_font = pygame.font.Font(bold, 13)
    big_font   = pygame.font.Font(bold, 80)

    # Agents + drones
    agents = [HumanAgent(), PIDAgent(), SACAgent()]
    drones = [Drone(a.name, a.color) for a in agents]

    targets = [(randrange(150, 650), randrange(150, 650)) for _ in range(NUM_TARGETS)]

    cx1, cy1, cs1 = 150.0, 200.0,  0.3
    cx2, cy2, cs2 = 400.0, 500.0, -0.2
    elapsed = 0.0
    step    = 0

    # Game loop
    while True:
        clock.tick(FPS)
        step    += 1
        elapsed += 1 / FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                elapsed = TIME_LIMIT  # trigger end
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                elapsed = TIME_LIMIT  # trigger end

        # Background
        screen.fill((131, 176, 181))
        cx1 += cs1
        if cx1 > WIDTH: cx1 = -cloud1.get_width()
        screen.blit(cloud1, (cx1, cy1))
        cx2 += cs2
        if cx2 < -cloud2.get_width(): cx2 = WIDTH
        screen.blit(cloud2, (cx2, cy2))
        screen.blit(sun, (630, -100))

        # Timer
        t_surf = time_font.render(f"Time : {max(0, int(TIME_LIMIT - elapsed))}", True, (255,255,255))
        screen.blit(t_surf, (WIDTH - t_surf.get_width() - 20, 20))

        for i, (drone, agent) in enumerate(zip(drones, agents)):
            tx, ty = targets[drone.target_idx]

            if drone.dead:
                drone.respawn_timer -= 1 / FPS
                if drone.name == "Human":
                    n = big_font.render(str(int(drone.respawn_timer) + 1), True, (255,255,255))
                    n.set_alpha(160)
                    screen.blit(n, (WIDTH//2 - n.get_width()//2, HEIGHT//2 - n.get_height()//2))
                if drone.respawn_timer <= 0:
                    drone._reset()
                continue

            # Get action
            if drone.name == "Human":
                tl, tr = agent.act()
            elif drone.name == "PID":
                tl, tr = agent.act([tx - drone.x, drone.xv, ty - drone.y, drone.yv, drone.angle, drone.angv])
            else:
                tl, tr = agent.act(SACAgent.get_obs(drone, targets))

            # Physics
            ang = drone.angle * pi / 180
            drone.xa = -(tl + tr) * sin(ang) / MASS
            drone.ya = GRAVITY - (tl + tr) * cos(ang) / MASS
            drone.anga = ARM * (tr - tl) / MASS
            drone.xv += drone.xa;  drone.yv += drone.ya;  drone.angv += drone.anga
            drone.x  += drone.xv;  drone.y  += drone.yv;  drone.angle += drone.angv

            dist = sqrt((drone.x - tx)**2 + (drone.y - ty)**2)
            if dist < 50 and drone.target_idx < NUM_TARGETS - 1:
                drone.target_idx += 1
                drone.score += 1
            elif dist > 1000:
                drone.dead = True
                drone.respawn_timer = RESPAWN

            # Draw
            alpha = 255 if drone.name == "Human" else 60

            bf = balloon_frames[int(step * 0.1) % len(balloon_frames)].copy()
            bf.set_alpha(alpha)
            screen.blit(bf, (tx - 15, ty - 26))

            df = pygame.transform.rotate(drone_frames[int(step * 0.3) % len(drone_frames)].copy(), drone.angle)
            df.set_alpha(alpha)
            screen.blit(df, (int(drone.x - df.get_width()//2), int(drone.y - df.get_height()//2)))

            lbl = label_font.render(drone.name, True, drone.color)
            screen.blit(lbl, (int(drone.x - lbl.get_width()//2), int(drone.y - 30)))

            hud_x = 20 + i * 130
            screen.blit(hud_font.render(drone.name,           True, drone.color),   (hud_x, 18))
            screen.blit(score_font.render(f"Score: {drone.score}", True, (255,255,255)), (hud_x, 42))

        if elapsed >= TIME_LIMIT:
            break

        pygame.display.flip()

    pygame.quit()

    # Results
    print("\n" + "=" * 40)
    print("         RESULTS")
    print("=" * 40)
    scores = [d.score for d in drones]
    for d in drones:
        print(f"  {d.name:<8} : {d.score} targets")
    print("=" * 40)
    print(f"  Winner : {drones[int(np.argmax(scores))].name}")
    print("=" * 40)


if __name__ == "__main__":
    run()
