"""
agents.py — All three drone controllers in one file
Author: Tanvi Aneja

Human  : arrow-key input
PID    : classical rule-based controller
SAC    : pre-trained reinforcement learning agent
"""

import os
import numpy as np
import pygame
from pygame.locals import K_UP, K_DOWN, K_LEFT, K_RIGHT
from stable_baselines3 import SAC
from math import sqrt, pi


# ---------------------------------------------------------------------------
# Human Agent
# ---------------------------------------------------------------------------
class HumanAgent:
    """Controlled by the player using arrow keys."""
    name = "Human"
    color = (100, 200, 255)

    def __init__(self):
        self.mean = 0.04
        self.amp  = 0.04
        self.diff = 0.003

    def act(self, obs=None):
        tl = tr = self.mean
        keys = pygame.key.get_pressed()
        if keys[K_UP]:    tl += self.amp;  tr += self.amp
        if keys[K_DOWN]:  tl -= self.amp;  tr -= self.amp
        if keys[K_LEFT]:  tl -= self.diff
        if keys[K_RIGHT]: tr -= self.diff
        return tl, tr


# ---------------------------------------------------------------------------
# PID Agent
# ---------------------------------------------------------------------------
class _PID:
    def __init__(self, Kp, Ki, Kd, hi, lo):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.hi, self.lo = hi, lo
        self._i = self._prev = 0.0

    def compute(self, error, dt=1/60):
        self._i += error * dt
        d = (error - self._prev) / dt if dt else 0
        self._prev = error
        return max(self.lo, min(self.hi, self.Kp*error + self.Ki*self._i + self.Kd*d))


class PIDAgent:
    """Classical nested PID loops — no learning, pure control theory."""
    name = "PID"
    color = (255, 180, 50)

    def __init__(self):
        self.mean = 0.04
        self.amp  = 0.04
        self.diff = 0.003
        self.x_pid  = _PID(0.2,  0, 0.2,  25,  -25)
        self.a_pid  = _PID(0.02, 0, 0.01, 1,   -1)
        self.y_pid  = _PID(2.5,  0, 1.5,  100, -100)
        self.yd_pid = _PID(1.0,  0, 0.0,  1,   -1)

    def act(self, obs):
        ex, xd, ey, yd, angle, ad = obs
        desired_angle = self.x_pid.compute(-ex)
        diff_act  = self.a_pid.compute(-(desired_angle - angle))
        desired_yd = self.y_pid.compute(ey)
        thrust_act = self.yd_pid.compute(-(desired_yd - yd))
        tl = self.mean + thrust_act * self.amp + diff_act * self.diff
        tr = self.mean + thrust_act * self.amp - diff_act * self.diff
        return tl, tr


# ---------------------------------------------------------------------------
# SAC Agent
# ---------------------------------------------------------------------------
class SACAgent:
    """Soft Actor-Critic reinforcement learning agent (pre-trained)."""
    name = "SAC"
    color = (100, 255, 120)

    def __init__(self, model_path="models/sac_drone.zip"):
        self.mean = 0.04
        self.amp  = 0.04
        self.diff = 0.003
        print(f"Loading SAC model from: {model_path}")
        self.model = SAC.load(model_path)

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        tl = self.mean + action[0] * self.amp + action[1] * self.diff
        tr = self.mean + action[0] * self.amp - action[1] * self.diff
        return tl, tr

    @staticmethod
    def get_obs(drone, targets):
        """Build the 7-dim SAC observation from current drone state."""
        tx, ty = targets[drone.target_idx]
        angle_up     = drone.angle / 180 * pi
        velocity     = sqrt(drone.xv**2 + drone.yv**2)
        dist         = sqrt((drone.x - tx)**2 + (drone.y - ty)**2) / 500
        ang_target   = np.arctan2(ty - drone.y, tx - drone.x)
        ang_vel_tgt  = ang_target - np.arctan2(drone.yv, drone.xv)
        return np.array([angle_up, velocity, drone.angv, dist, ang_target, ang_vel_tgt, dist], dtype=np.float32)
