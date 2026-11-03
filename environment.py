"""
environment.py — Custom Gymnasium Environment for 2D Drone Control
Author: Tanvi Aneja
Subject: Reinforcement Learning Lab Project

This file defines the drone simulation as a Markov Decision Process (MDP):
  - State  (observation): 7 values describing drone orientation, velocity,
                          and position relative to the current target
  - Action:               2 continuous values controlling left/right thruster forces
  - Reward:               +1.0 for reaching a target,
                          -1.0 for going out of bounds,
                          small negative penalty proportional to distance each step
  - Episode ends when:    drone flies out of bounds OR 6000 steps elapsed
"""

import numpy as np
import gymnasium as gym
from math import sin, cos, pi, sqrt


class DroneEnv(gym.Env):
    """
    2D Quadcopter Environment.

    The drone operates under Newtonian rigidbody physics.
    Two thrusters (left and right) apply forces that move and rotate the drone.
    The goal is to reach as many randomly placed targets as possible.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # ── Physics constants ────────────────────────────────────────────────
        self.gravity = 0.08          # Downward acceleration each step
        self.mass = 1                # Drone mass (kg, normalised)
        self.arm = 25                # Distance from centre to propeller (px)
        self.thruster_mean = 0.04    # Idle thrust (counteracts gravity roughly)
        self.thruster_amp = 0.04     # Max additional thrust (up/down)
        self.diff_amp = 0.003        # Max differential thrust (rotation)

        # ── Action space ─────────────────────────────────────────────────────
        # Two continuous values in [-1, 1]:
        #   action[0] → combined thrust change (positive = go up)
        #   action[1] → differential thrust    (positive = rotate right)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # ── Observation space ────────────────────────────────────────────────
        # 7-dimensional vector (all floats, unbounded):
        #   [0] angle_to_vertical   — drone tilt in radians
        #   [1] total_velocity      — magnitude of velocity vector
        #   [2] angular_velocity    — rotation speed
        #   [3] dist_to_target_norm — Euclidean distance / 500 (normalised)
        #   [4] angle_to_target     — direction to target (radians)
        #   [5] angle_vel_vs_target — alignment of velocity with target direction
        #   [6] dist_to_target_norm — repeated for symmetry (SAC model expects this)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    # ────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        """
        Reset episode: place drone at centre, spawn a random target.
        Returns (observation, info).
        """
        super().reset(seed=seed)

        # Drone state
        self.x_pos, self.y_pos = 400.0, 400.0   # Centre of 800×800 world
        self.x_vel, self.y_vel = 0.0, 0.0
        self.angle = 0.0                         # Degrees from vertical
        self.ang_vel = 0.0

        # Target
        self._spawn_target()
        self.steps = 0
        self.score = 0

        return self._get_obs(), {}

    # ────────────────────────────────────────────────────────────────────────
    def step(self, action):
        """
        Apply one action, advance physics by one timestep.
        Returns (observation, reward, terminated, truncated, info).
        """
        self.steps += 1

        # Decode normalised action into actual thruster forces
        tl = self.thruster_mean + action[0] * self.thruster_amp + action[1] * self.diff_amp
        tr = self.thruster_mean + action[0] * self.thruster_amp - action[1] * self.diff_amp

        # ── Newton's laws of motion ──────────────────────────────────────
        ang_rad = self.angle * pi / 180
        ax = -(tl + tr) * sin(ang_rad) / self.mass
        ay = self.gravity - (tl + tr) * cos(ang_rad) / self.mass
        ang_acc = self.arm * (tr - tl) / self.mass

        # Integrate velocity and position
        self.x_vel += ax;      self.y_vel += ay;    self.ang_vel += ang_acc
        self.x_pos += self.x_vel
        self.y_pos += self.y_vel
        self.angle += self.ang_vel

        # Distance to current target
        dist = sqrt(
            (self.x_pos - self.target_x) ** 2 +
            (self.y_pos - self.target_y) ** 2
        )

        # ── Reward ───────────────────────────────────────────────────────
        # Dense shaping: small negative reward = encourages moving closer
        reward = -dist / 500.0

        terminated = False
        truncated = False

        if dist < 50:               # ✅ Target reached
            reward = 1.0
            self.score += 1
            self._spawn_target()    # New random target

        if dist > 1000:             # ❌ Out of bounds
            reward = -1.0
            terminated = True

        if self.steps >= 6000:      # ⏱ Time limit
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {"score": self.score}

    # ────────────────────────────────────────────────────────────────────────
    def _spawn_target(self):
        """Place a new target randomly within the safe zone."""
        self.target_x = float(np.random.randint(150, 650))
        self.target_y = float(np.random.randint(150, 650))

    # ────────────────────────────────────────────────────────────────────────
    def _get_obs(self):
        """Construct the 7-dimensional observation vector."""
        angle_to_up = self.angle / 180 * pi
        velocity = sqrt(self.x_vel ** 2 + self.y_vel ** 2)
        dist_norm = sqrt(
            (self.x_pos - self.target_x) ** 2 +
            (self.y_pos - self.target_y) ** 2
        ) / 500.0
        angle_to_target = np.arctan2(
            self.target_y - self.y_pos,
            self.target_x - self.x_pos
        )
        angle_vel_target = angle_to_target - np.arctan2(self.y_vel, self.x_vel)

        return np.array([
            angle_to_up,
            velocity,
            self.ang_vel,
            dist_norm,
            angle_to_target,
            angle_vel_target,
            dist_norm,           # Repeated — model was trained with this shape
        ], dtype=np.float32)

    # ────────────────────────────────────────────────────────────────────────
    def render(self):
        pass  # Rendering handled by evaluate.py via pygame
