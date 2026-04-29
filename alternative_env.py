import math
import gymnasium as gym
import numpy as np
from math import sqrt
from scipy.stats import norm
from scipy.linalg import expm
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from scipy.integrate import dblquad

class AlternativeEnv(gym.Env):
    def __init__(self, n_targets=5, n_unknown_targets=100, space_size=100.0, d_state=4, fov_size=4.0, max_steps=100):
        super().__init__()

        # ── Target counts ──────────────────────────────────────────────────────────
        # Store both the initial counts (used for resets) and mutable runtime counts
        self.init_n_targets = n_targets
        self.init_n_unknown_target = n_unknown_targets
        self.n_targets = n_targets              # Known targets (observable from the start)
        self.n_unknown_targets = n_unknown_targets  # Hidden targets discovered during episode

        # ── Environment configuration ──────────────────────────────────────────────
        self.fov_size = fov_size                # Side length of the square field-of-view (world units)
        self.space_size = space_size            # Side length of the square world (world units)
        self.d_state = d_state                  # State dimension per target: (x, y, vx, vy)
        self.max_steps = max_steps              # Maximum timesteps before episode termination
        self.dt = 1.0                           # Simulation timestep (seconds)
        self.threshold_fov = 0.5               # Fraction of FOV overlap required to count as "seen"

        # ── Episode diagnostics ────────────────────────────────────────────────────
        self.lost_counter = 0                   # Tracks how many targets were lost this episode
        self.detect_counter = 0                 # Tracks how many detections occurred this episode