import gymnasium as gym
import numpy as np
from scipy.linalg import expm

class MultiTargetEnv(gym.Env):
    def __init__(self, n_targets=5, n_unknown_targets = 3, space_size=100.0, d_state=4, fov_size=2.0, max_steps=50, seed=None, mode="combined"):
        super().__init__()
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets
        self.mode = mode
        self.fov_size = fov_size
        self.space_size = space_size
        self.d_state = d_state  # state dimension per target (e.g., x,y,vx,vy)
        self.max_steps = max_steps
        self.dt = 1.0
        self.velocity = 1.0
        self.rng = np.random.default_rng(seed)

        # Cholesky size for covariance packing
        self.cholesky_size = d_state * (d_state + 1) // 2
        self.obs_dim_per_target = d_state + self.cholesky_size
        self.max_targets = self.n_targets + self.n_unknown_targets

        # Default initial covariance for new tracks (make accessible as self.P0)
        self.P0 = np.eye(self.d_state) * 0.5
        self.Q0 = np.eye(self.d_state) * 0.0

        # Discretising entire field 
        n_grid = max(1, int(np.floor(self.space_size / self.fov_size)))
        self.n_grid_cells = n_grid * n_grid

        x_vals = np.linspace(-self.space_size / 2 + self.fov_size / 2,
                            self.space_size / 2 - self.fov_size / 2,
                            n_grid)
        y_vals = np.linspace(-self.space_size / 2 + self.fov_size / 2,
                            self.space_size / 2 - self.fov_size / 2,
                            n_grid)
        self.grid_coords = np.array([[x, y] for x in x_vals for y in y_vals])
        self.visit_counts = np.zeros(self.n_grid_cells, dtype=int)

        # OBSERVATION: all targets (max_targets) * per-target obs + mask_length
        obs_len = self.max_targets * self.obs_dim_per_target + self.max_targets
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_len,), dtype=np.float32)

        # ACTION space (flat form)
        if self.mode == "search":
            # Only search actions (one per grid cell)
            self.n_actions = self.n_grid_cells
        elif self.mode == "track":
            # Only track actions (one per known target)
            self.n_actions = self.n_targets
        else:  # "combined"
            # Full range: all search + all track
            self.n_actions = self.n_grid_cells + self.max_targets

        self.action_space = gym.spaces.Discrete(self.n_actions)

        # known mask
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True

        # reward window
        self.reward_window_size = self.space_size / 2
        self.reward_window_speed = self.velocity

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.targets = [self._init_target(i) for i in range(self.n_targets)]
        self.unknown_targets = [
            self._init_unknown_target(i + self.n_targets)
            for i in range(self.n_unknown_targets)
        ]
        self.visit_counts[:] = 0   # reset search visit counts

        # Initialize search memory variables here
        self.last_search_idx = None
        self.prev_search_pos = None

        # Reset reward window
        self.reward_window_center = np.array([-self.space_size/4, self.space_size/4])
        self.reward_window_history = [self.reward_window_center.copy()]

        self.obs = self._get_obs()
        info = {}  # optional
        return self.obs, info
    
    # Action decoding logic 
    def decode_action(self, action_int):
        """Convert flat integer into (macro, micro_search, micro_track)."""
        if self.mode == "search":
            macro = 0
            micro_search = action_int
            micro_track = None
        elif self.mode == "track":
            macro = 1
            micro_search = None
            micro_track = action_int
        else:  # combined
            if action_int < self.n_grid_cells:
                macro = 0
                micro_search = action_int
                micro_track = None
            else:
                macro = 1
                micro_search = None
                micro_track = action_int - self.n_grid_cells
        return macro, micro_search, micro_track

    def step(self, action):
        macro, micro_search, micro_track = self.decode_action(action)
        micro = [0., 0.]
        target_id = None
        search_pos = None

        if macro == 0:  # SEARCH
            grid_idx = micro_search
            search_pos = self.grid_coords[grid_idx]
            micro = search_pos
        else:  # TRACK
            target_id = micro_track 
            # Check validity
            if not self.known_mask[target_id]:
                # Invalid track (e.g., unknown target) -->  return with action mask so agent can correct
                obs = self._get_obs()
                info = {"invalid_action": True, "action_mask": self.get_action_mask()}
                return obs, -1.0, False, False, info
            micro = target_id

        # Initialise reward
        total_iG = 0.0

        # Move reward window to the right with each step (same as target dynamics)
        self.reward_window_center[0] += self.reward_window_speed * self.dt
        self.reward_window_history.append(self.reward_window_center.copy())

        # generate measurement noise covariance (2x2)
        sigma_theta = np.deg2rad(1.0)  # 1 degree bearing noise
        sigma_r = 0.1        # 10 cm range noise

        R = np.diag([sigma_theta**2, sigma_r**2])

        # propagate all known targets
        for tgt in self.targets:

            # retrieve predictions by propagating each known target
            tgt['x'], tgt['P'] = MultiTargetEnv.propagate_target_2D(tgt['x'], tgt['P'], tgt.get('Q', self.Q0), dt=self.dt, rng=self.rng)

            # compute measurement related to action
            if target_id is not None and tgt['id'] == target_id:
                xUpdate, PUpdate = MultiTargetEnv.ekf_update(tgt['x'], tgt['P'], R)
                iG = compute_kl_divergence(tgt['x'], tgt['P'], xUpdate, PUpdate)
                
                # Update
                tgt['x'], tgt['P'] = xUpdate, PUpdate
                total_iG += iG

        # propagate unknowns
        for utgt in self.unknown_targets:
            # retrieve predictions by propagating each unknown target
            utgt['x'], utgt['P'] = MultiTargetEnv.propagate_target_2D(utgt['x'], utgt['P'], utgt.get('Q', self.Q0), dt=self.dt, rng=self.rng)

        # If TRACK macro, return here
        if target_id is not None:       
            
            # Construct next observation
            obs = self._get_obs()

            # Simple reward = total information gain (KL)
            reward = total_iG / 1e3   # scale down

            # Termination
            self.step_count += 1
            done = self.step_count >= self.max_steps
            truncated = False

            # Info dict can include diagnostics
            info = {"macro": macro, "micro": micro, "target_id": target_id, "reward_info_gain": total_iG, "action_mask": self.get_action_mask()}
            self.obs = obs

            return obs, reward, done, truncated, info

        # SEARCH macro: Check detections
        detections = []
        fov_halfWidth = self.fov_size / 2.0
        for obj in self.targets + self.unknown_targets:
            dx = obj['x'][0] - search_pos[0]
            dy = obj['x'][1] - search_pos[1]
            if abs(dx) <= fov_halfWidth and abs(dy) <= fov_halfWidth:
                detections.append(obj)

        

        # Compute reward
        if len(detections) > 0:
            threshold = 3.0  # example Mahalanobis threshold
            for det in detections:
                distances = [
                    mahalanobis_distance(det['x'], known['x'], known['P'])
                    for known in self.targets
                ]
                # If all distances > threshold --> new detection
                if all(d > threshold for d in distances):
                    total_iG = 10.0
                    # Promote detection to a new known target
                    idx = self.unknown_targets.index(det)
                    self._add_new_tracking_target(idx)
                    break

        # --- SEARCH macro: update visit counts and compute reward ---
        self.visit_counts[grid_idx] += 1
        reward = total_iG
        exploration_bonus = 5.0 / np.log(self.visit_counts[grid_idx] + 2)

        # Penalize staying still
        if hasattr(self, "last_search_idx") and self.last_search_idx == grid_idx:
            exploration_bonus -= 0.5
        self.last_search_idx = grid_idx
        self.prev_search_pos = search_pos

        # --- Upper-half bonus ---
        if search_pos[1] > 0:  # cell in upper half
            exploration_bonus += 5.0   

        reward += exploration_bonus

        # --- Reward window bonus (uniform inside window) ---
        half_w = self.reward_window_size / 2.0
        x_in_window = abs(search_pos[0] - self.reward_window_center[0]) <= half_w
        y_in_window = abs(search_pos[1] - self.reward_window_center[1]) <= half_w

        if x_in_window and y_in_window:
            window_bonus = 5.0  # uniform reward inside window
        else:
            window_bonus = 0.0

        reward += window_bonus

        # Construct next observation
        obs = self._get_obs()

        # Termination
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False

        # Info dict can include diagnostics
        info = {"macro": macro, "micro": micro, "target_id": target_id, "reward_info_gain": total_iG, "action_mask": self.get_action_mask()}
        self.obs = obs

        return obs, reward, done, truncated, info

    def _init_target(self, target_id,  y_range=None):
        """Initialize target with random position in left half and fixed velocity."""
        if y_range is None:
            y_low, y_high = -self.space_size/2, self.space_size/2
        else:
            y_low, y_high = y_range

        # Sample x exclusively in left half: [-space_size/2, 0)
        x0_left = self.rng.uniform(-self.space_size/2, 0)
        y0 = self.rng.uniform(y_low, y_high)

        pos0 = np.array([x0_left, y0])
        vel0 = np.array([self.velocity, 0.0])
        x0 = np.concatenate([pos0, vel0])
        P0 = self.P0.copy()
        Q = np.eye(self.d_state) * 0.
        return {"id": target_id, "x": x0, "P": P0, "Q": Q}
    
    def _init_unknown_target(self, target_id, x_range=None, y_range=None):
        if x_range is None:
            # Left half of the field
            x_low, x_high = -self.space_size/2, 0
        else:
            x_low, x_high = x_range
        if y_range is None:
            # Upper half of the field
            y_low, y_high = 0, self.space_size/2
        else:
            y_low, y_high = y_range

        x0 = self.rng.uniform(x_low, x_high)
        y0 = self.rng.uniform(y_low, y_high)
        pos0 = np.array([x0, y0])
        vel0 = np.array([self.velocity, 0.0])  # move rightward
        x0_full = np.concatenate([pos0, vel0])
        P0 = self.P0.copy()
        Q = np.eye(self.d_state) * 0.0
        return {"id": target_id, "x": x0_full, "P": P0, "Q": Q}
    
    def get_action_mask(self):
        """Return dictionary of valid discrete actions."""
        mask = {
            "macro": np.ones(2, dtype=bool),  # always can choose search or track
            "micro_search": np.ones(self.n_grid_cells, dtype=bool),  # all grid cells valid
            "micro_track": self.known_mask.copy()  # only known targets valid
        }
        return mask

    def _get_obs(self):
        """Flatten all target states and covariances into one observation vector."""
        all_targets = self.targets + self.unknown_targets
        obs = []
        for tgt in all_targets:
            obs.append(np.concatenate([tgt['x'], tgt['P'][np.tril_indices(self.d_state)]]))
        obs_vec = np.concatenate(obs)
        mask_vec = self.known_mask.astype(np.float32)
        return np.concatenate([obs_vec, mask_vec])
        
    def sample_track_action(self):
        """Sample a valid tracking action from currently known targets."""
        valid_ids = np.where(self.known_mask)[0]
        if len(valid_ids) == 0:
            # fallback: no known targets yet, pick random valid grid cell instead
            return {"macro": 0, "micro_search": int(self.rng.integers(self.n_grid_cells)), "micro_track": 0}
        target_id = int(self.rng.choice(valid_ids))
        return {"macro": 1, "micro_search": 0, "micro_track": target_id}
    
    def _add_new_tracking_target(self, unknown_idx):
        """Promote unknown_targets[unknown_idx] into known targets."""
        if len(self.targets) >= self.max_targets:
            return
        new = self.unknown_targets.pop(unknown_idx)
        new_id = len(self.targets)
        new_target = {
            "id": new_id,
            "x": new['x'].copy(),
            "P": self.P0.copy(),
            "Q": new.get('Q', self.Q0)
        }
        self.targets.append(new_target)
        self.known_mask[new_id] = True

    @staticmethod
    def propagate_target_2D(x, P, Q, dt, rng):
        """Propagate 2D constant velocity target state and covariance."""
        F = MultiTargetEnv.constant_velocity_F_2D(dt)
        w = rng.multivariate_normal(np.zeros(len(x)), Q)
        x_next = F @ x + w
        P_next = F @ P @ F.T + Q
        return x_next, P_next
    
    @staticmethod
    def constant_velocity_F_2D(dt):
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def extract_measurement(x):
        # Access first two entries (x, y)
        px, py = x[:2]

        # Compute range and bearing
        theta = np.arctan2(py, px)
        r = np.sqrt(px**2 + py**2)

        # Compute observation matrix
        H11 = -py / (px**2 + py**2)
        H12 =  px / (px**2 + py**2)
        H21 =  px / r
        H22 =  py / r

        # Full Jacobian (2x4, assuming state = [x, y, vx, vy])
        H = np.array([
            [H11, H12, 0.0, 0.0],  # Bearing partials
            [H21, H22, 0.0, 0.0]   # Range partials
        ])
        return theta, r, H
    
    
    def ekf_update(x, P, R):
        """
        Perform one EKF measurement update (range-bearing).
        Inputs:
            x: state vector (4,)
            P: covariance matrix (4x4)
            R: measurement noise covariance (2x2)
        Returns:
            x_upd, P_upd
        """
        # Predict measurement
        theta, r, H = MultiTargetEnv.extract_measurement(x)
        
        # Innovation
        y = np.array([0.0, 0.0])
        
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        
        # Updated state and covariance
        x_upd = x + K @ y
        P_upd = (np.eye(len(x)) - K @ H) @ P
        
        return x_upd, P_upd

def compute_kl_divergence(mean_p, cov_p, mean_q, cov_q):
    """
    Compute the Kullback–Leibler divergence D_KL(P || Q) between two multivariate Gaussians.

    Parameters
    ----------
    mean_p : np.ndarray
        Mean vector of distribution P (n,)
    cov_p : np.ndarray
        Covariance matrix of distribution P (n x n)
    mean_q : np.ndarray
        Mean vector of distribution Q (n,)
    cov_q : np.ndarray
        Covariance matrix of distribution Q (n x n)

    Returns
    -------
    float
        The KL divergence D_KL(P || Q)
    """
    n = mean_p.shape[0]

    # Ensure inputs are NumPy arrays
    mean_p = np.atleast_1d(mean_p)
    mean_q = np.atleast_1d(mean_q)
    cov_p = np.atleast_2d(cov_p)
    cov_q = np.atleast_2d(cov_q)

    # Compute inverses and determinants
    inv_cov_q = np.linalg.inv(cov_q)
    det_cov_p = np.linalg.det(cov_p)
    det_cov_q = np.linalg.det(cov_q)

    # Compute trace term
    trace_term = np.trace(inv_cov_q @ cov_p)

    # Mean difference
    mean_diff = mean_q - mean_p
    mean_term = mean_diff.T @ inv_cov_q @ mean_diff

    # Log determinant ratio term
    log_det_term = np.log(det_cov_q / det_cov_p)

    # Combine terms (0.5 * [log|Σq|/|Σp| - n + Tr(invΣq Σp) + (μq - μp)^T invΣq (μq - μp)])
    d_kl = 0.5 * (log_det_term - n + trace_term + mean_term)

    return float(d_kl)

def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)