import math
import gymnasium as gym
import numpy as np
from math import sqrt
from scipy.stats import norm
from scipy.linalg import expm
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib import MaskablePPO
from scipy.integrate import dblquad

class MultiTargetEnv(gym.Env):
    def __init__(self, n_targets=5, n_unknown_targets = 100, space_size=100.0, d_state=4, fov_size=4.0, max_steps=100, seed=None, mode="combined"):
        super().__init__()
        self.init_n_targets = n_targets
        self.init_n_unknown_target = n_unknown_targets
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets
        self.mode = mode
        self.fov_size = fov_size
        self.space_size = space_size
        self.d_state = d_state  # state dimension per target (e.g., x,y,vx,vy)
        self.max_steps = max_steps
        self.dt = 1.0          
        self.threshold_fov = 0.5 
        self.lost_counter = 0
        self.detect_counter = 0
        self.rng = np.random.default_rng(seed)

        # generate measurement noise covariance (2x2)
        sigma_theta = np.deg2rad(1.0)  # 1 degree bearing noise
        sigma_r = 0.1        # 10 cm range noise

        self.R = np.diag([sigma_theta**2, sigma_r**2])

        # Parameters related to dynamical motion
        self.motion_model = self.rng.choice(["L", "T"], size=self.n_targets + self.n_unknown_targets)       # L=linear motion, T=coordinated turn
        self.motion_params = self.rng.uniform(0.05, 0.3, size=self.n_targets + self.n_unknown_targets) / 10        # contains either linear velocity or turn rate

        # Cholesky size for covariance packing
        self.cholesky_size = d_state * (d_state + 1) // 2
        self.obs_dim_per_target = d_state + self.cholesky_size
        self.max_targets = self.init_n_targets + self.init_n_unknown_target

        # Default initial covariance for new tracks (make accessible as self.P0)
        self.P0 = np.eye(self.d_state) * 0.1
        self.P0[-2:, -2:] = np.eye(2) * 0.001
        self.Q0 = np.eye(self.d_state) * 1e-1

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

        # Initialise maps as 2D
        self.detection_history = np.zeros((n_grid, n_grid), dtype=np.float32)
        self.recency_map = np.zeros((n_grid, n_grid), dtype=np.float32)
        self.roi_mask = self._build_roi_mask()

        """ # OBSERVATION: all targets (max_targets) * per-target obs + mask_length
        obs_len = self.max_targets * self.obs_dim_per_target + self.max_targets
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(obs_len,), dtype=np.float32) """
        if mode == "search":
            # Observation is just visit counts
            obs_len = self.n_grid_cells
            """ self.observation_space = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(obs_len,),
                dtype=np.float32
            ) """
                        # Shape becomes (4, n_grid, n_grid) — four information channels
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=(4, n_grid, n_grid),
                dtype=np.float32
            )

        # New per-target dimension:
        if mode == "track":
            self.obs_dim_per_target = 1   # dx, dy, vx, vy, p_fov, known_mask
            #self.obs_dim_per_target = 3

            #obs_len = self.init_n_target * self.obs_dim_per_target
            obs_len = self.max_targets * self.obs_dim_per_target
            self.observation_space = gym.spaces.Box(
                low=-1.0, 
                high=1.0,
                shape=(obs_len,),
                dtype=np.float32
            )

        # ACTION space (flat form)
        if self.mode == "search":
            # Only search actions (one per grid cell)
            self.n_actions = self.n_grid_cells
        elif self.mode == "track":
            # Only track actions (one per known target)
            #self.n_actions = self.n_targets
            self.n_actions = self.max_targets
        else:  # "combined"
            # Full range: all search + all track
            self.n_actions = self.n_grid_cells + self.max_targets

        self.action_space = gym.spaces.Discrete(self.n_actions)

        # known mask
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True

        self.reset(seed=seed)

    def _build_roi_mask(self):
        n_grid = int(np.sqrt(self.n_grid_cells))
        mask = np.zeros((n_grid, n_grid), dtype=np.float32)
        for grid_idx, search_pos in enumerate(self.grid_coords):
            x_index = grid_idx // n_grid
            y_index = grid_idx % n_grid
            if search_pos[1] > 0:
                mask[x_index, y_index] = 1.0
        return mask
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.n_targets = self.init_n_targets
        self.n_unknown_targets = self.init_n_unknown_target
        self.lost_counter = 0
        self.detect_counter = 0

        #reset dynamics
        self.motion_model = self.rng.choice(["L", "T"], size=self.n_targets + self.n_unknown_targets)       # L=linear motion, T=coordinated turn
        self.motion_params = self.rng.uniform(0.05, 0.3, size=self.n_targets + self.n_unknown_targets) 

        self.targets = [self._init_target(i) for i in range(self.init_n_targets)]
        self.unknown_targets = [
            self._init_unknown_target(i + self.init_n_targets)
            for i in range(self.init_n_unknown_target)
        ]
        self.visit_counts[:] = 0   # reset search visit counts
        self.detection_history[:] = 0.0
        self.recency_map[:] = 0.0

        self.episode_detection_reward = 0.0
        self.episode_roi_reward = 0.0
        self.episode_n_detections = 0

        # Initialize search memory variables here
        self.last_search_idx = None
        self.prev_search_pos = None

        # known mask
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True

        """ if self.mode=="track":
            self.obs = np.full(5, -1e9, dtype=np.float32)
        else: """
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
    
    def encode_action(self, macro, micro_search=None, micro_track=None):
        """
        Convert hierarchical action (macro + micro) into a flat integer.
        Mirrors decode_action().
        
        Parameters:
            macro: 0 for search, 1 for track
            micro_search: index of search grid (if macro==0)
            micro_track: target index (if macro==1)
        
        Returns:
            action_int: single integer representing the action
        """
        if macro == 0:  # SEARCH
            if micro_search is None:
                raise ValueError("micro_search must be provided for macro=0")
            return micro_search
        elif macro == 1:  # TRACK
            if micro_track is None:
                raise ValueError("micro_track must be provided for macro=1")
            if self.mode == "track":
                return micro_track
            elif self.mode == "combined":
                return self.n_grid_cells + micro_track
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        else:
            raise ValueError(f"Invalid macro action: {macro}")

    def action_masks(self) -> np.ndarray:
        """
        Return a boolean array where True = valid action, False = invalid.
        Required by MaskablePPO.
        """
        if self.mode == "search":
            # All search grid cells are always valid
            return np.ones(self.n_actions, dtype=bool)
        
        elif self.mode == "track":
            # Only known targets are valid
            return self.known_mask.copy()
        
        else:  # "combined"
            # First n_grid_cells actions = search (always valid)
            # Next max_targets actions = track (valid if known)
            mask = np.ones(self.n_actions, dtype=bool)
            mask[self.n_grid_cells:] = self.known_mask
            return mask
        
    def step(self, action):
        macro, micro_search, micro_track = self.decode_action(action)
        micro = [0., 0.]
        target_id = []
        search_pos = None
        n_grid = int(np.sqrt(self.n_grid_cells))

        if macro == 0:  # SEARCH
            grid_idx = micro_search
            search_pos = self.grid_coords[grid_idx]
            micro = search_pos
        else:  # TRACK
            target_id.append(micro_track) 
            micro = micro_track

        # Initialise reward
        total_iG = 0.0

        # reward related to neglected known targets
        lost_reward = 0.0
        prob_reward = 0.0
        lost = 0
        lost_targets = []

        # --- Catastrophic failure: a target was lost ---
        """ if self.n_targets < self.init_n_target:
            obs = self._get_obs()
            self.step_count += 1

            reward = -1.0   
            done = True
            truncated = False

            info = {
                "catastrophic_loss": True,
                "remaining_targets": self.n_targets,
                "action_mask": self.get_action_mask()
            }

            self.obs = obs
            return obs, reward, done, truncated, info """

        # propagate all known targets
        for tgt in self.targets.copy():

            idx = tgt['id']  # global index
            model = self.motion_model[idx]
            param = self.motion_params[idx]

            # retrieve predictions by propagating each known target
            tgt['x'], tgt['P'] = MultiTargetEnv.propagate_target_2D(tgt['x'], tgt['P'], tgt.get('Q', self.Q0), dt=self.dt, rng=self.rng, motion_model=model, motion_param=param)

            # compute measurement related to action
            if target_id and idx == micro:
                xUpdate, PUpdate = MultiTargetEnv.ekf_update(tgt['x'], tgt['P'], self.R, MultiTargetEnv.extract_measurement_bearingRange)
                iG = MultiTargetEnv.compute_kl_divergence(tgt['x'], tgt['P'], xUpdate, PUpdate)
                probSimple = compute_fov_prob_single(self.fov_size, tgt['x'], tgt['P'])
                prob = MultiTargetEnv.compute_fov_prob_full(tgt['P'], self.fov_size, self.fov_size)
                #print(prob)
                #print(probFull)
                #print(probFull - prob)
                # Update
                tgt['x'], tgt['P'] = xUpdate, PUpdate
                #total_iG = iG
                
                probSimple = compute_fov_prob_single(self.fov_size, tgt['x'], tgt['P'])
                prob = MultiTargetEnv.compute_fov_prob_full(tgt['P'], self.fov_size, self.fov_size)
                #print(prob)
                #print(probFull)
                #print(probFull - prob)
                prob_reward += prob
                lost = 1 - prob
                if lost>total_iG:
                    total_iG = lost_reward

            # Otherwise: compute FOV-probability reward for this neglected target
            else:
                prob = MultiTargetEnv.compute_fov_prob_full(tgt['P'], self.fov_size, self.fov_size)
                probSimple = compute_fov_prob_single(self.fov_size, tgt['x'], tgt['P'])
                #print(1-prob)
                """ if (1-prob)>total_iG:
                    total_iG = 1-prob
                prob_reward += prob """
                if prob<self.threshold_fov:
                    # target is considered as lost
                    lost_reward = lost_reward-0.25
                    lost_targets.append(tgt)
                    self._remove_lost_tracking_target(idx)
                else:
                    prob_reward += prob


        # propagate unknowns
        for utgt in self.unknown_targets:

            idx = utgt['id']
            model = self.motion_model[idx]
            param = self.motion_params[idx]

            # retrieve predictions by propagating each unknown target
            utgt['x'], utgt['P'] = MultiTargetEnv.propagate_target_2D(utgt['x'], utgt['P'], utgt.get('Q', self.Q0), dt=self.dt, rng=self.rng, motion_model=model, motion_param=param)

        # If TRACK macro, return here            
        if target_id:       
            # Construct next observation
            obs = self._get_obs(target_id)
            self.step_count += 1

            # Check validity
            if not self.known_mask[target_id]:
                invalid_action = True
            else:
                invalid_action = False
                """ # Invalid track (e.g., unknown target) -->  return with action mask so agent can correct
                info = {"invalid_action": True, "action_mask": self.get_action_mask(), "lost_target": lost_targets}
                # Termination
                done = self.step_count >= self.max_steps
                return obs, -1.0, done, False, info """

            # Simple reward = total information gain (KL) or punishment for loosing target
            """ if lost_reward<0:
                reward = lost_reward
            else:
                reward = total_iG / 1e3   # scale down
            
            if self.n_targets<self.init_n_target:
                reward = reward - 5 * (self.init_n_target-self.n_targets) """
            """ if lost == total_iG:
                reward = 1
            else:
                reward = -1
            if self.n_targets == self.init_n_target:
                reward += 1 """
            reward = prob_reward
            """ # Penalty for losing targets this step
            if lost_targets:
                obs = self._get_obs()
                self.step_count += 1

                reward = -1.0   
                done = True
                truncated = False

                info = {
                    "catastrophic_loss": True,
                    "remaining_targets": self.n_targets,
                    "action_mask": self.get_action_mask()
                }

                self.obs = obs
                return obs, reward, done, truncated, info """

            # Termination
            done = self.step_count >= self.max_steps
            truncated = False

            # Info dict can include diagnostics
            info = {
                "macro": macro,
                "micro": micro,
                "target_id": target_id,
                "reward_info_gain": total_iG,
                "action_mask": self.get_action_mask(),
                "lost_target": lost_targets,
                "n_known": np.sum(self.known_mask),  # Track this!
                "n_lost_this_step": len(lost_targets),
                "detect_counter": self.detect_counter,
                "lost_counter": self.lost_counter
            }
            self.obs = obs

            return obs, reward, done, truncated, info

        # SEARCH macro: Check detections
        detections = []
        fov_halfWidth = self.fov_size / 2.0
        for obj in self.targets + self.unknown_targets:

            try:
                dx = obj['x'][0] - search_pos[0]
                dy = obj['x'][1] - search_pos[1]
                if abs(dx) <= fov_halfWidth and abs(dy) <= fov_halfWidth:
                    detections.append(obj)
            except Exception as e:
                # Catch internal failure
                obs = self._get_obs() if hasattr(self, "_get_obs") else self.observation_space.sample()

                reward = -100.0  # strong penalty for failure
                terminated = True
                truncated = False

                info = {
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                }

                return obs, reward, terminated, truncated, info

        # Compute reward
        # Convert flat grid_idx to 2D coordinates
        x_index = grid_idx // n_grid
        y_index = grid_idx % n_grid

        if len(detections) > 0:
            threshold = 3.0  # example Mahalanobis threshold
            for det in detections:
                distances = [
                    mahalanobis_distance(det['x'], known['x'], known['P'])
                    for known in self.targets
                ]
                # If all distances > threshold --> new detection
                if all(d > threshold for d in distances):
                    total_iG += 10.0
                    # Promote detection to a new known target
                    self._add_new_tracking_target(det['id'])
                    target_id.append(det['id'])
                    self.detection_history[x_index, y_index] = min(
                        self.detection_history[x_index, y_index] + 1.0, 1.0
                    )
                    
        # --- SEARCH macro: update visit counts and compute reward ---
        # Update recency map — decay everything, then mark current location
        self.recency_map *= 0.99
        self.recency_map[x_index, y_index] = 1.0

        # Update detection history — decay everything
        self.detection_history *= 0.95
        self.visit_counts[grid_idx] += 1
        reward = total_iG
        """ exploration_bonus = 1.0 / np.log(self.visit_counts[grid_idx] + 2)

        # Penalize staying still
        if hasattr(self, "last_search_idx") and self.last_search_idx == grid_idx:
            exploration_bonus -= 0.5 """
        self.last_search_idx = grid_idx
        self.prev_search_pos = search_pos

        # --- Upper-half bonus ---
        exploration_bonus = 0
        if search_pos[1] > 0:  # cell in upper half
            exploration_bonus += 1.0   

        #reward += exploration_bonus
        # Accumulate episode stats
        self.episode_detection_reward += total_iG
        self.episode_roi_reward += exploration_bonus
        self.episode_n_detections += len(detections)

        # Construct next observation
        obs = self._get_obs()

        # Termination
        self.step_count += 1
        done = (self.step_count >= self.max_steps) #or (reward > 9)
        truncated = False

        # Info dict can include diagnostics
        info = {"macro": macro, "micro": micro, 
                "target_id": target_id, 
                "reward_info_gain": total_iG, 
                "action_mask": self.get_action_mask(), 
                "lost_target": lost_targets, 
                "episode_detection_reward": self.episode_detection_reward,
                "episode_roi_reward": self.episode_roi_reward,
                "episode_n_detections": self.episode_n_detections}
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
        vel0 = np.array([self.motion_params[target_id], 0.0])
        x0 = np.concatenate([pos0, vel0])
        covMultiplier = [0.7, 0.75, 0.85, 1.0]
        P0 = self.P0.copy() * np.random.choice(covMultiplier)

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
        vel0 = np.array([1.0, 0.0])         # velocity for CT model
        if self.motion_model[target_id] == "L":
            vel0 = np.array([self.motion_params[target_id], 0.0])  # move rightward
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

    def _get_obs(self, target_id=None):
        """Flatten all target states and covariances into one observation vector."""
        if self.mode=="search":
            n_grid = int(np.sqrt(self.n_grid_cells))
            # Channel 1 — normalised visit counts
            visit_2d = self.visit_counts.reshape(n_grid, n_grid).astype(np.float32)
            visit_map = np.log1p(visit_2d) / np.log1p(visit_2d.max() + 1)
            
            # Channel 2 — recent detection map (decaying sum of past detections)
            # Set to 1.0 where detection occurred, decay by 0.95 each step
            detection_map = self.detection_history.copy()

            # Channel 3 — recency map (1.0 = pointed here last step, decays over time)
            recency_map = self.recency_map.copy()

            # Channel 4 — ROI mask (static, 1.0 inside ROI, 0.0 outside)
            roi_map = self.roi_mask.copy()

            return np.stack([
                visit_map,                  
                detection_map,    
                recency_map,           
                roi_map,              
            ], axis=0).astype(np.float32)
            #return self.visit_counts.astype(np.float32)
            """ all_targets = self.targets + self.unknown_targets
            obs = []
            for tgt in all_targets:
                obs.append(np.concatenate([tgt['x'], tgt['P'][np.tril_indices(self.d_state)]]))
            obs_vec = np.concatenate(obs)
            mask_vec = self.known_mask.astype(np.float32)
            output = np.concatenate([obs_vec, mask_vec])
            return output """

        """Return a compact, RL-observation vector."""
        """ all_targets = [] 
        for i in range(self.max_targets):
            for tgt_known in self.targets:
                if tgt_known["id"] == i:
                    all_targets.append(tgt_known)
                    break
            for tgt_unknown in self.unknown_targets:
                if tgt_unknown["id"] == i:
                    all_targets.append(tgt_unknown)
                    break """
        by_id = {t["id"]: t for t in self.targets}
        by_id.update({t["id"]: t for t in self.unknown_targets})

        all_targets = [by_id[k] for k in sorted(by_id)]

            
        obs_list = []

        for i, tgt in enumerate(all_targets):
            x, y, vx, vy = tgt["x"]
            trace = np.trace(tgt["P"])
            p_fov = MultiTargetEnv.compute_fov_prob_full(tgt['P'], self.fov_size, self.fov_size)
            #p_fov = compute_fov_prob_single(self.fov_size, tgt["x"], tgt["P"])
            known = 1.0 if self.known_mask[tgt["id"]] else 0.0
            
            # Provide more informative features:
            obs_list.extend([
                trace * known#,      # Uncertainty (your current feature)
                #p_fov * known,      # Risk of losing this target (NEW!)
                #known               # Is this target tracked? (NEW!)
            ])
        
        return np.array(obs_list, dtype=np.float32)
        
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

        # remove from unknown list
        for i, element in enumerate(self.unknown_targets):
            if element.get('id') == unknown_idx:
                new = self.unknown_targets.pop(i)
                break
        else:
            new = None   # not found
        #new = self.unknown_targets.pop(unknown_idx-self.init_n_target)
        new_target = {
            "id": unknown_idx,
            "x": new['x'].copy(),
            "P": self.P0.copy(),
            "Q": new.get('Q', self.Q0)
        }
        self.targets.append(new_target)
        self.known_mask[unknown_idx] = True
        self.n_targets += 1
        self.n_unknown_targets -= 1
        self.detect_counter += 1

    def _remove_lost_tracking_target(self, target_id):
        """Move a known target back into unknown targets when it is lost."""
    
        # Find the matching target in the known list by ID
        for i, tgt in enumerate(self.targets):
            if tgt['id'] == target_id:
                
                # Remove from known list
                removed = self.targets.pop(i)

                # Move to unknown list (keep its covariance!)
                self.unknown_targets.append({
                    "id": target_id,
                    "x": removed["x"].copy(),
                    "P": removed["P"].copy(),
                    "Q": removed.get("Q", self.Q0)
                })

                # Update mask and counters
                self.known_mask[target_id] = False
                self.n_targets -= 1
                self.n_unknown_targets += 1
                self.lost_counter += 1
                return

    @staticmethod
    def propagate_target_2D(x, P, Q, dt, rng, motion_model="L", motion_param=1.0):
        """
        Propagate a 2D target state based on its motion model.

        Parameters:
            x : np.array
                State vector [x, y, vx, vy]
            P : np.array
                Covariance matrix
            Q : np.array
                Process noise covariance
            dt : float
                Time step
            rng : np.random.Generator
                Random number generator for stochastic noise
            motion_model : str
                "L" = linear, "T" = coordinated turn
            motion_param : float
                Linear velocity for "L", turn rate for "T"

        Returns:
            x_new, P_new : propagated state and covariance
        """
        F = None
        if motion_model == "L":
            F = MultiTargetEnv.constant_velocity_F_2D(dt)
        else:
            F = MultiTargetEnv.constant_turnrate_F_2D(dt, motion_param)
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
    def constant_turnrate_F_2D(dt, omega):
        """
        Build 4x4 F for state ordering [x, y, vx, vy].
        """

        #modify omega to increase radius of targets
        #omega = omega / 10
        wdt = omega * dt
        # handle small w via stable evaluations:
        if abs(wdt) < 1e-8:
            # Use Taylor expansions directly for the needed terms:
            # s1 = sin(wdt)/omega  ~= dt * (1 - z^2/6 + z^4/120)
            # s2 = (1-cos(wdt))/omega ~= dt*(z/2 - z^3/24 + ...)
            z = wdt
            s1 = dt * (1.0 - z*z/6.0 + z**4/120.0)
            s2 = dt * (z/2.0 - z**3/24.0 + z**5/720.0)
        else:
            s1 = np.sin(wdt) / omega
            s2 = (1.0 - np.cos(wdt)) / omega

        c = np.cos(wdt)
        s = np.sin(wdt)

        F = np.array([
            [1.0, 0.0,      s1,         s2],
            [0.0, 1.0,     -s2,         s1],
            [0.0, 0.0,       c,        -s ],
            [0.0, 0.0,       s,         c ]
        ])
        return F
    
    @staticmethod
    def extract_measurement_bearingRange(x):
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
        Gk = np.array([theta, r])
        return H, Gk
    
    @staticmethod
    def extract_measurement_XY(x):
        posX = x[0]
        posY = x[1]
        Gk = np.array([posX, posY])

        # Full Jacobian (2x4, assuming state = [x, y, vx, vy])
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],  # x partials
            [0.0, 1.0, 0.0, 0.0]   # y partials
        ])
        return H, Gk

    
    def ekf_update(x, P, R, obsFcn):
        """
        Perform one EKF measurement update.
        Inputs:
            x: state vector (4,)
            P: covariance matrix (4x4)
            R: measurement noise covariance (2x2)
        Returns:
            x_upd, P_upd
        """
        # Predict measurement
        H, Gk = obsFcn(x)
        
        # Innovation
        y = np.zeros_like(Gk)
        
        # Innovation covariance
        S = H @ P @ H.T + R
        
        # Kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        
        # Updated state and covariance
        x_upd = x + K @ y
        P_upd = (np.eye(len(x)) - K @ H) @ P
        
        return x_upd, P_upd

    @staticmethod
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

    @staticmethod
    def compute_fov_prob_full(P, fov_x, fov_y=None, rtol=1e-8):
        """
        Compute probability that a 2D Gaussian state remains inside a rectangular FOV
        using full covariance integration (no independence assumption).
        This function can be used as well to approximate the FOV probability for a sensor 
        located in Earth center pointing towards geostationary satellites.

        This function should not be used for large FOV and for satellites with large
        declination angles. 

        Parameters
        ----------
        P : 2x2 numpy array
            Position covariance matrix
        fov_x : float
            Full FOV width in x
        fov_y : float, optional
            Full FOV width in y (defaults to fov_x)
        rtol : float
            Relative tolerance for quadrature

        Returns
        -------
        prob : float
            Probability of remaining inside the FOV
        """

        if fov_y is None:
            fov_y = fov_x

        hx = fov_x / 2.0
        hy = fov_y / 2.0

        # slice P to only take into account positional/angular values
        P2 = P[0:2, 0:2]
        w, _ = np.linalg.eig(P2)

        # Ensure positive definiteness if needed
        detP = np.linalg.det(P2)
        if detP <= 0:
            raise ValueError("Covariance matrix must be positive definite")

        Pinv = np.linalg.inv(P2)

        # Gaussian exponent
        def integrand(y, x):
            quad = (
                Pinv[0, 0] * x**2
                + (Pinv[0, 1] + Pinv[1, 0]) * x * y
                + Pinv[1, 1] * y**2
            )
            return math.exp(-0.5 * quad)

        atol = 1e-13

        integral = dblquad(
            integrand,
            -hx, hx,
            lambda x: -hy,
            lambda x:  hy,
            epsabs=atol,
            epsrel=rtol
        )[0]

        prob = (1.0 / (2.0 * math.pi * math.sqrt(detP))) * integral
        #print(f"{integral}; {detP}; {prob}; {w}")
        return prob

def mahalanobis_distance(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff.T @ np.linalg.inv(cov) @ diff)

def compute_fov_prob_single(fov, x, P):
    """
    Compute FOV-probability analytically under the assumption that the integration 
    axes are independent.
    """

    #half_fov = fov*0.75 / 2.0   # radians 
    half_fov = fov / 2.0   # radians 

    prob = 1.0
    for i in range(2):
        pos_var = P[i, i]
        pos_std = sqrt(pos_var)

        # Numerical safety
        if pos_std < 1e-8:
            pos_std = 1e-8

        # Probability that target's x-pos ∈ [-half_fov, +half_fov] ---
        # Gaussian N(0, σ)
        dist = norm(loc=0.0, scale=pos_std)
        prob *= dist.cdf(half_fov) - dist.cdf(-half_fov)
    return prob


    