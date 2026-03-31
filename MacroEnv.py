import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

from sb3_contrib import MaskablePPO

from deterministic_tracker import select_best_action_pFOV, select_best_action_sumTrace
from multi_target_env import MultiTargetEnv, compute_fov_prob_single

def unwrap_env(env):
        # Monitor wrapper
        if hasattr(env, "env"):
            return unwrap_env(env.env)
        # VecEnv wrapper (DummyVecEnv, SubprocVecEnv)
        if hasattr(env, "venv"):
            return unwrap_env(env.venv)
        # Done, real env
        return env

class MacroEnv(gym.Env):
    """
    Hierarchical Macro Environment for Search-Track switching.
    """

    def __init__(self, n_targets=5, n_unknown_targets=100, space_size=100.0,
                 d_state=4, fov_size=4.0, max_steps=100,
                 search_agent=None, track_agent=None, seed=None, heuristicTracker=False):
        super().__init__()

        # --------------------
        # Micro agents (provided externally)
        # --------------------
        self.search_agent = search_agent
        self.track_agent = track_agent

        # --------------------
        # Macro action/observation space
        # --------------------
        self.rng = np.random.default_rng(seed)
        self.action_space = gym.spaces.Discrete(2)  # 0=search, 1=track
        self.obs_dim_per_target = 1   # trace of covariance
        self.init_n_targets = n_targets
        self.init_n_unknown_targets = n_unknown_targets
        self.max_targets = self.init_n_targets + self.init_n_unknown_targets
        self.boundary = np.sqrt(1.0e-2)
        self._last_prob_sum = 0.0
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0]),
            high=np.array([1.0,  1.0]),
            dtype=np.float32
        )
        """ self.observation_space = gym.spaces.Box(
                low=0.0, 
                high=np.inf,
                shape=(self.max_targets, self.obs_dim_per_target),
                dtype=np.float32
            ) """
        """ self.observation_space = gym.spaces.Box(
            low=-1.0, 
            high=1.0,
            shape=(obs_len,),
            dtype=np.float32
        ) """
        self.fov_size = fov_size
        self.threshold_fov = 0.7
        self.tracking_requested = 0.8
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets
        self.lost_counter = 0
        self.detect_counter = 0
        self.space_size = space_size
        self.d_state = d_state  # state dimension per target (e.g., x,y,vx,vy)
        self.max_steps = max_steps
        # Parameters related to dynamical motion
        self.motion_model = self.rng.choice(["L", "T"], size=self.n_targets + self.n_unknown_targets)       # L=linear motion, T=coordinated turn
        self.motion_params = self.rng.uniform(0.05, 0.3, size=self.n_targets + self.n_unknown_targets)          # contains either linear velocity or turn rate
        self.P0 = np.eye(self.d_state) * 0.2
        self.P0[-2:, -2:] = np.eye(2) * 0.05
        self.Q0 = np.eye(self.d_state) * 1e-1
        n_grid = max(1, int(np.floor(self.space_size / self.fov_size)))
        self.n_grid_cells = n_grid * n_grid
        self.visit_counts = np.zeros(self.n_grid_cells, dtype=int)
        self.heuristicTracker = heuristicTracker
        # --------------------
        # Micro environments (created internally)
        # --------------------
        self.search_env = search_agent.get_env().envs[0]
        """ if track_agent == None:
            self.track_env  = None
        else: """
        self.track_env  = track_agent.get_env().envs[0]

    # ---------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.n_targets = self.init_n_targets
        self.n_unknown_targets = self.init_n_unknown_targets
        self.lost_counter = 0
        self.detect_counter = 0
        self._last_prob_sum = 0.0

        #reset dynamics
        self.motion_model = self.rng.choice(["L", "T"], size=self.n_targets + self.n_unknown_targets)       # L=linear motion, T=coordinated turn
        self.motion_params = self.rng.uniform(0.05, 0.3, size=self.n_targets + self.n_unknown_targets) 

        self.targets = [self._init_target(i) for i in range(self.init_n_targets)]
        self.unknown_targets = [
            self._init_unknown_target(i + self.init_n_targets)
            for i in range(self.init_n_unknown_targets)
        ]
        self.visit_counts[:] = 0   # reset search visit counts

        # Initialize search memory variables here
        self.last_search_idx = None
        self.prev_search_pos = None

        # known mask
        self.known_mask = np.zeros(self.max_targets, dtype=bool)
        self.known_mask[:self.n_targets] = True

        # Reset base environment first
        #obs, info = self.base_env.reset(seed=seed)
        
        real_search_env = unwrap_env(self.search_env)
        real_track_env  = unwrap_env(self.track_env)

        # Sync micro environments to match base_env
        _sync_envs(self, real_search_env)
        _sync_envs(self, real_track_env)

        self.obs = self._get_obs()
        info = {}

        return self.obs, info
    
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
            #"macro": np.ones(2, dtype=bool),  # always can choose search or track
            #"micro_search": np.ones(self.n_grid_cells, dtype=bool),  # all grid cells valid
            "micro_track": self.known_mask.copy()  # only known targets valid
        }
        return mask

    # ---------------------------------------------------------------------

    def step(self, macro_action):
        """
        macro_action: 0=SEARCH, 1=TRACK
        Executes K micro steps with the selected micro agent.
        """
        # unwrap both micro envs
        real_search_env = unwrap_env(self.search_env)
        real_track_env  = unwrap_env(self.track_env)

        # sync both with current world
        _sync_envs(self, real_search_env)
        _sync_envs(self, real_track_env)

        # Compute macro reward
        probSum = 0
        for tgt in self.targets:
            idx = tgt['id']  # global index
            model = self.motion_model[idx]
            param = self.motion_params[idx]

            predState, predCov = MultiTargetEnv.propagate_target_2D(tgt['x'], tgt['P'], tgt.get('Q', self.Q0), dt=real_search_env.dt, rng=self.rng, motion_model=model, motion_param=param)
            prob = compute_fov_prob_single(self.boundary, predState, predCov)
            probSum += prob

        if probSum == sum(self.known_mask) and macro_action == 0:
            macro_reward = 1
        elif probSum != sum(self.known_mask) and macro_action == 1:
            macro_reward = 1
        else:
            macro_reward = 0

        # choose agent
        if macro_action == 0:
            obs = real_search_env.obs
            micro_action, _ = self.search_agent.predict(obs, deterministic=False)
            next_obs, micro_reward, done, truncated, info = real_search_env.step(micro_action)
            """ trackingNeeded, trackingReallyNeeded = self._compute_track_reward(self.base_env)
            if (sum(trackingNeeded) + sum(trackingReallyNeeded))>=1:
                # tracking nedded instead of 
                macro_reward = -0.5 * (sum(trackingNeeded) + sum(trackingReallyNeeded))
            else:
                macro_reward = 1 """

            """ if len(info["lost_target"]) == 0:
                #macro_reward = self._compute_search_reward(real_search_env)
                trackingNeeded, trackingReallyNeeded = self._compute_track_reward(self)
                if any(trackingNeeded):
                    macro_reward = -2.0
                    if any(trackingReallyNeeded):
                        macro_reward = macro_reward - 3.0
                else:
                    macro_reward = +5.0
            else:
                # target got lost
                macro_reward = -10 * len(info["lost_target"]) """
            """ _sync_envs(real_search_env, self)
                obs = self._get_obs()
                self.step_count += 1

                reward = -100.0   
                done = True
                truncated = False

                info = {
                    "catastrophic_loss": True,
                    "remaining_targets": self.n_targets,
                    "action_mask": self.get_action_mask()
                }

                self.obs = obs
                return obs, reward, done, truncated, info """

            # sync back into base env
            _sync_envs(real_search_env, self)

        else:
            obs = real_track_env.obs
            if self.heuristicTracker:
                micro_action, best_ig, best_update = select_best_action_sumTrace(real_track_env, real_track_env.dt)
            else:
                if isinstance(self.track_agent, MaskablePPO):
                    action_masks = real_track_env.action_masks()
                    micro_action, _ = self.track_agent.predict(obs, action_masks=action_masks)
                else:
                    micro_action, _ = self.track_agent.predict(obs, deterministic=False)
            next_obs, micro_reward, done, truncated, info = real_track_env.step(micro_action)
            """ trackingNeeded, trackingReallyNeeded = self._compute_track_reward(self)
            if any(trackingNeeded):
                macro_reward = 2.0
                if any(trackingReallyNeeded):
                    macro_reward = macro_reward + 3.0
            else:
                macro_reward = -5.0

            # sync back into base env """
            _sync_envs(real_track_env, self)

        next_obs = self._get_obs()
        self.obs = next_obs
        #macro_reward = sum(next_obs)
        """ if info["lost_target"]:
            self.step_count += 1

            reward = -100.0   
            done = True
            truncated = False

            info = {
                "catastrophic_loss": True,
                "remaining_targets": self.n_targets,
                "action_mask": self.get_action_mask()
            }

            return next_obs, reward, done, truncated, info """
        done = self.step_count >= self.max_steps
        return next_obs, macro_reward, done, truncated, info

    # ---------------------------------------------------------------------
    def _get_obs(self, target_id=None):
        """Extracts target covariance trace."""

        """ known_by_id = {t["id"]: t for t in self.targets}

        obs_list = []

        for i in range(self.init_n_targets + self.init_n_unknown_targets):
            if i in known_by_id:
                tgt = known_by_id[i]
                p_fov = compute_fov_prob_single(
                    self.fov_size, tgt["x"], tgt["P"]
                )
            else:
                p_fov = 0.0

            obs_list.append(p_fov)

        return np.array(obs_list, dtype=np.float32) """

        """ by_id = {t["id"]: t for t in self.targets}
        by_id.update({t["id"]: t for t in self.unknown_targets})

        all_targets = [by_id[k] for k in sorted(by_id)]
        features = []

        for tgt in all_targets:
            trace = np.trace(tgt["P"])
            p_fov = compute_fov_prob_single(self.boundary, tgt["x"], tgt["P"])
            known = 1.0 if self.known_mask[tgt["id"]] else 0.0

            features.append([
                trace * known,
                p_fov * known
            ])

        obs = np.stack(features, axis=0)  # shape: (num_targets, 2)
        
        #print(obs.shape)
        return obs.astype(np.float32) """

        """Returns normalised sum of p_fov over known targets as a scalar."""

        n_known = sum(self.known_mask)

        if n_known == 0:
            self._last_prob_sum = 0.0
            return np.array([0.0, 0.0], dtype=np.float32)

        prob_sum = 0.0
        for tgt in self.targets:
            if self.known_mask[tgt["id"]]:
                prob_sum += compute_fov_prob_single(self.boundary, tgt["x"], tgt["P"])

        normalised = prob_sum / n_known

        # Compute delta from last observation
        delta = normalised - self._last_prob_sum
        self._last_prob_sum = normalised  # update for next call

        obs = np.array([normalised, delta], dtype=np.float32)  # shape: (2,)
        return obs

        """ all_targets = [] 
        for i in range(self.init_n_targets + self.init_n_unknown_targets):
            for tgt_known in self.base_env.targets:
                if tgt_known["id"] == i:
                    all_targets.append(tgt_known)
                    break
            for tgt_unknown in self.base_env.unknown_targets:
                if tgt_unknown["id"] == i:
                    all_targets.append(tgt_unknown)
                    break

            
        obs_list = []

        for i, tgt in enumerate(all_targets):
            x, y, vx, vy = tgt["x"]
            trace = np.trace(tgt["P"])

            p_fov = compute_fov_prob_single(self.fov_size, tgt["x"], tgt["P"])

            #known = 1.0 if self.known_mask[tgt["id"]] else 0.0

            obs_list.extend([p_fov])

        return np.array(obs_list, dtype=np.float32) """

    # ---------------------------------------------------------------------
    def _compute_search_reward(self, env):
        """
        Reward = 1 if all known targets' uncertainties fit inside FOV.
        """
        post_in_fov_unc = []
        post_track_required = []
        target_lost = []
        #post_in_fov = self._uncertainty_within_fov(env, margin=1.0)
        for obj in env.targets:
            x = obj['x']
            P = obj['P']
            prob = compute_fov_prob_single(self.fov_size, x, P)

            # check if probability of being in FOV is larger than threshold
            post_in_fov_unc.append(prob>self.tracking_requested)
            post_track_required.append(prob<self.tracking_requested and prob > self.threshold_fov)
            target_lost.append(prob<self.threshold_fov)
        
        if any(target_lost):
            return -1*sum(target_lost)

        return 2.0 if all(post_in_fov_unc) else 0 #if any(post_track_required)

    def _compute_track_reward(self, env):
        """
        Reward = 1 if any target's 3sigma uncertainty ellipse exceeds 80% of FOV.
        """
        risk = []
        riskier = []
        #pre_in_fov = self._uncertainty_within_fov(env, margin=0.8)
        for obj in env.targets:
            x = obj['x']
            P = obj['P']
            prob = compute_fov_prob_single(self.fov_size, x, P)

            # check if probability of being in FOV is lower than threshold
            risk.append(prob<self.tracking_requested)
            riskier.append(prob<self.threshold_fov)

        """ needs_tracking = any(risk)
        return 2.0 if needs_tracking else 3.0 if any(riskier) else -1.0  """
        return risk, riskier
        
    def _uncertainty_within_fov(self, env, margin=1.0):
        """
        Returns list of bools indicating whether each target's positional
        uncertainty (3σ ellipse radius) fits within a scaled fraction of the FOV.
        margin = 1.0 → full FOV limit
        margin = 0.8 → 80% of FOV limit (early warning)
        """
        fov_half = self.fov_size / 2.0
        limit = margin * fov_half
        within = []

        for obj in env.targets:
            P = obj['P']  # covariance matrix
            if P.shape[0] >= 2:
                P_xy = P[:2, :2]
                eigvals = np.linalg.eigvals(P_xy)
                r_i = 3.0 * np.sqrt(np.max(np.real(eigvals)))  # 3σ radius
            else:
                r_i = 3.0 * np.sqrt(P[0, 0])

            within.append(r_i <= limit)

        return within
    
def _sync_envs(source_env, dest_env):
    """
    Synchronize dynamic state between MultiTargetEnv instances.
    Copies all relevant state information so that both environments
    represent the same physical world at the current time step.
    """
    # --- Core target and environment state ---
    dest_env.targets = [copy.deepcopy(t) for t in source_env.targets]
    dest_env.unknown_targets = [copy.deepcopy(t) for t in source_env.unknown_targets]
    dest_env.n_targets = source_env.n_targets
    dest_env.n_unknown_targets = source_env.n_unknown_targets

    # --- Tracking-related state ---
    dest_env.known_mask = np.copy(source_env.known_mask)
    #dest_env.obs = np.copy(source_env.obs)
    dest_env.motion_model = np.copy(source_env.motion_model)
    dest_env.motion_params = np.copy(source_env.motion_params)
    dest_env.lost_counter = source_env.lost_counter
    dest_env.detect_counter = source_env.detect_counter

    # --- Search-related state ---
    dest_env.visit_counts = np.copy(source_env.visit_counts)
    dest_env.last_search_idx = source_env.last_search_idx
    dest_env.prev_search_pos = (
        None if source_env.prev_search_pos is None
        else np.copy(source_env.prev_search_pos)
    )

    # --- Time step ---
    dest_env.step_count = source_env.step_count

        # --- Optional: RNG state, if you want identical stochasticity ---
        #dest_env.rng.bit_generator.state = copy.deepcopy(source_env.rng.bit_generator.state)