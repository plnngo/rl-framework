import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy

from multi_target_env import MultiTargetEnv

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
                 d_state=4, fov_size=2.0, max_steps=300,
                 search_agent=None, track_agent=None):
        super().__init__()

        # --------------------
        # Base environment (combined mode)
        # --------------------
        self.base_env = MultiTargetEnv(n_targets=n_targets,
                                       n_unknown_targets=n_unknown_targets,
                                       space_size=space_size,
                                       d_state=d_state,
                                       fov_size=fov_size,
                                       max_steps=max_steps,
                                       mode="combined")

        # --------------------
        # Micro agents (provided externally)
        # --------------------
        self.search_agent = search_agent
        self.track_agent = track_agent

        # --------------------
        # Macro action/observation space
        # --------------------
        self.action_space = gym.spaces.Discrete(2)  # 0=search, 1=track
        self.observation_space = self.base_env.observation_space
        self.fov_size = fov_size

        # --------------------
        # Micro environments (created internally)
        # --------------------
        self.search_env = search_agent.get_env().envs[0]
        self.track_env  = track_agent.get_env().envs[0]

    # ---------------------------------------------------------------------
    def reset(self, **kwargs):
        # Reset base environment first
        obs, info = self.base_env.reset(**kwargs)

        real_search_env = unwrap_env(self.search_env)
        real_track_env  = unwrap_env(self.track_env)

        # Sync micro environments to match base_env
        self._sync_envs(self.base_env, real_search_env)
        self._sync_envs(self.base_env, real_track_env)

        return obs, info

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
        self._sync_envs(self.base_env, real_search_env)
        self._sync_envs(self.base_env, real_track_env)

        # choose agent
        if macro_action == 0:
            obs = real_search_env.obs
            micro_action, _ = self.search_agent.predict(obs, deterministic=False)
            next_obs, _, done, truncated, info = real_search_env.step(micro_action)
            macro_reward = self._compute_search_reward(real_search_env)

            # sync back into base env
            self._sync_envs(real_search_env, self.base_env)

        else:
            obs = real_track_env.obs
            micro_action, _ = self.track_agent.predict(obs, deterministic=False)
            next_obs, _, done, truncated, info = real_track_env.step(micro_action)
            macro_reward = self._compute_track_reward(self.base_env)

            # sync back into base env
            self._sync_envs(real_track_env, self.base_env)

        return next_obs, macro_reward, done, truncated, info

    # ---------------------------------------------------------------------
    def _build_macro_obs(self, obs):
        """Extracts target state + covariance + mask into flattened vector."""
        target_states = self.base_env.get_target_states()      # shape [n_targets, 4]
        target_covs = self.base_env.get_target_covariances()   # shape [n_targets, 2x2 or diag]
        obs_list = []
        for i in range(self.max_targets):
            if i < self.n_targets:
                s = target_states[i]
                c = np.diag(target_covs[i]) if target_covs[i].ndim == 2 else target_covs[i]
                obs_list.append(np.concatenate([s, c]))
            else:
                obs_list.append(np.zeros(self.obs_dim_per_target))
        obs_flat = np.concatenate(obs_list + [self.known_mask.astype(np.float32)])
        return obs_flat.astype(np.float32)

    # ---------------------------------------------------------------------
    def _compute_search_reward(self, env):
        """
        Reward = 1 if all known targets' uncertainties fit inside FOV.
        """
        post_in_fov_unc = self._uncertainty_within_fov(env, margin=1.0)
        return 1.0 if all(post_in_fov_unc) else 0.0

    def _compute_track_reward(self, env):
        """
        Reward = 1 if any target's 3σ uncertainty ellipse exceeds 80% of FOV.
        """
        pre_in_fov_unc = self._uncertainty_within_fov(env, margin=0.8)
        needs_tracking = not all(pre_in_fov_unc)
        return 1.0 if needs_tracking else 0.0
        
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
    
    def _sync_envs(self, source_env, dest_env):
        """
        Synchronize dynamic state between MultiTargetEnv instances.
        Copies all relevant state information so that both environments
        represent the same physical world at the current time step.
        """
        # --- Core target and environment state ---
        dest_env.targets = [copy.deepcopy(t) for t in source_env.targets]
        dest_env.unknown_targets = [copy.deepcopy(t) for t in source_env.unknown_targets]

        # --- Tracking-related state ---
        dest_env.known_mask = np.copy(source_env.known_mask)
        dest_env.obs = np.copy(source_env.obs)
        dest_env.motion_model = np.copy(source_env.motion_model)
        dest_env.motion_params = np.copy(source_env.motion_params)

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