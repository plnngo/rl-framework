import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import BaseCallback

from MCTSenv import MCTS
from multi_target_env import MultiTargetEnv

# === CONFIG ===
algos = ["PPO", "DQN", "Random"]
seeds = [42, 123, 321]
total_timesteps = 60_000 #100_000
mode = "search"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)


class EpisodeRewardCSVLogger(BaseCallback):
    """
    Logs episode-level rewards and cumulative timesteps to a CSV file.
    Mimics LivePlotCallback without plotting.
    """

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.cumulative_timesteps = 0
        self.timesteps_list = []
        self.rewards_list = []

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        for r, done in zip(rewards, dones):
            self.current_episode_reward += r
            self.current_episode_length += 1
            self.cumulative_timesteps += 1

            if done:
                # Record cumulative timestep at end of episode and reward
                self.timesteps_list.append(self.cumulative_timesteps)
                self.rewards_list.append(self.current_episode_reward)

                # Reset for next episode
                self.current_episode_reward = 0
                self.current_episode_length = 0

        return True

    def _on_training_end(self) -> None:
        df = pd.DataFrame({
            "cumulative_timestep": self.timesteps_list,
            "episode_reward": self.rewards_list
        })
        df.to_csv(self.save_path, index=False)
        if self.verbose > 0:
            print(f"[EpisodeRewardCSVLogger] Saved CSV to {self.save_path}")


def train_with_seed(algo_name, seed):
    """Train PPO or DQN with a given seed and save current-episode reward per step."""
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, seed=seed, mode=mode)

    # --- Select algorithm ---
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            gamma=0.9148472308668416,
            n_steps=1024,          # rollout length per policy update
            ent_coef=0.025794301927672722,
            learning_rate=9.530911967958957e-05,
            vf_coef=0.217347167922477,
            max_grad_norm=0.5693505994783516,
            gae_lambda=0.808567431707699,
            n_epochs=10,
            clip_range=0.21812912572546356,
            batch_size=64,
            verbose=1,
            seed=seed
        )
    elif algo_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            gamma=0.9111564115590075,    #0.9820106516911145,
            learning_rate=2.567873776981162e-5,     #2.7e-5,
            buffer_size=140_000,     #200_000,
            learning_starts=1000,
            train_freq=8,          # step-by-step updates to match logging
            batch_size=64,
            target_update_interval=1000,
            exploration_fraction=0.2279227219731062,
            exploration_final_eps=0.01,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError("Unknown algorithm")

    # --- Create callback for logging ---
    save_file = f"{save_dir}/{algo_name.lower()}_{mode}_seed{seed}_current.csv"
    callback = EpisodeRewardCSVLogger(save_path=save_file, verbose=1)

    # --- Train with callback ---
    model.learn(total_timesteps=total_timesteps, callback=callback)


def run_random_policy_and_save(env_fn, seed, total_timesteps=total_timesteps, mode="search", save_dir="results"):
    """Run a random policy and save current-episode reward at every step, compatible with RL CSVs."""
    os.makedirs(save_dir, exist_ok=True)

    env = env_fn(seed=seed)
    obs, _ = env.reset()

    current_episode_reward = 0
    current_episode_length = 0
    cumulative_timesteps = 0

    rewards_list = []
    timesteps_list = []
    lengths_list = []

    while cumulative_timesteps < total_timesteps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        current_episode_reward += reward
        current_episode_length += 1
        cumulative_timesteps += 1

        if done or truncated:
            # Log episode-level info
            rewards_list.append(current_episode_reward)
            timesteps_list.append(cumulative_timesteps)
            lengths_list.append(current_episode_length)

            # Reset for next episode
            current_episode_reward = 0
            current_episode_length = 0
            obs, _ = env.reset()

    # Save CSV
    df = pd.DataFrame({
        "cumulative_timestep": timesteps_list,
        "episode_reward": rewards_list,
        "episode_length": lengths_list
    })
    filename = f"{save_dir}/random_{mode}_seed{seed}.csv"
    df.to_csv(filename, index=False)
    print(f"[Random Policy] Saved: {filename}")

def run_mcts_policy_and_save(env_fn, mcts, seed, total_timesteps=1000, mode=None, save_dir="results", visualize=False):
    """
    Run MCTS-driven policy and save episode rewards, lengths, and timesteps, compatible with RL CSVs.

    Parameters
    ----------
    env_fn : callable
        Function that returns a fresh environment instance when called.
    mcts : MCTS
        Initialized MCTS object.
    seed : int
        Random seed for environment initialization.
    total_timesteps : int
        Total number of timesteps to simulate.
    mode : str
        "search", "track", or "both", passed to environment.
    save_dir : str
        Directory to save CSV results.
    visualize : bool
        Whether to visualize the MCTS tree inline at each episode start.
    """
    os.makedirs(save_dir, exist_ok=True)

    env = env_fn(seed=seed)
    obs = env.reset()
    
    current_episode_reward = 0
    current_episode_length = 0
    cumulative_timesteps = 0

    rewards_list = []
    timesteps_list = []
    lengths_list = []

    while cumulative_timesteps < total_timesteps:
        print(cumulative_timesteps)
        # --- Plan next action using MCTS ---
        mcts.reset_tree(obs)
        best_action = mcts.choose()
        
        # Optional: visualize tree
        if visualize:
            src = mcts.visualize_mcts_tree(mcts.root, max_depth=3)
            try:
                from IPython.display import display
                display(src)  # inline visualization in notebooks
            except ImportError:
                print("IPython not installed, skipping visualization")

        # Step environment
        next_obs, reward, done, truncated, info = env.step(best_action)

        current_episode_reward += reward
        current_episode_length += 1
        cumulative_timesteps += 1

        # Update MCTS root for next step (re-rooting)
        mcts.root = mcts.re_root(best_action)
        obs = next_obs

        if done or truncated:
            # Log episode info
            rewards_list.append(current_episode_reward)
            timesteps_list.append(cumulative_timesteps)
            lengths_list.append(current_episode_length)

            # Reset counters
            current_episode_reward = 0
            current_episode_length = 0
            obs = env.reset()

    # Save CSV
    df = pd.DataFrame({
        "cumulative_timestep": timesteps_list,
        "episode_reward": rewards_list,
        "episode_length": lengths_list
    })
    filename = f"{save_dir}/mcts_{mode}_seed{seed}.csv"
    df.to_csv(filename, index=False)
    print(f"[MCTS Policy] Saved: {filename}")


for seed in seeds:
    env_fn = lambda seed=seed: MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=seed, mode=mode)

    print("Start training with PPO")
    # PPO
    train_with_seed("PPO", seed)

    print("Start training with DQN")
    # DQN
    train_with_seed("DQN", seed)

    print("Start random")
    # Random
    run_random_policy_and_save(env_fn, seed, total_timesteps)
     # Instantiate MCTS for this environment
    #env = env_fn()
    #mcts = MCTS(env, n_simulations=1000, rollout_depth=5, mode="search")  # adjust parameters

    #print("Start MCTS policy")
    #run_mcts_policy_and_save(env_fn, mcts, seed, total_timesteps=total_timesteps, mode=mode, save_dir="results", visualize=False)