import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor

from multi_target_env import MultiTargetEnv

# import your environment
# from your_env_module import MultiTargetEnv

# === CONFIG ===
algos = ["PPO", "DQN", "Random"]
seeds = [0, 1, 2]
total_timesteps = 1_000
mode = "search"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)


def train_with_seed(algo_name, seed):
    """Train PPO or DQN with a given seed and save per-episode rewards."""
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, seed=seed, mode=mode)
    env = Monitor(env)

    # --- Select algorithm ---
    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            gamma=0.9,
            n_steps=256,
            learning_rate=3e-4,
            seed=seed,
            verbose=0,
        )
    elif algo_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            gamma=0.9820106516911145,
            learning_rate=2.7e-5,
            buffer_size=200_000,
            learning_starts=1000,
            train_freq=16,
            batch_size=64,
            target_update_interval=2500,
            exploration_fraction=0.24,
            exploration_final_eps=0.01,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError("Unknown algorithm")

    # --- Train ---
    model.learn(total_timesteps=total_timesteps)

    # --- Extract per-episode rewards from monitor ---
    ep_rewards = env.get_episode_rewards()  # or from your callback

    # --- Save results ---
    df = pd.DataFrame({
        "episode": np.arange(len(ep_rewards)),
        "reward": ep_rewards,
    })
    filename = f"{save_dir}/{algo_name.lower()}_{mode}_seed{seed}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

def run_random_policy_and_save(env_fn, seed, total_timesteps=total_timesteps, mode="search", save_dir="results"):
    """Run random policy on environment with given seed and save per-episode rewards."""
    os.makedirs(save_dir, exist_ok=True)
    
    env = env_fn(seed=seed)
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    timesteps = 0
    rewards = []

    while timesteps < total_timesteps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        timesteps += 1

        if done or truncated:
            rewards.append(episode_reward)
            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()

    # Save rewards per episode
    df = pd.DataFrame({
        "episode": np.arange(len(rewards)),
        "reward": rewards
    })
    filename = f"{save_dir}/random_{mode}_seed{seed}.csv"
    df.to_csv(filename, index=False)
    print(f"[Random Policy] Saved: {filename}")


for seed in seeds:
    env_fn = lambda seed=seed: MultiTargetEnv(n_targets=5, n_unknown_targets=15, seed=seed, mode="search")
    
    # PPO
    train_with_seed("PPO", seed)
    
    # DQN
    train_with_seed("DQN", seed)
    
    # Random policy
    run_random_policy_and_save(env_fn, seed, total_timesteps)