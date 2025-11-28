import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.cm as cm
import pandas as pd
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from multi_target_env import MultiTargetEnv
from train_agent import SharedLivePlot, LivePlotCallback  # assumes you already have these


# === CONFIG ===
algos = ["PPO", "DQN", "Random"]
seeds = [42, 123, 321]
total_timesteps = 20_000
mode = "track"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)


# === ENVIRONMENT WRAPPER ======================================================

class RandomSeedEnv(gym.Env):
    """Recreates MultiTargetEnv with a random seed at every reset."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed_list, mode="search", n_targets=5, n_unknown_targets=100):
        super().__init__()

        self.seed_list = seed_list
        self.mode = mode
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets

        # Create an initial environment so we can expose spaces
        self.env = MultiTargetEnv(
            n_targets=self.n_targets,
            n_unknown_targets=self.n_unknown_targets,
            seed=np.random.choice(self.seed_list),
            mode=self.mode,
        )

        # SB3 Requires these:
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        # Recreate env with a new seed each reset
        seed = np.random.choice(self.seed_list)
        self.env = MultiTargetEnv(
            n_targets=self.n_targets,
            n_unknown_targets=self.n_unknown_targets,
            seed=seed,
            mode=self.mode,
        )
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        if self.env:
            self.env.close()


# === TRAINING FUNCTIONS ======================================================

def train_agent(algo_name, env, plotter, color, total_timesteps, save_dir):
    """Train PPO or DQN across all seeds."""
    print(f"=== Training {algo_name} across all seeds ===")

    if algo_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            gamma=0.99, #0.9158906517459942, #0.9790210247139031,   
            n_steps=128, #1024,
            ent_coef=0.005, #0.0013392533378982774, #0.03158387252345037, 
            learning_rate=0.0003, #0.0006609120604125945,    #0.0007266996909845838,   
            vf_coef=0.5, #0.3088596031455526, #0.8349958665992091, 
            max_grad_norm=1.0, #0.3693696658724082, #0.7926595046318459,   
            gae_lambda=0.8, #0.9572736445545191, #0.886078389184115,  
            n_epochs=10,
            clip_range=0.5, #0.3890633888493019, #0.38403499436025856,  
            batch_size=256, #128, #256,
            verbose=1,
        )

    elif algo_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            gamma=0.9576075081024715,
            learning_rate=0.00016587775102695835,
            buffer_size=30000,
            learning_starts=1000,
            train_freq=1,
            batch_size=64,
            target_update_interval=12000,
            exploration_fraction=0.7893585250785973,
            exploration_final_eps=0.09706959490521018,
            verbose=1,
            gradient_steps=4,
            max_grad_norm=0.6395054936295579,
        )
    else:
        raise ValueError("Unknown algorithm")

    # Create live plot callback
    callback = LivePlotCallback(
        plotter=plotter,
        name=algo_name,
        color=color,
        plot_interval=50,
    )

    # Train model
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save model
    model_path = os.path.join(save_dir, f"{algo_name.lower()}_{mode}_trained.zip")
    model.save(model_path)
    print(f"[{algo_name}] Model saved to {model_path}")

    env.close()


def run_random_policy(plotter, color, total_timesteps, save_dir):
    """Run a random policy as a baseline with live plotting and CSV logging."""
    print("=== Running Random policy across all seeds ===")

    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, mode=mode)
    obs, _ = env.reset()
    cumulative_timesteps = 0
    current_episode_reward = 0
    current_episode_length = 0

    rewards_list = []
    timesteps_list = []

    plotter.register_line("Random", color=color)

    while cumulative_timesteps < total_timesteps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        current_episode_reward += reward
        current_episode_length += 1
        cumulative_timesteps += 1

        if done or truncated:
            rewards_list.append(current_episode_reward)
            timesteps_list.append(cumulative_timesteps)
            plotter.update("Random", current_episode_reward, current_episode_length)

            current_episode_reward = 0
            current_episode_length = 0
            obs, _ = env.reset()

    # Save results to CSV
    df = pd.DataFrame({
        "cumulative_timestep": timesteps_list,
        "episode_reward": rewards_list
    })
    csv_path = os.path.join(save_dir, f"random_{mode}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Random Policy] Results saved to {csv_path}")

    env.close()


# === MAIN ====================================================================

def main():
    np.random.seed(0)
    shared_plotter = SharedLivePlot("Agent Comparison")

    # PPO
    color_ppo = cm.get_cmap("tab10")(0)
    env_ppo = DummyVecEnv([lambda: RandomSeedEnv(seeds, mode=mode)])
    train_agent("PPO", env_ppo, shared_plotter, color_ppo, total_timesteps, save_dir)

    # DQN
    """ color_dqn = cm.get_cmap("tab10")(1)
    env_dqn = DummyVecEnv([lambda: RandomSeedEnv(seeds, mode=mode)])
    train_agent("DQN", env_dqn, shared_plotter, color_dqn, total_timesteps, save_dir) """

    # Random Policy
    color_rand = cm.get_cmap("tab10")(2)
    run_random_policy(shared_plotter, color_rand, total_timesteps, save_dir)

    # Finalize plot
    shared_plotter.finalize()


if __name__ == "__main__":
    main()


    # Instantiate MCTS for this environment
    #env = env_fn()
    #mcts = MCTS(env, n_simulations=1000, rollout_depth=5, mode="search")  # adjust parameters

    #print("Start MCTS policy")
    #run_mcts_policy_and_save(env_fn, mcts, seed, total_timesteps=total_timesteps, mode=mode, save_dir="results", visualize=False)