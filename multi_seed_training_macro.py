import os
import numpy as np
import gymnasium as gym
import matplotlib.cm as cm
import pandas as pd

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from MacroEnv import MacroEnv
from multi_target_env import MultiTargetEnv
from train_agent import SharedLivePlot, LivePlotCallback


# =============================================================================
# CONFIG
# =============================================================================

algos = ["PPO", "DQN", "Random"]
seeds = [42, 123, 321]
total_timesteps = 50_000
save_dir = "macro_results"
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# MACRO-ENV WRAPPER: RANDOM SEED EACH RESET
# =============================================================================

class MacroRandomSeedEnv(gym.Env):
    """
    Recreates MacroEnv with random seeds at every reset.
    This guarantees generalization over seeds during training.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed_list, n_targets=5, n_unknown_targets=100, fov_size=4.0):
        super().__init__()

        self.seed_list = seed_list
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets
        self.fov_size = fov_size
        self.init_n_target = n_targets
        self.init_n_unknown_target = n_unknown_targets

        # Build initial env to expose observation and action space
        self.env = MacroRandomSeedEnv._make_env(self.n_targets, self.n_unknown_targets, np.random.choice(seed_list))
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    # Build a MacroEnv for a given seed
    def _make_env(n_targets, n_unknown_targets, seed):
        env_search = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="search"
        )
        env_track = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="track"
        )
        search_agent = PPO.load("agents/ppo_search_trained", env=env_search)
        track_agent = PPO.load("agents/ppo_track_trained", env=env_track)

        return MacroEnv(
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            search_agent=search_agent,
            track_agent=track_agent,
        )

    def reset(self, **kwargs):
        seed = np.random.choice(self.seed_list)
        self.env = MacroRandomSeedEnv._make_env(self.n_targets, self.n_unknown_targets, seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        if self.env:
            self.env.close()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_agent(algo_name, env, plotter, color, total_timesteps, save_dir):
    """
    Trains PPO or DQN with your tuned hyperparameters.
    """

    print(f"\n=== Training {algo_name} on MacroEnv ===")

    if algo_name == "PPO":
        # Insert best tuned PPO parameters here
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0003568750531511821,
            n_steps=512,
            batch_size=128,
            gamma=0.9286417039953475,
            gae_lambda=0.9413945148574974,
            clip_range=0.15498735004187073,
            ent_coef=0.03513597942956761,
            vf_coef=0.6977889828322719,
            max_grad_norm=0.9820965340387906,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=1,
        )

    elif algo_name == "DQN":
        # Insert best tuned DQN parameters here
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.00011112022098715457,
            buffer_size=100_000,
            batch_size=64,
            gamma=0.9735540181788884,
            train_freq=1,
            gradient_steps=8,
            learning_starts=2000,
            exploration_fraction=0.20831602508681876,
            exploration_final_eps=0.044816056736162124,
            target_update_interval=15_000,
            max_grad_norm=1.8972857042327866,
            policy_kwargs=dict(net_arch=[128, 128]),
            verbose=1,
        )

    else:
        raise ValueError("Unknown RL algorithm.")

    # Live plotting callback
    callback = LivePlotCallback(
        plotter=plotter,
        name=algo_name,
        color=color,
        plot_interval=50,
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # Save the model
    model_path = os.path.join(save_dir, f"{algo_name.lower()}_macro_trained.zip")
    model.save(model_path)
    print(f"[{algo_name}] Model saved to {model_path}")

    env.close()


# =============================================================================
# RANDOM POLICY BASELINE
# =============================================================================

def run_random_policy(plotter, color, total_timesteps, save_dir):
    print("\n=== Running Random Policy on MacroEnv ===")

    env = MacroRandomSeedEnv(seeds)
    obs, _ = env.reset()

    total_steps = 0
    episode_reward = 0
    episode_length = 0

    rewards_list = []
    step_list = []

    plotter.register_line("Random", color=color)

    while total_steps < total_timesteps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        episode_reward += reward
        episode_length += 1
        total_steps += 1

        if done or truncated:
            plotter.update("Random", episode_reward, episode_length)
            rewards_list.append(episode_reward)
            step_list.append(total_steps)

            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()

    df = pd.DataFrame({
        "episode_reward": rewards_list,
        "cumulative_steps": step_list
    })
    csv_path = os.path.join(save_dir, "random_macro_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Random] results saved to {csv_path}")

    env.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    shared_plotter = SharedLivePlot("MacroEnv Agent Comparison")

    # --- PPO ---
    color_ppo = cm.get_cmap("tab10")(0)
    env_ppo = DummyVecEnv([lambda: MacroRandomSeedEnv(seeds)])
    train_agent("PPO", env_ppo, shared_plotter, color_ppo, total_timesteps, save_dir)

    # --- DQN ---
    """ color_dqn = cm.get_cmap("tab10")(1)
    env_dqn = DummyVecEnv([lambda: MacroRandomSeedEnv(seeds)])
    train_agent("DQN", env_dqn, shared_plotter, color_dqn, total_timesteps, save_dir) """

    # --- RANDOM POLICY ---
    color_random = cm.get_cmap("tab10")(2)
    run_random_policy(shared_plotter, color_random, total_timesteps, save_dir)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()
