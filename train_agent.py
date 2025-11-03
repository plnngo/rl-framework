import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from multi_target_env import MultiTargetEnv


# =========================================================
# Shared Live Plot Manager
# =========================================================
class SharedLivePlot:
    def __init__(self, title):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.lines = {}
        self.ax.set_xlabel("Timesteps")
        self.ax.set_ylabel("Episode Reward")
        self.ax.set_title(title)

    def register_line(self, name, color):
        """Register a new line for a training run."""
        line, = self.ax.plot([], [], color=color, label=name)
        self.lines[name] = {"line": line, "timesteps": [], "rewards": []}
        self.ax.legend()

    def update(self, name, episode_reward, episode_length):
        """Append new data and refresh plot."""
        data = self.lines[name]
        # Append cumulative timestep and reward
        total_timesteps = sum(data["timesteps"]) + episode_length if data["timesteps"] else episode_length
        data["timesteps"].append(episode_length if not data["timesteps"] else episode_length)
        data["rewards"].append(episode_reward)

        # Update line
        cumulative_timesteps = np.cumsum(data["timesteps"])
        data["line"].set_data(cumulative_timesteps, data["rewards"])
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def finalize(self):
        plt.ioff()
        self.ax.legend()
        plt.show()

    def get_average_reward(self, name):
        data = self.lines[name]
        return np.mean(data["rewards"]) if len(data["rewards"]) > 0 else np.nan


# =========================================================
# PPO Callback with Live Plotting
# =========================================================
class LivePlotCallback(BaseCallback):
    def __init__(self, plotter, name, color=None, plot_interval=100, verbose=0):
        super().__init__(verbose)
        self.plotter = plotter
        self.name = name
        self.color = color
        self.plot_interval = plot_interval
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_training_start(self):
        self.plotter.register_line(self.name, color=self.color)

    def _on_step(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        for r, done in zip(rewards, dones):
            self.current_episode_reward += r
            self.current_episode_length += 1

            # Regular episodic update
            if done:
                self.plotter.update(self.name, self.current_episode_reward, self.current_episode_length)
                self.current_episode_reward = 0
                self.current_episode_length = 0


        return True


# =========================================================
# Random Policy Runner
# =========================================================
def run_random_policy_with_plot(env, plotter, name="Random", total_timesteps=50_000):
    plotter.register_line(name, color="tab:red")

    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    timesteps = 0

    while timesteps < total_timesteps:
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        episode_length += 1
        timesteps += 1

        if done or truncated:
            plotter.update(name, episode_reward, episode_length)
            episode_reward = 0
            episode_length = 0
            obs, _ = env.reset()

    print(f"[Random Policy] Finished {timesteps} timesteps.")
    return

        
def main():

    mode = mode="search"

    # Create shared plotter
    plotter = SharedLivePlot("Training Progress: PPO vs Random Policy")

    # Create environment
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, seed=42, mode=mode)

    # Initialize DQN
    gamma = 0.9
    total_timesteps = 50_000

    # --- Run random policy ---
    print("Running random policy baseline...")
    run_random_policy_with_plot(env, plotter, name="Random", total_timesteps=total_timesteps)

    # DQN tracking works with this set up
    """ model = DQN(
        "MlpPolicy", env,
        gamma=gamma,
        exploration_fraction=0.3,
        exploration_final_eps=0.1,
        learning_starts=1000,
        target_update_interval=500,
        buffer_size=50_000,
        train_freq=4,
        verbose=1
    ) """

    model = PPO(
        "MlpPolicy",
        env,
        gamma=gamma,
        n_steps=256,          # rollout length per policy update
        ent_coef=0.01,         # encourage exploration
        learning_rate=3e-4,
        vf_coef=0.5,           # value function loss weight
        max_grad_norm=0.5,
        gae_lambda=0.95,       # GAE for smoother advantage estimates
        n_epochs=10,           # optimization epochs per update
        clip_range=0.2,        # PPO clipping parameter
        batch_size=64,         # minibatch size
        verbose=1
    )

    ppo_callback = LivePlotCallback(plotter, name="PPO", color="tab:orange")
    model.learn(total_timesteps=total_timesteps, callback=ppo_callback)

    # Create filename automatically
    gamma_str = str(gamma).replace('.', '')
    filename = f"ppo2_sensor_tasking_{mode}_gamma{gamma_str}_steps{total_timesteps}"
    #filename = f"dqn_sensor_tasking_search_gamma{gamma_str}_steps{total_timesteps}"

    # Save trained model
    model.save(filename)
    print(f"Model saved as {filename}.zip")
    
    # --- Train DQN ---
    print("Training DQN agent...")
    dqn_model = DQN(
            "MlpPolicy",
            env,
            gamma=0.9820106516911145,
            learning_rate=2.7139001406888036e-5,
            buffer_size=200_000,
            learning_starts=1000,
            train_freq=16,
            batch_size=64,
            target_update_interval=2500,
            exploration_fraction=0.23896606821738825,
            exploration_final_eps=0.01,
            verbose=1
            )

    dqn_callback = LivePlotCallback(plotter, name="DQN", color="tab:green")
    dqn_model.learn(total_timesteps=total_timesteps, callback=dqn_callback)

    # Save DQN model
    dqn_filename = f"dqn_sensor_tasking_{mode}_gamma09820_steps{total_timesteps}"
    dqn_model.save(dqn_filename)
    print(f"DQN model saved as {dqn_filename}.zip")

    # --- Summary comparison ---
    avg_random = plotter.get_average_reward("Random")
    avg_ppo = plotter.get_average_reward("PPO")
    avg_dqn = plotter.get_average_reward("DQN")

    print("\n=== Performance Summary ===")
    print(f"Average Random Reward: {avg_random:.2f}")
    print(f"Average PPO Reward:    {avg_ppo:.2f}")
    print(f"Average DQN Reward:    {avg_dqn:.2f}")

    # Simple ranking logic
    best = max([("Random", avg_random), ("PPO", avg_ppo), ("DQN", avg_dqn)], key=lambda x: x[1])
    print(f"\nBest performer: {best[0]} with average reward {best[1]:.2f}")

    plotter.finalize()


if __name__ == "__main__":
    main()
