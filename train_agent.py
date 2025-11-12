import os
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

    mode = "search"

    save_dir = "agents"
    os.makedirs(save_dir, exist_ok=True)

    # Create shared plotter
    plotter = SharedLivePlot("Training Progress: PPO vs Random Policy")

    # Create environment
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode=mode)

    # Initialize DQN
    gamma = 0.9148472308668416  #gamma = 0.9682225477210584 #track
    total_timesteps = 60_000

    # --- Run random policy ---
    print("Running random policy baseline...")
    run_random_policy_with_plot(env, plotter, name="Random", total_timesteps=total_timesteps)

    print("Training PPO agent to search...")
    model = PPO(
        "MlpPolicy",
        env,
        gamma=gamma,
        n_steps=1024,       #256,          # rollout length per policy update
        ent_coef=0.025794301927672722,  #0.01,         # encourage exploration
        learning_rate=9.530911967958957e-05,    #3e-4,
        vf_coef=0.217347167922477,  #0.5,           # value function loss weight
        max_grad_norm=0.5693505994783516,   #0.5,
        gae_lambda=0.808567431707699,   #0.95,       # GAE for smoother advantage estimates
        n_epochs=10,           # optimization epochs per update
        clip_range=0.21812912572546356,     #0.2,        # PPO clipping parameter
        batch_size=64,         # minibatch size
        verbose=1
    )
    """ print("Training PPO agent to track...")
    model = PPO(
        "MlpPolicy",
        env,
        gamma=0.9682225477210584,
        n_steps=128,       #256,          # rollout length per policy update
        ent_coef=0.03687230703244832,  #0.01,         # encourage exploration
        learning_rate=3.349227183953058e-05,    #3e-4,
        vf_coef=0.38050639142707177,  #0.5,           # value function loss weight
        max_grad_norm=0.4411024602318411,   #0.5,
        gae_lambda=0.9165059491087159,   #0.95,       # GAE for smoother advantage estimates
        n_epochs=10,           # optimization epochs per update
        clip_range=0.3391385210948936,     #0.2,        # PPO clipping parameter
        batch_size=256,         # minibatch size
        verbose=1
    ) """

    ppo_callback = LivePlotCallback(plotter, name="PPO", color="tab:orange")
    model.learn(total_timesteps=total_timesteps, callback=ppo_callback)

    # Create filename automatically
    gamma_str = str(gamma).replace('.', '')
    filename = f"{save_dir}/ppo2_sensor_tasking_{mode}_gamma{gamma_str}_steps{total_timesteps}"

    # Save trained model
    model.save(filename)
    print(f"Model saved as {filename}.zip")
    
    # --- Train DQN ---
    print("Training DQN agent to search...")
    dqn_model = DQN(
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
            verbose=0,
            )
    
    """ print("Training DQN agent to track...")
    dqn_model = DQN(
            "MlpPolicy",
            env,
            gamma=0.9526564282734367,    #0.9820106516911145,
            learning_rate=0.0004139635188401231,     #2.7e-5,
            buffer_size=150000,     #200_000,
            learning_starts=1000,
            train_freq=1,          # step-by-step updates to match logging
            batch_size=128,
            target_update_interval=3000,
            exploration_fraction=0.3021246896216901,
            exploration_final_eps=0.01,
            verbose=0,
            ) """

    dqn_callback = LivePlotCallback(plotter, name="DQN", color="tab:green")
    dqn_model.learn(total_timesteps=total_timesteps, callback=dqn_callback)

    # Save DQN model
    dqn_filename = f"{save_dir}/dqn_sensor_tasking_{mode}_gamma0911_steps{total_timesteps}"
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
    best = max([("PPO", avg_ppo), ("DQN", avg_dqn)], key=lambda x: x[1])
    print(f"\nBest performer: {best[0]} with average reward {best[1]:.2f}")

    plotter.finalize()


if __name__ == "__main__":
    main()
