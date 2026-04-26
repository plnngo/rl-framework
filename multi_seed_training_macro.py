import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import wandb

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from MacroEnv import MacroEnv
from multi_target_env import MultiTargetEnv
from train_agent import DQNLivePlotCallback, PPOLivePlotCallback, SharedLivePlot


# =============================================================================
# CONFIG
# =============================================================================

algos = ["PPO", "DQN", "Random"]
seeds = [42, 123, 321]
total_timesteps = 500
job_id = os.environ.get("SLURM_JOB_ID", str(int(time.time())))
save_dir = f"macro_results/job_{job_id}"
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

    def __init__(self, seed_list, n_targets=5, n_unknown_targets=100, fov_size=4.0, heuristicTracker=True):
        super().__init__()

        self.seed_list = seed_list
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets
        self.fov_size = fov_size
        self.init_n_target = n_targets
        self.init_n_unknown_target = n_unknown_targets
        self.heuristicTracker = heuristicTracker

        # Build initial env to expose observation and action space
        self.env = MacroRandomSeedEnv._make_env(self.n_targets, self.n_unknown_targets, seed = int(np.random.choice(self.seed_list)), heuristicTracker=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    # Build a MacroEnv for a given seed
    def _make_env(n_targets, n_unknown_targets, seed, heuristicTracker=False, tracker=None):
        env_search = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="search"
        )
        env_track = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="track"
        )
        search_agent = PPO.load("agents/ppo_search_trained_slowTargets_obsSpace4Channels", env=env_search)
        if tracker == "dqn":
            track_agent = DQN.load("agents/dqn_track_trained_IEEE_covTrace", env=env_track)
        else:
            track_agent = MaskablePPO.load("agents/maskableppo_track_trained_IEEE_randomSpawn", env=env_track)

        return MacroEnv(
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            search_agent=search_agent,
            track_agent=track_agent,
            seed=seed,
            heuristicTracker=heuristicTracker
        )

    def reset(self, **kwargs):
        seed = int(np.random.choice(self.seed_list))
        self.env = MacroRandomSeedEnv._make_env(self.n_targets, self.n_unknown_targets, seed, heuristicTracker=self.heuristicTracker)
        return self.env.reset(seed=seed)

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
            learning_rate=0.00044614011815097786, #0.0003568750531511821,
            n_steps=384, #512,
            batch_size=128,
            gamma=0.999, #0.952199041429737, #0.9286417039953475,
            gae_lambda=0.9180541616290008, #0.9413945148574974,
            clip_range=0.12790513745344512,#0.15498735004187073,
            ent_coef=0.03565509800026144, #0.03513597942956761,
            vf_coef=0.8845018186381477, #0.6977889828322719,
            max_grad_norm=0.9464008722411874, #0.9820965340387906,
            policy_kwargs=dict(net_arch=[128, 64]),#dict(net_arch=[64, 64]),
            verbose=1,
        )

    elif algo_name == "DQN":
        # Insert best tuned DQN parameters here
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=8.450225441659817e-05, #0.00010917134021893014,
            buffer_size=50000, #20_000,
            batch_size=32, #64,
            gamma=0.9564895579103538, #0.965849106792237,
            train_freq=4,
            gradient_steps=8,
            learning_starts=5000,
            exploration_fraction=0.6439219138656918, #0.3908456292795791,
            exploration_final_eps=0.02838206605656649,  #0.05447874562759111,
            target_update_interval=15000, #5_000,
            max_grad_norm=1.805905030739738, #1.058094230288445,
            policy_kwargs=dict(net_arch=[128, 128]), #dict(net_arch=[256, 256]),
            verbose=1,
        )

    else:
        raise ValueError("Unknown RL algorithm.")

    # Separate eval env — must be a different instance

    # ── Initialise wandb ──────────────────────────────────────────
    run = wandb.init(
        entity="p-l-n-ngo-tu-delft",
        project="macro-rl",
        name=f"{algo_name}_job{job_id}",
        config={
            "algo": algo_name,
            "total_timesteps": total_timesteps,
            "gamma": model.gamma,
            "learning_rate": model.learning_rate,
            "net_arch": [128, 64] if algo_name == "PPO" else [128, 128],
        }
    )
    # ─────────────────────────────────────────────────────────────

    print(f"save_dir exists: {os.path.exists(save_dir)}")
    print(f"seeds value: {seeds}")
    eval_env = DummyVecEnv([lambda: MacroRandomSeedEnv(seeds)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        eval_freq=500,          # evaluate every 500 training steps
        n_eval_episodes=20,     # average over 20 episodes
        deterministic=True,     # no epsilon-greedy
        render=False,
        verbose=1,
    )

    if algo_name == "PPO":
        live_callback = PPOLivePlotCallback(plotter=plotter, name=algo_name, color=color)
    elif algo_name == "DQN":
        live_callback = DQNLivePlotCallback(plotter=plotter, name=algo_name, color=color)

    # Combine both callbacks
    callbacks = CallbackList([live_callback, eval_callback])

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # ── Sanity check: random policy on eval env ────────────────────
    obs = eval_env.reset()
    total_reward = 0
    done = False
    while not done:
        action = [eval_env.action_space.sample()]
        obs, reward, done, _ = eval_env.step(action)
        total_reward += reward.item()
    print(f"[{algo_name}] Random policy on eval env: {total_reward}")
    # ──────────────────────────────────────────────────────────────

    # Save the model
    model_path = os.path.join(save_dir, f"{algo_name.lower()}_macro_trained_heuristic_track.zip")
    model.save(model_path)
    print(f"[{algo_name}] Model saved to {model_path}")

    # ── Plot eval results ──────────────────────────────────────────
    eval_path = os.path.join(save_dir, "evaluations.npz")
    if os.path.exists(eval_path):
        data = np.load(eval_path)
        timesteps    = data["timesteps"]
        mean_rewards = data["results"].mean(axis=1)
        std_rewards  = data["results"].std(axis=1)

        plt.figure(figsize=(10, 4))
        plt.plot(timesteps, mean_rewards, label=f"{algo_name} (eval, deterministic)")
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
            label="±1 std"
        )
        plt.axhline(y=100, color="green", linestyle="--", label="Heuristic (100)")
        plt.axhline(y=50,  color="red",   linestyle="--", label="Random baseline (50)")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title(f"{algo_name} Evaluation Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{algo_name}_eval_curve.png"))
        plt.show()
    else:
        print(f"[{algo_name}] No evaluations.npz found at {eval_path}")
    # ──────────────────────────────────────────────────────────────

    run.finish()
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
    shared_plotter = SharedLivePlot("MacroEnv Agent Comparison", job_id)

    # --- PPO ---
    color_ppo = cm.get_cmap("tab10")(0)
    env_ppo = DummyVecEnv([lambda: MacroRandomSeedEnv(seeds)])
    train_agent("PPO", env_ppo, shared_plotter, color_ppo, total_timesteps, save_dir)

    # --- DQN ---
    color_dqn = cm.get_cmap("tab10")(1)
    env_dqn = DummyVecEnv([lambda: MacroRandomSeedEnv(seeds)])
    #train_agent("DQN", env_dqn, shared_plotter, color_dqn, total_timesteps, save_dir)

    # --- RANDOM POLICY ---
    color_random = cm.get_cmap("tab10")(2)
    #run_random_policy(shared_plotter, color_random, total_timesteps, save_dir)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()
