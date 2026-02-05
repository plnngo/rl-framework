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
total_timesteps = 60_000
mode = "search"
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)


# === ENVIRONMENT WRAPPER ======================================================

class RandomSeedEnv(gym.Env):
    """Recreates MultiTargetEnv with a random seed at every reset."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, seed_list, mode="track", n_targets=5, n_unknown_targets=100):
        super().__init__()

        self.seed_list = seed_list
        self.mode = mode
        self.n_targets = n_targets
        self.n_unknown_targets = n_unknown_targets

        # Create an initial environment so we can expose spaces
        self.env = MultiTargetEnv(
            n_targets=self.n_targets,
            n_unknown_targets=self.n_unknown_targets,
            seed = int(np.random.choice(self.seed_list)),
            mode=self.mode,
        )

        # SB3 Requires these:
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        # Recreate env with a new seed each reset
        seed = int(np.random.choice(self.seed_list))
        self.env = MultiTargetEnv(
            n_targets=self.n_targets,
            n_unknown_targets=self.n_unknown_targets,
            seed=seed,
            mode=self.mode,
        )
        return self.env.reset(seed=seed)

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
            gamma=0.9, #0.91174708735671, #0.9951316261625073, #0.9158906517459942, #0.9790210247139031,   
            n_steps=128, #1024, #512,
            ent_coef=0.005, #0.027289552689688718, #0.04145091999215972, #0.0013392533378982774, #0.03158387252345037, 
            learning_rate=0.0003, #0.00020547893214762386, #5.5104886882390796e-05, #0.0003, #0.0006609120604125945,    #0.0007266996909845838,   
            vf_coef=0.5, #0.6728668016049194, #0.58944318346072, #0.3088596031455526, #0.8349958665992091, 
            max_grad_norm=1.0, #0.38276422639245206, #0.40904524719654467, #0.3693696658724082, #0.7926595046318459,   
            gae_lambda=0.8, #0.9668902419115192, #0.9190087075813886, #0.9572736445545191, #0.886078389184115,  
            n_epochs=10,
            clip_range=0.5, #0.17974547307789224, #0.32790623385659234, #0.3890633888493019, #0.38403499436025856,  
            batch_size=256, #128, #32, #64,
            verbose=1,
        )

    elif algo_name == "DQN":
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate= 5.449952468830448e-05, #8.399722143201183e-05, #0.00010077055018104607, #0.00034674195012021614, #0.00016587775102695835, #0.00010685084135079141,
            buffer_size=30000, #30000,
            batch_size=128, #64, #32,
            gamma=0.9728252031515489, #0.9558374288830532, #0.9578429379766629, #0.9296592176892341, #0.9576075081024715, #0.9001853852142295,
            train_freq=4, #1,
            gradient_steps=8, #4,
            learning_starts=10000, #1000, #5000,
            exploration_fraction= 0.6102870049368956, #0.5040249936593768, #0.5207217410900746, #0.5742209831043317, #0.7893585250785973, #0.7398965813587768,
            exploration_final_eps= 0.031764573244963186, # 0.030534260752888866, #0.08327658175267656, #0.03548013259170907, #0.09706959490521018, #0.03683504126497514,
            target_update_interval=12000, #8000,
            max_grad_norm= 1.1777762743735063, #0.8178478937152223, #0.2673890378339314, #0.7279834488656272, #1.3411398428359127, #0.6395054936295579,
            policy_kwargs=dict(net_arch=[256, 256]), #128, 128
            verbose=1,
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
    model_path = os.path.join(save_dir, f"{algo_name.lower()}_{mode}_trained_IEEE.zip")
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
            #plotter.update("Random", current_episode_reward, current_episode_length)
            if 490 <= current_episode_reward <= 500:
                        plotter.update(
                            "Random",
                            current_episode_reward,
                            current_episode_length
                        )

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
    shared_plotter = SharedLivePlot("Agent comparison during training")

    # PPO
    color_ppo = cm.get_cmap("tab10")(0)
    env_ppo = DummyVecEnv([lambda: RandomSeedEnv(seeds, mode=mode)])
    #train_agent("PPO", env_ppo, shared_plotter, color_ppo, total_timesteps, save_dir)

    # DQN
    color_dqn = cm.get_cmap("tab10")(1)
    env_dqn = DummyVecEnv([lambda: RandomSeedEnv(seeds, mode=mode)])
    train_agent("DQN", env_dqn, shared_plotter, color_dqn, total_timesteps, save_dir)

    # Random Policy
    color_rand = cm.get_cmap("tab10")(2)
    #run_random_policy(shared_plotter, color_rand, total_timesteps, save_dir)

    # Finalize plot
    shared_plotter.finalize()


if __name__ == "__main__":
    main()


    # Instantiate MCTS for this environment
    #env = env_fn()
    #mcts = MCTS(env, n_simulations=1000, rollout_depth=5, mode="search")  # adjust parameters

    #print("Start MCTS policy")
    #run_mcts_policy_and_save(env_fn, mcts, seed, total_timesteps=total_timesteps, mode=mode, save_dir="results", visualize=False)