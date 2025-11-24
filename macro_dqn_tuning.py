import numpy as np
import optuna
import matplotlib.cm as cm

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from MacroEnv import MacroEnv
from multi_target_env import MultiTargetEnv

# Optional: your live-plot classes
from train_agent import LivePlotCallback, SharedLivePlot


# ============================================================================
# ENV CREATION
# ============================================================================

def make_macro_env(n_targets=5, n_unknown_targets=100, fov_size=2.0, seed=None):
    """
    Create a MacroEnv wrapped in DummyVecEnv for DQN.
    """

    def _init():
        # Create micro envs for loading PPO agents
        env_search = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=n_unknown_targets,
                                    seed=seed, mode="search")
        env_track = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=n_unknown_targets,
                                   seed=seed, mode="track")

        # Load pretrained micro-agents
        search_agent = PPO.load("agents/ppo_search_trained", env=env_search)
        track_agent = PPO.load("agents/ppo_track_trained", env=env_track)

        env = MacroEnv(
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            fov_size=fov_size,
            search_agent=search_agent,
            track_agent=track_agent
        )
        return env

    return DummyVecEnv([_init])


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_dqn_macro(params, trial_name, total_timesteps=50_000, plotter=None, color=None, seed=None):
    """
    Train a DQN agent on the MacroEnv with given hyperparameters.
    """

    env = make_macro_env(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        exploration_fraction=params["exploration_fraction"],
        exploration_final_eps=params["exploration_final_eps"],
        train_freq=params["train_freq"],
        gradient_steps=params["gradient_steps"],
        learning_starts=params["learning_starts"],
        target_update_interval=params["target_update_interval"],
        max_grad_norm=params["max_grad_norm"],
        policy_kwargs=params["policy_kwargs"],
        verbose=0,
        seed=seed,
    )

    # Optional live plot
    if plotter is None:
        plotter = SharedLivePlot("Tune Macro-DQN")

    live_callback = LivePlotCallback(
        plotter, name=trial_name, color=color, plot_interval=5
    )

    model.learn(total_timesteps=total_timesteps, callback=live_callback)

    env.close()
    return model


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective_macro_dqn(trial, shared_plotter=None):

    params = {
        # --- Learning ---
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 4e-4, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.99),

        # --- Replay buffer ---
        "buffer_size": trial.suggest_categorical("buffer_size", [20_000, 50_000, 100_000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),

        # --- Exploration ---
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.2, 0.8),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.02, 0.1),

        # --- Training schedule ---
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [1, 4, 8]),
        "learning_starts": trial.suggest_categorical("learning_starts", [2000, 5000, 10000]),

        # --- Target update ---
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [5000, 10000, 15000]
        ),

        # --- Stabilization ---
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.5, 2.0),

        # --- Network architecture ---
        "policy_kwargs": dict(
            net_arch=trial.suggest_categorical(
                "net_arch", [(128, 128), (256, 256)]
            ),
        ),
    }

    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    seeds = [42, 123, 321]
    rewards = []

    for seed in seeds:
        model = train_dqn_macro(
            params,
            trial_name=f"{trial_name}_seed{seed}",
            total_timesteps=30_000,
            plotter=shared_plotter,
            color=color,
            seed=seed,
        )

        # Evaluation
        eval_env = make_macro_env(seed=seed)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        eval_env.close()
        rewards.append(mean_reward)

    return float(np.mean(rewards))


# ============================================================================
# MAIN
# ============================================================================

def main():
    shared_plotter = SharedLivePlot("Macro-DQN Hyperparameter Tuning")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(lambda trial: objective_macro_dqn(trial, shared_plotter), n_trials=10)

    print("Best trial:")
    print(study.best_trial.params)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()
