# tune_dqn.py
import numpy as np
import optuna
import matplotlib.cm as cm
from stable_baselines3.common.evaluation import evaluate_policy
from train_agent import SharedLivePlot
from train_dqn import train_dqn
from multi_target_env import MultiTargetEnv


def make_env(mode="search", n_targets=5, n_unknown_targets=15):
    """Utility function to create and return the environment."""
    return MultiTargetEnv(n_targets=n_targets, n_unknown_targets=n_unknown_targets, mode=mode)


def objective(trial, shared_plotter=None):
    """Optuna objective function for tuning DQN hyperparameters."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_int("buffer_size", 50_000, 150_000, step=10_000),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
        "target_update_interval": trial.suggest_int("target_update_interval", 1000, 3000, step=500),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
    }

    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    seeds = [42, 123, 321]
    rewards = []
    for seed in seeds:
        model = train_dqn(
            params,
            trial_name=f"trial_{trial.number}_seed{seed}",
            total_timesteps=50_000,
            plotter=shared_plotter,
            color=None,
            seed=seed,  # Pass seed to environment/model if possible
        )
        env = make_env()
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()
        rewards.append(mean_reward)
    return np.mean(rewards)


def main():
    """Main entry point for Optuna study."""
    shared_plotter = SharedLivePlot("Tune DQN")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(lambda trial: objective(trial, shared_plotter), n_trials=10)

    print("Best trial:")
    print(study.best_trial.params)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()