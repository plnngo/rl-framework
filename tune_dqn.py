import numpy as np
import optuna
import matplotlib.cm as cm
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from train_agent import LivePlotCallback, SharedLivePlot
from multi_target_env import MultiTargetEnv


# --- ENV CREATION ------------------------------------------------------------

def make_env(mode="track", n_targets=5, n_unknown_targets=100, seed=None):
    """Utility function to create and return the environment."""
    # Always return the same setup for training and evaluation
    def _init():
        env = MultiTargetEnv(
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            mode=mode,
            seed=seed,
        )
        return env

    # DQN requires a vectorized environment
    return DummyVecEnv([_init])


# --- TRAINING LOOP -----------------------------------------------------------

def train_dqn(params, trial_name, total_timesteps=50_000, plotter=None, color=None, seed=None):
    """Train a DQN agent with given parameters."""
    env = make_env(seed=seed)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        buffer_size=params["buffer_size"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        exploration_fraction=params["exploration_fraction"],
        target_update_interval=params["target_update_interval"],
        train_freq=params["train_freq"],
        verbose=0,
        seed=seed,
    )

    # Use shared plotter if provided
    if plotter is None:
        plotter = SharedLivePlot("Tune DQN")

    live_callback = LivePlotCallback(plotter, name=trial_name, color=color, plot_interval=5)

    model.learn(total_timesteps=total_timesteps, callback=live_callback)

    env.close()
    return model


# --- OPTUNA OBJECTIVE --------------------------------------------------------

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
            trial_name=f"{trial_name}_seed{seed}",
            total_timesteps=50_000,
            plotter=shared_plotter,
            color=color,
            seed=seed,
        )

        # Create eval env consistent with training
        env = make_env(seed=seed)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        rewards.append(mean_reward)

    avg_reward = np.mean(rewards)
    return avg_reward


# --- MAIN --------------------------------------------------------------------

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