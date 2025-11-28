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
        # --- Learning ---
        "learning_rate": trial.suggest_float(
            "learning_rate", 5e-5, 4e-4, log=True
        ),
        "gamma": trial.suggest_float(
            "gamma", 0.9, 0.95
        ),

        # --- Replay buffer ---
        "buffer_size": trial.suggest_categorical(
            "buffer_size", [30_000, 50_000, 100_000]
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size", [32, 64, 128]
        ),

        # --- Behaviour (exploration schedule) ---
        "exploration_fraction": trial.suggest_float(
            "exploration_fraction", 0.5, 0.8
        ),
        "exploration_final_eps": trial.suggest_float(
            "exploration_final_eps", 0.03, 0.1
        ),

        # --- Training schedule ---
        "train_freq": trial.suggest_categorical(
            "train_freq", [1, 4]
        ),
        "gradient_steps": trial.suggest_categorical(
            "gradient_steps", [4, 8]
        ),
        "learning_starts": trial.suggest_categorical(
            "learning_starts", [5000, 10000]
        ),

        # --- Target network update ---
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [8000, 12000]
        ),

        # --- Stabilization ---
        "max_grad_norm": trial.suggest_float(
            "max_grad_norm", 0.2, 2.0
        ),

        # --- Network architecture ---
        "policy_kwargs": dict(
            net_arch=trial.suggest_categorical(
                "net_arch", [(128, 128), (256, 256)]
            )
        ),
    }
    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    seeds = [42, 123, 321]
    rewards = []

    for seed in seeds:
        model = train_dqn(
            params,
            trial_name=f"{trial_name}_seed{seed}",
            total_timesteps=20_000,
            plotter=shared_plotter,
            color=color,
            seed=seed,
        )

        env = make_env(seed=seed)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()
        rewards.append(mean_reward)

    return float(np.mean(rewards))


# --- MAIN --------------------------------------------------------------------

def main():
    """Main entry point for Optuna study."""
    shared_plotter = SharedLivePlot("Tune DQN")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(lambda trial: objective(trial, shared_plotter), n_trials=7)

    print("Best trial:")
    print(study.best_trial.params)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()