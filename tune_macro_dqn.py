import numpy as np
import optuna
import matplotlib.cm as cm
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from multi_seed_training_macro import MacroRandomSeedEnv
from train_agent import LivePlotCallback, SharedLivePlot

# from multi_target_env import MultiTargetEnv        # (already used inside MacroRandomSeedEnv)


# =============================================================================
# --- ENV CREATION ------------------------------------------------------------
# =============================================================================

def make_macro_env(seed_list, n_targets=5, n_unknown_targets=100, fov_size=4.0):
    """
    Create a DummyVecEnv that internally constructs a MacroRandomSeedEnv,
    which in turn generates a new MacroEnv on every reset with a random seed.
    """

    def _init():
        return MacroRandomSeedEnv(
            seed_list=seed_list,
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            fov_size=fov_size,
        )

    return DummyVecEnv([_init])


# =============================================================================
# --- TRAINING LOOP -----------------------------------------------------------
# =============================================================================

def train_dqn_macro(
    params,
    trial_name,
    seed_list,
    total_timesteps=15_000,
    plotter=None,
    color=None,
):
    """
    Train a DQN agent on the Macro environment with given hyperparameters.
    """
    env = make_macro_env(seed_list)

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
    )

    # Use shared plotter if provided
    if plotter is None:
        plotter = SharedLivePlot("Tune MacroDQN")

    callback = LivePlotCallback(
        plotter,
        name=trial_name,
        color=color,
        plot_interval=5,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    env.close()

    return model


# =============================================================================
# --- OPTUNA OBJECTIVE --------------------------------------------------------
# =============================================================================

def objective_macro(trial, seed_list, shared_plotter=None):
    """
    Optuna objective function for tuning DQN hyperparameters in the Macro environment.
    """

    params = {
        # --- Learning ---
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 4e-4, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.97),

        # --- Replay ---
        "buffer_size": trial.suggest_categorical("buffer_size", [30_000, 50_000, 100_000]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),

        # --- Exploration schedule ---
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.4, 0.8),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.03, 0.1),

        # --- Training schedule ---
        "train_freq": trial.suggest_categorical("train_freq", [1, 4]),
        "gradient_steps": trial.suggest_categorical("gradient_steps", [4, 8]),
        "learning_starts": trial.suggest_categorical("learning_starts", [5000, 10000]),

        # --- Target update ---
        "target_update_interval": trial.suggest_categorical(
            "target_update_interval", [8000, 12000]
        ),

        # --- Stabilisation ---
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.2, 2.0),

        # --- Network ---
        "policy_kwargs": dict(
            net_arch=trial.suggest_categorical("net_arch", [(128, 128), (256, 256)])
        ),
    }

    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    # Evaluate with several runs for robustness
    evaluation_rewards = []

    for eval_run in range(3):
        model = train_dqn_macro(
            params,
            trial_name=f"{trial_name}_run{eval_run}",
            seed_list=seed_list,
            total_timesteps=40_000,
            plotter=shared_plotter,
            color=color,
        )

        # evaluate new env instance (random seed each reset)
        eval_env = make_macro_env(seed_list)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        eval_env.close()

        evaluation_rewards.append(mean_reward)

    return float(np.mean(evaluation_rewards))


# =============================================================================
# --- MAIN --------------------------------------------------------------------
# =============================================================================

def main():
    """
    Main entry point for Optuna hyperparameter search.
    """

    # Full seed list used by MacroRandomSeedEnv for sampling
    seed_list = [42, 123, 321]

    shared_plotter = SharedLivePlot("Tune MacroDQN")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(lambda trial: objective_macro(trial, seed_list, shared_plotter), n_trials=10)

    print("\nBest hyperparameters:")
    print(study.best_trial.params)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()