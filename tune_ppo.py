import numpy as np
import optuna
import matplotlib.cm as cm
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from train_agent import LivePlotCallback, SharedLivePlot
from multi_target_env import MultiTargetEnv


def make_env(mode="track", n_targets=5, n_unknown_targets=100, seed=None):
    """Utility function to create and return the environment."""
    return MultiTargetEnv(n_targets=n_targets, n_unknown_targets=n_unknown_targets, mode=mode)


def train_ppo(params, trial_name, total_timesteps=50_000, plotter=None, color=None, seed=None):
    """Train a PPO agent with given parameters."""
    env = make_env(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        n_steps=params["n_steps"],
        batch_size=params["batch_size"],
        gamma=params["gamma"],
        gae_lambda=params["gae_lambda"],
        clip_range=params["clip_range"],
        ent_coef=params["ent_coef"],
        vf_coef=params["vf_coef"],
        max_grad_norm=params["max_grad_norm"],
        verbose=0,
        seed=seed,
    )

    # Use shared plotter if provided
    if plotter is None:
        plotter = SharedLivePlot("Tune PPO")

    live_callback = LivePlotCallback(plotter, name=trial_name, color=color, plot_interval=5)

    model.learn(total_timesteps=total_timesteps, callback=live_callback)

    env.close()
    return model


def objective(trial, shared_plotter=None):
    """Optuna objective function for tuning PPO hyperparameters."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.90, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),
    }

    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    seeds = [42, 123, 321]
    rewards = []

    for seed in seeds:
        model = train_ppo(
            params,
            trial_name=f"{trial_name}_seed{seed}",
            total_timesteps=50_000,
            plotter=shared_plotter,
            color=color,
            seed=seed,
        )
        env = make_env(seed=seed)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()
        rewards.append(mean_reward)

    avg_reward = np.mean(rewards)


    return avg_reward


def main():
    """Main entry point for Optuna study."""
    shared_plotter = SharedLivePlot("Tune PPO")

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