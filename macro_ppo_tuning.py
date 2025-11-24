import numpy as np
import optuna
import matplotlib.cm as cm

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from MacroEnv import MacroEnv
from multi_target_env import MultiTargetEnv
from train_agent import LivePlotCallback, SharedLivePlot


# ============================================================================
# ENV CREATION
# ============================================================================

def make_macro_env(n_targets=5, n_unknown_targets=100, fov_size=2.0, seed=None):
    """
    Create the MacroEnv, wrapped in DummyVecEnv for PPO compatibility.
    PPO can also work without VecEnv, but this ensures consistency.
    """

    def _init():
        # Micro-envs for loading PPO policies
        env_search = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="search"
        )
        env_track = MultiTargetEnv(
            n_targets=n_targets, n_unknown_targets=n_unknown_targets,
            seed=seed, mode="track"
        )

        # Pretrained PPO micro agents
        search_agent = PPO.load("agents/ppo_search_trained", env=env_search)
        track_agent = PPO.load("agents/ppo_track_trained", env=env_track)

        env = MacroEnv(
            n_targets=n_targets,
            n_unknown_targets=n_unknown_targets,
            fov_size=fov_size,
            search_agent=search_agent,
            track_agent=track_agent,
        )
        return env

    return DummyVecEnv([_init])


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_macro_ppo(params, trial_name, total_timesteps=80_000, plotter=None, color=None, seed=None):
    """
    Train PPO on the MacroEnv using provided hyperparameters.
    """

    env = make_macro_env(seed=seed)

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
        policy_kwargs=params["policy_kwargs"],
        verbose=0,
        seed=seed,
    )

    if plotter is None:
        plotter = SharedLivePlot("Tune Macro-PPO")

    live_callback = LivePlotCallback(
        plotter,
        name=trial_name,
        color=color,
        plot_interval=5
    )

    model.learn(total_timesteps=total_timesteps, callback=live_callback)

    env.close()
    return model


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective_macro_ppo(trial, shared_plotter=None):

    # Hyperparameter search space
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "n_steps": trial.suggest_categorical("n_steps", [128, 256, 512, 1024]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_float("gamma", 0.92, 0.9999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.05),
        "vf_coef": trial.suggest_float("vf_coef", 0.1, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 1.0),

        "policy_kwargs": dict(
            net_arch=trial.suggest_categorical(
                "net_arch", [(64, 64), (128, 128), (256, 256)]
            ),
        ),
    }

    trial_name = f"trial_{trial.number}"
    color = cm.get_cmap("tab20")(trial.number % 20) if shared_plotter else None

    seeds = [42, 123, 321]
    rewards = []

    for seed in seeds:
        model = train_macro_ppo(
            params,
            trial_name=f"{trial_name}_seed{seed}",
            total_timesteps=60_000,
            plotter=shared_plotter,
            color=color,
            seed=seed,
        )

        eval_env = make_macro_env(seed=seed)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
        eval_env.close()
        rewards.append(mean_reward)

    return float(np.mean(rewards))


# ============================================================================
# MAIN
# ============================================================================

def main():
    shared_plotter = SharedLivePlot("Tune Macro-PPO")

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner()
    )

    study.optimize(lambda t: objective_macro_ppo(t, shared_plotter), n_trials=15)

    print("Best trial:")
    print(study.best_trial.params)

    shared_plotter.finalize()


if __name__ == "__main__":
    main()
