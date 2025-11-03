# tune_dqn.py
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from train_agent import SharedLivePlot
from train_dqn import train_dqn
from multi_target_env import MultiTargetEnv

def objective(trial):
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "buffer_size": trial.suggest_int("buffer_size", 20000, 200000, step=20000),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "gamma": trial.suggest_uniform("gamma", 0.90, 0.999),
        "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0.05, 0.5),
        "target_update_interval": trial.suggest_int("target_update_interval", 500, 5000, step=500),
        "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
    }

    trial_name = f"trial_{trial.number}"

    # Train model (with live plotting)
    model = train_dqn(params, trial_name=trial_name, total_timesteps=1000)

    # Evaluate after training
    mode = "search"
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, mode=mode)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    # Return the metric to maximize
    return mean_reward

if __name__ == "__main__":
    # Create a shared plotter for all trials
    shared_plotter = SharedLivePlot("Tune DQN")

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "buffer_size": trial.suggest_int("buffer_size", 20000, 200000, step=20000),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.5),
            "target_update_interval": trial.suggest_int("target_update_interval", 500, 5000, step=500),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8, 16]),
        }

        trial_name = f"trial_{trial.number}"

        # Pick a unique color for the trial (using a colormap)
        import matplotlib.cm as cm
        color = cm.get_cmap("tab20")(trial.number % 20)

        # Train model with shared plotter
        model = train_dqn(params, trial_name=trial_name, total_timesteps=50000, plotter=shared_plotter, color=color)

        # Evaluate after training
        mode = "search"
        env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, mode=mode)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        env.close()

        return mean_reward

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    print(study.best_trial.params)

    # Keep the final plot open
    shared_plotter.finalize()