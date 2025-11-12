from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList
from multi_target_env import MultiTargetEnv
from train_agent import LivePlotCallback, SharedLivePlot

def train_dqn(params, trial_name="Trial", total_timesteps=50000, plotter=None, color="tab:green", seed=None):
    """
    Train a single DQN model with the given hyperparameters.
    Returns the trained model and final evaluation score.
    """

    mode = "track"

    # Create environment
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=seed, mode=mode)

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
    )

    # Use shared plotter if provided
    if plotter is None:
        plotter = SharedLivePlot("Tune DQN")

    live_callback = LivePlotCallback(plotter, name=trial_name, color=color, plot_interval=5)

    model.learn(total_timesteps=total_timesteps, callback=live_callback)

    env.close()
    return model