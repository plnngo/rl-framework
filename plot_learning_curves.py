import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem, t
from scipy.interpolate import interp1d

# === CONFIG ===
save_dir = "results"
algos = ["PPO", "DQN", "Random"]
#algos = ["MCTS"]
#algos = ["DQN", "Random"]
mode = "search"
alpha = 0.05  # 95% CI
colors = {"PPO": "tab:orange", "DQN": "tab:green", "Random": "tab:red", "MCTS": "tab:blue"}

# === Load CSVs and align cumulative timesteps ===
all_data = {}
max_timesteps = 0

for algo in algos:
    pattern = os.path.join(save_dir, f"{algo.lower()}_{mode}_seed*.csv")
    files = sorted(glob.glob(pattern))
    seeds_data = []
    seeds_timesteps = []

    for f in files:
        df = pd.read_csv(f)
        seeds_data.append(df["episode_reward"].values)
        seeds_timesteps.append(df["cumulative_timestep"].values)
        max_timesteps = max(max_timesteps, df["cumulative_timestep"].iloc[-1])

    all_data[algo] = {"rewards": seeds_data, "timesteps": seeds_timesteps}

# === Define common x-axis for interpolation ===
common_timesteps = np.linspace(0, max_timesteps, 1000)  # adjust resolution if needed

# === Compute mean and 95% CI using interpolation ===
def interpolate_rewards(t, rewards, x_new):
    """Interpolate episode-level rewards onto a common x-axis."""
    y_interp = []
    for ts, r in zip(t, rewards):
        f = interp1d(ts, r, kind='previous', bounds_error=False, fill_value=(r[0], r[-1]))
        y_interp.append(f(x_new))
    return np.array(y_interp)

plt.figure(figsize=(10, 6))

for algo in algos:
    rewards = all_data[algo]["rewards"]
    timesteps = all_data[algo]["timesteps"]

    # Interpolate onto common x-axis
    aligned_rewards = interpolate_rewards(timesteps, rewards, common_timesteps)

    # Compute mean and CI
    mean = np.mean(aligned_rewards, axis=0)
    se = sem(aligned_rewards, axis=0)
    ci = se * t.ppf(1 - alpha/2, len(aligned_rewards)-1)

    plt.plot(common_timesteps, mean, label=algo, color=colors[algo])
    plt.fill_between(common_timesteps, mean - ci, mean + ci, color=colors[algo], alpha=0.2)

plt.xlabel("Cumulative timestep")
plt.ylabel("Episode reward")
plt.title("Learning curves for each algorithm")
plt.legend()
plt.tight_layout()
plt.show()