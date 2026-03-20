import copy
import random
import time
from matplotlib.colors import to_rgba
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import KalmanFilter
from LSBatchFilter import Fx_cv_cont, batch_estimate_single_target, f_cv_cont, generate_truth_states, plot_errors_and_sigmas, process_estimates, f_ct_linear, Fx_ct_linear
from deterministic_tracker import select_best_action_IG, select_best_action_pFOV
from multi_seed_training import RandomSeedEnv
from multi_target_env import MultiTargetEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from matplotlib.patches import Ellipse, Rectangle
from scipy.integrate import solve_ivp
from collections import defaultdict

def plot_cov_ellipse(cov, mean, ax, n_std=1.0, **kwargs):
    """Plot an ellipse representing the covariance matrix cov centered at mean."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ell)


def visualize_initial_positions(env):
    """Plot the initial positions of known and unknown targets."""
    fig, ax = plt.subplots()

    # Draw discretized grid
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    # Known targets (blue)
    for tgt in env.targets:
        ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target")
        #plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="blue", alpha=0.3)

    # Unknown targets (orange)
    for utgt in env.unknown_targets:
        ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target")
        #plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="orange", alpha=0.3)

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Initial Target Positions with Grid Overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    # Combine duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    plt.show()


def run_random_policy_search(env, n_steps):
    """Task sensor to search in grid cells corresponding to initial unknown target positions."""
    #obs = env.reset()
    positions, covariances = [], []

    fov_half = env.fov_size / 2.0

    # Find grid cells closest to unknown targets
    search_indices = []
    for utgt in env.unknown_targets:
        pos = utgt["x"][:2]
        pos = np.array([pos[0] + 1, pos[1]])
        dists = np.linalg.norm(env.grid_coords - pos, axis=1)
        grid_idx = np.argmin(dists)
        search_indices.append(grid_idx)

    print("Grid indices for initial unknown targets:", search_indices)

    for step, grid_idx in enumerate(search_indices):
        search_pos = env.grid_coords[grid_idx]
        action = {"macro": 0, "micro_search": grid_idx}
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step+1:02d}: Search at {search_pos}, Reward={reward:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw all grid cells
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        # Draw current FOV (green)
        fov_rect = Rectangle(
            (search_pos[0] - fov_half, search_pos[1] - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="green",
            facecolor="none",
            linestyle="--",
            lw=2,
        )
        ax.add_patch(fov_rect)

        # Plot known (blue) + unknown (orange)
        for tgt in env.targets:
            ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target" if step == 0 else "")
        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")

        # Draw uncertainty ellipses 
        for tgt in env.targets:
            plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red", alpha=0.3)
        for utgt in env.unknown_targets:
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="red", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: Search at {search_pos}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        # Combine legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        plt.show(block=True)  # <-- opens one figure per step, waits until closed

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    plt.show()
    return np.array(positions), np.array(covariances)

def run_random_policy_track(env, n_steps):
    """
    Executes only TRACK macro-actions with random valid known targets at each step.
    Visualizes the state and uncertainty after every tracking action.
    """
    positions, covariances = [], []
    figures = []  # store figure handles
    exceed_target = []

    for step in range(n_steps):
        valid_ids = np.flatnonzero(env.known_mask)
        if len(valid_ids) == 0:
            print("No known targets available for tracking at step", step)
            break

        #action = int(env.rng.choice(valid_ids))
        action = 1
        obs, reward, done, truncated, info = env.step(action)
        if len(info["lost_target"]) > 0:
            print(f"Step {step+1:02d}: lost target")
        """ for tgt in range(env.n_targets):
            exceed, x, P = analyse_tracking_task(obs, tgt, env, confidence=0.95)
            if exceed:
                exceed_target.append([tgt, step]) """


        print(f"Step {step+1:02d}: TRACK target {action}, Reward={reward:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))
        figures.append(fig)  # keep it open!

        # (your plotting code remains unchanged)
        fov_half = env.fov_size / 2.0
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        for tgt in env.targets:
            if tgt["id"] == action:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="red", s=120, marker="*", label="Tracked Target")
            else:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", s=40, marker="o", label="Known Target" if step == 0 else "")
        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")

        for tgt in env.targets:
            plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red" if tgt["id"] == action else "blue", alpha=0.5)
        for utgt in env.unknown_targets:
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="red", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: TRACK target {action}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    # --- Show all figures together at the end ---
    print(env.n_targets)
    #plt.show()
    return np.array(positions), np.array(covariances)

def run_random_policy_combined(env, n_steps):
    """
    Random policy combining SEARCH (macro=0) and TRACK (macro=1) actions.
    Visualizes the environment state with uncertainty ellipses each step.
    Highlights currently tracked target with red marker.
    """
    positions, covariances = [], []
    fov_half = env.fov_size / 2.0

    # Find grid indices closest to unknown targets for SEARCH
    search_indices = []
    for utgt in env.unknown_targets:
        pos = utgt["x"][:2]
        pos = np.array([pos[0] + 1, pos[1]])  # optional x offset
        dists = np.linalg.norm(env.grid_coords - pos, axis=1)
        grid_idx = np.argmin(dists)
        search_indices.append(grid_idx)

    print("Grid indices for initial unknown targets:", search_indices)

    for step in range(n_steps):
        # Choose macro action randomly: 0=SEARCH, 1=TRACK
        macro = env.rng.choice([0, 1])

        if macro == 0 and len(search_indices) > 0:
            # SEARCH action sampling
            grid_idx = search_indices[step % len(search_indices)]
            action = {"macro": 0, "micro_search": grid_idx, "micro_track": 0}
            current_tracked = None
            print(f"Step {step+1:02d}: SEARCH at grid cell {grid_idx} pos {env.grid_coords[grid_idx]}")

        else:
            # TRACK action sampling
            valid_ids = np.where(env.known_mask)[0]
            if len(valid_ids) == 0:
                # Fallback to SEARCH if no known targets
                if len(search_indices) > 0:
                    grid_idx = search_indices[step % len(search_indices)]
                    action = {"macro": 0, "micro_search": grid_idx, "micro_track": 0}
                    current_tracked = None
                    print(f"Step {step+1:02d}: Fallback SEARCH at {grid_idx}")
                else:
                    print(f"Step {step+1:02d}: No known targets for TRACK or search indices, aborting.")
                    break
            else:
                target_id = int(env.rng.choice(valid_ids))
                action = {"macro": 1, "micro_search": 0, "micro_track": target_id}
                current_tracked = target_id
                print(f"Step {step+1:02d}: TRACK target {target_id}")

        obs, reward, done, truncated, info = env.step(action)

        # Visualization
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw all grid cells
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)

        # If SEARCH, draw FOV rectangle in green
        if action["macro"] == 0:
            search_pos = env.grid_coords[action["micro_search"]]
            fov_rect = Rectangle(
                (search_pos[0] - fov_half, search_pos[1] - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                lw=2,
            )
            ax.add_patch(fov_rect)

        # Plot known and unknown targets with colour coding
        for tgt in env.targets:
            if current_tracked is not None and tgt["id"] == current_tracked:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="red", s=120, marker="*", label="Tracked Target")
                plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red", alpha=0.7)
            else:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", s=40, marker="o", label="Known Target" if step == 0 else "")
                plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="blue", alpha=0.3)

        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="orange", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal")
        ax.set_title(f"Step {step+1}: {'TRACK target ' + str(current_tracked) if current_tracked is not None else 'SEARCH'}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        plt.show(block=True)

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    return np.array(positions), np.array(covariances)


def plot_results(env, positions, covariances):
    """Plot 2D trajectories and uncertainty ellipses for known and unknown targets."""
    fig, ax = plt.subplots()

    n_known = env.n_targets
    n_unknown = env.n_unknown_targets

    # Plot known targets (blue)
    for i in range(n_known):
        ax.plot(positions[:, i, 0], positions[:, i, 1], color="blue", label=f"Known {i}")
        for step in range(len(positions)):
            plot_cov_ellipse(
                covariances[step, i],
                positions[step, i],
                ax,
                n_std=1.0,
                edgecolor="blue",
                alpha=0.3,
            )

    # Plot unknown targets (orange)
    for j in range(n_unknown):
        idx = n_known + j
        ax.plot(positions[:, idx, 0], positions[:, idx, 1], color="orange", label=f"Unknown {j}")
        for step in range(len(positions)):
            plot_cov_ellipse(
                covariances[step, idx],
                positions[step, idx],
                ax,
                n_std=1.0,
                edgecolor="orange",
                alpha=0.3,
            )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Target Trajectories (Known + Unknown)")
    ax.legend()
    ax.grid(True)
    plt.show()

def _unpack_cholesky(ch_pack, d):
    """
    Unpack a lower-triangular matrix L (d x d) from the packed vector ch_pack.
    Packing assumed row-by-row for lower triangle:
      for i in range(d):
          for j in range(i+1):
              append(L[i, j])
    Returns L (d x d).
    """
    L = np.zeros((d, d), dtype=float)
    idx = 0
    for i in range(d):
        for j in range(i + 1):
            L[i, j] = ch_pack[idx]
            idx += 1
    return L

def extract_target_state_cov(target_idx, env):
    """
    Extract both the d-dimensional state vector and its corresponding dxd covariance
    matrix for a given target index from the flat observation vector `obs`.

    Parameters
    ----------
    target_idx : int
        Index of the target to extract (0-based).
    env : MultiTargetEnv
        Environment instance containing dimensional parameters.

    Returns
    -------
    x : np.ndarray
        State vector of shape (d_state,).
    P : np.ndarray
        Covariance matrix of shape (d_state, d_state).
    """
    x = None
    P = None
    # Extract state vector (first d_state elements)
    for tgt in env.targets:
        if tgt["id"] == target_idx:
            x = tgt["x"]
            P = tgt["P"]
            break

    return x, P

@staticmethod
def analyse_tracking_task(target_idx, env, confidence=0.95):
    """
    Test whether the 2D positional covariance (first two state dims) for target
    exceeds the env.fov_size.

    By default this uses the 95% confidence ellipse (chi2 = 5.991 for 2D).
    It computes the semi-major axis = sqrt(chi2 * lambda_max) where lambda_max
    is the largest eigenvalue of the 2x2 positional covariance.
    We consider the target's covariance to 'exceed' the FOV when the FULL-length
    major axis (2 * semi_major) is larger than env.fov_size.

    Return: 
    True/False. If the target is masked (mask <= 0.5) then returns False.
    x : np.ndarray
        State vector of shape (d_state,).
    P : np.ndarray
        Covariance matrix of shape (d_state, d_state).
    """
    # check mask: mask vector placed after all per-target blocks
    """ mask_offset = int(env.max_targets * env.obs_dim_per_target)
    mask_val = obs[mask_offset + int(target_idx)]
    if mask_val <= 0.5:
        return True, None, None """

    x, P = extract_target_state_cov(target_idx, env)
    if x is None:
        return False, None , None
    # positional covariance assumed to be in state dims [0,1] (x,y)
    posP = P[:2, :2]

    # handle degenerate / non-PD posP
    # force symmetric
    posP = 0.5 * (posP + posP.T)
    # small regularization if needed
    try:
        eigvals = np.linalg.eigvalsh(posP)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvals(posP).real
    # clip negative tiny eigenvalues to zero
    eigvals = np.clip(eigvals, 0.0, None)
    lambda_max = np.max(eigvals)

    # chi-square value for 2D at given confidence:
    # 0.68 -> 2.279, 0.90 -> 4.605, 0.95 -> 5.991, 0.99 -> 9.210
    # Here we use the 95% default
    if confidence == 0.95:
        chi2_val = 5.991
    elif confidence == 0.90:
        chi2_val = 4.605
    elif confidence == 0.99:
        chi2_val = 9.210
    else:
        # approximate general quantile using scipy would be best, but avoid dependency:
        # for uncommon values fallback to 95% constant
        chi2_val = 5.991

    semi_major = np.sqrt(chi2_val * lambda_max)  # semi-major axis length
    full_major = 2.0 * semi_major

    return full_major > float(env.fov_size), x, P

def evaluate_agent_track(env, model=None, n_episodes=1, random_policy=False, deterministic_policy=False, deterministic_policy_alternative = False, seed=None, maskable=False, fov=4):
    rewards = []
    exceedFOV = []
    illegal_actions = []

    # for logging the final episode
    last_episode_log = {}
    last_env = None

    for ep in trange(n_episodes, desc="Evaluating"):
        obs = env.obs
        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_env = copy.deepcopy(env)
            episode_log = {}        # create a temporary log if this is the last episode
        done = False
        total_reward = 0.0
        illegal_action = 0.0
        t = 0  # timestep counter

        while not done:
            # --- Choose action ---
            if random_policy:
                #action = env.action_space.sample()
                action = np.random.randint(0, 5, dtype=np.int64)
            elif deterministic_policy:
                action, best_ig, best_update = select_best_action_pFOV(env, env.dt, fov)
            elif deterministic_policy_alternative:
                action, best_ig, best_update = select_best_action_IG(env, env.dt)
                #action = 4
            elif maskable:
                action_masks = env.action_masks()
                action, _ = model.predict(obs, action_masks=action_masks)

            else:
                action, _ = model.predict(obs, deterministic=False)

            # --- Step environment ---
            obs, reward, done, truncated, info = env.step(action)
            if not info["action_mask"]["micro_track"][np.asarray(action).item()]:
                illegal_action = illegal_action + 1
            total_reward += reward

            # --- Analyse targets ---
            if ep == n_episodes - 1:
                episode_log[t] = {}

            if ep == n_episodes - 1:
                for tgt in env.targets:
                    exceed, x, P = analyse_tracking_task(tgt["id"], env, confidence=0.95)

                    # if this is the last episode, store everything
                    if not info["action_mask"]["micro_track"][np.asarray(action).item()]:
                        """ if info.get("invalid_action"):
                            illegalActs += 1 """
                        continue
                    if tgt["id"] in info["target_id"]:
                        if x is None:
                            print("error")
                        episode_log[t][tgt["id"]] = {
                            "id": tgt["id"],
                            "state": x.copy(),
                            "cov": P.copy(),
                        }

            t += 1  # increment timestep

        rewards.append(total_reward)
        exceedFOV.append(env.init_n_targets-env.n_targets)
        illegal_actions.append(illegal_action)

        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_episode_log = episode_log
        else:
            obs, _ = env.reset(seed=seed)


    return rewards, exceedFOV, last_env, last_episode_log, illegal_actions

def visualize_search_pointing_heatmap(env, pointing_history):
    """
    Plot grid and overlay SEARCH actions colored by timestep.
    """
    fig, ax = plt.subplots()

    # Draw grid
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    if len(pointing_history) > 0:
        grid_indices, timesteps = zip(*pointing_history)
        grid_positions = np.array([env.grid_coords[i] for i in grid_indices])

        sc = ax.scatter(
            grid_positions[:, 0],
            grid_positions[:, 1],
            c=timesteps,
            cmap="viridis",
            s=60,
            alpha=0.9,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Timestep")

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    #ax.set_title("Agent Pointing Over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.show()

def visualize_unknown_target_heatmap(env, unknown_target_history):
    """
    Plot unknown target positions over time using a time-normalized colormap.
    """
    fig, ax = plt.subplots()

    # Optional: draw grid for reference
    fov_half = env.fov_size / 2.0
    for gx, gy in env.grid_coords:
        rect = Rectangle(
            (gx - fov_half, gy - fov_half),
            env.fov_size,
            env.fov_size,
            edgecolor="lightgray",
            facecolor="none",
            linewidth=0.5,
        )
        ax.add_patch(rect)

    if len(unknown_target_history) > 0:
        data = np.array(unknown_target_history)
        xs, ys, ts = data[:, 0], data[:, 1], data[:, 2]

        sc = ax.scatter(
            xs,
            ys,
            c=ts,
            cmap="viridis",   # distinct from agent heatmap
            s=60,
            alpha=0.9,
        )

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Timestep")

    ax.set_xlim(-env.space_size / 2, env.space_size / 2)
    ax.set_ylim(-env.space_size / 2, env.space_size / 2)
    ax.set_aspect("equal", "box")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)

    #ax.set_title("Unknown Target Positions Over Time")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    plt.show()

def evaluate_agent_search(env, model=None, n_episodes=100, random_policy=False, seed=None):
    """
    Evaluates an RL agent or random policy on the given environment,
    tracking episode rewards and detection counts.

    Detection count is inferred by comparing the evolution of the action mask:
    whenever a new target becomes trackable (mask bit switches from 0 to 1),
    we count it as a detection.

    Parameters:
        env: MultiTargetEnv instance
        model: trained SB3 model (PPO, DQN, etc.)
        n_episodes: number of episodes
        random_policy: if True, sample random actions instead of using the model

    Returns:
        rewards: list of total rewards per episode
        detections: list of detection counts per episode
    """
    rewards = []
    detection_count = []

    for ep in trange(n_episodes, desc="Evaluating"):

        obs, _ = env.reset(seed=int(np.random.choice(seed)))
        done = False
        total_reward = 0.0
        pointing_history = []  # list of (grid_idx, timestep)
        unknown_target_history = []  
        t = 0

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, done, truncated, info = env.step(action)
            macro, micro_search, micro_track = env.decode_action(action)
            pointing_history.append((micro_search, t))
            for utgt in env.unknown_targets:
                x, y = utgt["x"][0], utgt["x"][1]
                unknown_target_history.append((x, y, t))
            total_reward += reward
            """ # Compare action mask with previous one to detect new trackable targets
            known_targets = sum(info["action_mask"]["micro_track"])
            if known_targets>detections:
                detect_count3 = detect_count3 + (known_targets - detections)
                detections = known_targets """
            t=t+1
        if ep == n_episodes - 1:
            visualize_search_pointing_heatmap(env, pointing_history)
            visualize_unknown_target_heatmap(env, unknown_target_history)
        rewards.append(total_reward)
        detection_count.append(env.detect_counter)

    return rewards, detection_count

@staticmethod
def plot_violin(results_dict, ylabel="Episode Reward"):
    """
    Plots a violin plot comparing metrics (e.g., rewards or detections) across agents.
    """
    colors = {
        "PPO": "blue",
        "DQN": "orange",
        "Random": "red",
        "Heuristic": "green",
        "MCTS": "purple",
        "Maskable PPO": "grey",
        "0.95": "blue",
        "0.5": "orange",
        "0.1": "red"
    }
    
    data = []
    labels = []
    for label, values in results_dict.items():
        data.extend(values)
        labels.extend([label] * len(values))

    sns.violinplot(x=labels, y=data, inner="quart", cut=0, palette=colors)
    plt.xlabel("Agent")
    plt.ylabel(ylabel)
    plt.title(f"Distribution of {ylabel}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

def visualize_trained_agent(env, model, n_steps=20):
    """Run the trained agent and visualize its decisions and environment state."""
    obs, _ = env.reset()
    fov_half = env.fov_size / 2.0

    for step in range(n_steps):
        # --- Get action from trained DQN ---
        action, _ = model.predict(obs, deterministic=False)
        macro, micro_search, micro_track = env.decode_action(action)

        # --- Apply action in environment ---
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- Create figure ---
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: Reward={reward:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # --- Draw grid ---
        for gx, gy in env.grid_coords:
            rect = Rectangle(
                (gx - fov_half, gy - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="lightgray",
                facecolor="none",
                linewidth=0.5,
            )
            ax.add_patch(rect)
        
        """ # --- Draw reward window ---
        rw_half = env.reward_window_size / 2.0
        rw_center = env.reward_window_center
        reward_rect = Rectangle(
            (rw_center[0] - rw_half, rw_center[1] - rw_half),
            env.reward_window_size,
            env.reward_window_size,
            edgecolor="magenta",
            facecolor="none",
            linestyle="-",
            linewidth=2,
            label="Reward Window"
        )
        ax.add_patch(reward_rect) """

        # --- Visualize the agent’s chosen action ---
        if macro == 0:  # SEARCH
            search_pos = env.grid_coords[micro_search]
            fov_rect = Rectangle(
                (search_pos[0] - fov_half, search_pos[1] - fov_half),
                env.fov_size,
                env.fov_size,
                edgecolor="green",
                facecolor="none",
                linestyle="--",
                lw=2,
                label="Search FOV",
            )
            ax.add_patch(fov_rect)
            print(f"Step {step+1:02d}: Search at {search_pos}, Reward={reward:.4f}")
        else:  # TRACK
            target_id = micro_track
            tgt = env.targets[target_id]
            ax.scatter(
                tgt["x"][0], tgt["x"][1],
                c="red", s=120, marker="*",
                label=f"Tracked target {target_id}"
            )
            print(f"Step {step+1}: TRACK target {target_id}, Reward={reward:.4f}")

        # --- Plot targets ---
        for tgt in env.targets:
            ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target")

            # Plot 2D covariance ellipse if available
            if "P" in tgt:
                P_xy = tgt["P"][:2, :2]  # take only position covariance
                plot_cov_ellipse(P_xy, tgt["x"][:2], ax, edgecolor="blue", alpha=0.4)

        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="lower left")

        #plt.show(block=True)
        plt.pause(4)
        plt.close(fig)

        if done:
            obs, _ = env.reset()

@staticmethod
def plot_detection_bar_chart(results_dict):
    """
    Plots a bar chart showing the mean number of detections (count 2) per agent.

    Parameters:
        results_dict: dict mapping agent name -> list of detection counts per episode
                      e.g. {"Random": random_detections, "PPO": ppo_detections, "DQN": dqn_detections}
    """
    colors = {
        "PPO": "orange",
        "DQN": "green",
        "Random": "red"
    }

    agents = list(results_dict.keys())
    means = [np.mean(results_dict[a]) for a in agents]
    stds = [np.std(results_dict[a]) for a in agents]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(agents, means, yerr=stds, capsize=6,
                   color=[colors[a] for a in agents], alpha=0.8)

    plt.xlabel("Agent")
    plt.ylabel("Total Number of Detections")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)

    # Add value labels on top of bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                 f"{mean:.2f}", ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_positions(positions, env=None, show_start_end=True):
    """
    Plots the target positions over time on an x-y field.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (n_steps, n_targets, 2), containing x and y positions.
    env : optional
        Environment object, used to set plot limits if provided.
    show_start_end : bool, default=True
        If True, mark start and end positions for each target.
    """
    n_steps, n_targets, _ = positions.shape

    plt.figure(figsize=(8, 8))
    
    # Plot each target's trajectory
    for i in range(n_targets):
        traj = positions[:, i, :]
        plt.plot(traj[:, 0], traj[:, 1], '-', lw=2, label=f'Target {i}')
        if show_start_end:
            plt.scatter(traj[0, 0], traj[0, 1], c='green', marker='o', s=60)  # Start
            plt.scatter(traj[-1, 0], traj[-1, 1], c='red', marker='x', s=80)  # End
    
    # Add grid / field visualization
    if env is not None:
        plt.xlim(-env.space_size / 2, env.space_size / 2)
        plt.ylim(-env.space_size / 2, env.space_size / 2)
    else:
        all_x = positions[:, :, 0].flatten()
        all_y = positions[:, :, 1].flatten()
        plt.xlim(np.min(all_x) - 10, np.max(all_x) + 10)
        plt.ylim(np.min(all_y) - 10, np.max(all_y) + 10)

    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.title("Target Trajectories over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

@staticmethod
def plot_means_lost_targets(ppo, knownPPO, dqn = [], knownDQN =[], random=[], knownRandom =[], det=[], knowndet=[], labels=None):
    """
    ppo, dqn, random are lists of arrays.
    Compute:
        - global mean over all entries
        - global std over all entries
    knownPPO, knownDQN, knownRandom:
        - compute simple means (kept unchanged)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    # --- Compute statistics ---
    means = np.array([
        np.mean([np.mean(arr) for arr in ppo]),
        np.mean([np.mean(arr) for arr in dqn]),
        np.mean([np.mean(arr) for arr in random]),
        np.mean([np.mean(arr) for arr in det])
    ])

    stds = np.array([
        np.std([np.mean(arr) for arr in ppo], ddof=1),
        np.std([np.mean(arr) for arr in dqn], ddof=1),
        np.std([np.mean(arr) for arr in random], ddof=1),
        np.std([np.mean(arr) for arr in det], ddof=1)
    ])

    # --- Known means remain unchanged ---
    known_means = np.array([
        np.mean(knownPPO),
        np.mean(knownDQN),
        np.mean(knownRandom),
        np.mean(knowndet)
    ])

    # --- Plotting ---
    x = np.arange(len(labels))
    colors = ["blue", "orange", "red", "green"]
    light_colors = [to_rgba(c, alpha=0.35) for c in colors]

    plt.figure(figsize=(7, 5))

    # Baseline bars (tracking targets)
    baseline_bars = plt.bar(
        x, known_means,
        width=0.6,
        color=light_colors,
        label="Number of tracking targets",
        zorder=1
    )

    for bar, value in zip(baseline_bars, known_means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black"
        )

    # Main bars (lost targets, with std)
    bars = plt.bar(
        x, means,
        yerr=stds,
        capsize=8,
        color=colors,
        label="Number of lost targets",
        zorder=2
    )

    # Bar height labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom"
        )

    plt.xticks(x, labels)
    plt.ylabel("Mean Lost Targets")
    plt.title("Mean Lost Targets Across Agents (Global Mean + Std)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def unpack_cholesky(L_flat, dim):
    """
    Convert a flat vector of length dim*(dim+1)/2 into a lower-triangular matrix.
    """
    L = np.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        for j in range(i + 1):
            L[i, j] = L_flat[idx]
            idx += 1
    return L

def propagate_known_targets_over_episode(last_env):
    """
    Extract all known targets from last_env and propagate their states forward
    over one episode duration using the correct motion model and parameters.

    Returns:
        traj (dict):
            {
                t: {tgt_id: x_t (4D vector)},
                ...
            }
    """
    # Episode duration in steps
    T = last_env.max_steps
    dt = last_env.dt
    rng = last_env.rng

    # Known targets are those with known_mask == True
    known_ids = np.where(last_env.known_mask)[0]

    # storage: trajectories[t][id] = position vector
    traj = {}

    # Extract initial states for each known target
    # Observation encoding: consecutive blocks of obs_dim_per_target
    obs = last_env._get_obs()   # use env’s getter (better than last obs)
    d = last_env.obs_dim_per_target
    targets = {}

    for tgt_id in known_ids:
        # Extract the slice for that target
        start = tgt_id * d
        end   = start + d

        block = obs[start:end]

        # First d_state entries are the mean vector x
        x0 = block[:last_env.d_state].copy()

        # Next entries represent the Cholesky factor L packed row-wise
        chol_size = last_env.cholesky_size
        L_flat = block[last_env.d_state:last_env.d_state + chol_size]
        L = unpack_cholesky(L_flat, last_env.d_state)
        P0 = L @ L.T

        # Store the current state for propagation
        targets[tgt_id] = {
            "x": x0,
            "P": P0,
            "motion": last_env.motion_model[tgt_id],
            "param":  last_env.motion_params[tgt_id],
        }

    # ---- Propagation loop ----
    for t in range(T):
        traj[t] = {}

        for tgt_id in known_ids:
            tgt = targets[tgt_id]

            x, P = MultiTargetEnv.propagate_target_2D(
                tgt["x"], tgt["P"], last_env.Q0,
                dt=dt, rng=rng,
                motion_model=tgt["motion"],
                motion_param=tgt["param"]
            )

            # store updated state
            targets[tgt_id]["x"] = x
            targets[tgt_id]["P"] = P

            traj[t][tgt_id] = x.copy()

    return traj

def compute_state_differences_with_ekf(traj_pred, last_episode_log, last_env, R):
    """
    Apply EKF update to the predicted trajectories and compare with the
    logged 'posterior' states in last_episode_log.

    Inputs:
        traj_pred: dict[t][tgt_id] = predicted state
        last_episode_log: dict[t][tgt_id] = logged EKF-like state
        last_env: env instance (for P0, dynamics, dt)
        R: measurement noise covariance (2x2)

    Returns:
        diffs: dict[t][tgt_id] = (x_upd - x_pred)
        compared: dict[t][tgt_id] = (x_upd - x_log)
    """

    diffs = {}
    compared = {}

    for t in traj_pred.keys():
        diffs[t] = {}
        compared[t] = {}

        for tgt_id, x_pred in traj_pred[t].items():

            # We do not have P_pred stored, but environments usually propagate P similarly.
            # You should store P when generating traj_pred — if not, assume P0.
            # Here: fallback to P0
            P_pred = last_env.P0.copy()

            # EKF update
            x_upd, P_upd = MultiTargetEnv.ekf_update(
                x_pred.copy(),
                P_pred.copy(),
                R, 
                MultiTargetEnv.extract_measurement_bearingRange
            )

            # Difference: updated EKF - predicted
            diffs[t][tgt_id] = x_upd - x_pred

            # If we have a logged state, compare as well
            if t in last_episode_log and tgt_id in last_episode_log[t]:
                x_log = last_episode_log[t][tgt_id]["state"]
                compared[t][tgt_id] = x_upd - x_log

    return diffs, compared

def constant_obs_all_targets(estimates=None):

    tracks = {tid: [] for tid in estimates.keys()}

    for target_id, entry in estimates.items():

        if not isinstance(entry, dict):
            continue

        Xest = entry.get("Xest")
        P_mat = entry.get("P_mat")

        if Xest is None or P_mat is None:
            continue

        L = Xest.shape[1]

        for k in range(L):

            tracks[target_id].append({
                "t": k,
                "state": Xest[:, k],
                "cov": P_mat[:, :, k]
            })

    return tracks

def extract_tracks_from_log(last_episode_log, n_targets=5):
    """
    Convert last_episode_log into per-target time-ordered tracks.

    Returns:
        tracks: dict[target_id] = list of dicts with
            { "t": timestep, "state": ..., "cov": ..., "exceedFOV": ... }
    """

    # Prepare empty track containers
    tracks = {tid: [] for tid in range(n_targets)}

    # Iterate through sorted timesteps
    for t in sorted(last_episode_log.keys()):
        snapshot = last_episode_log[t]

        # snapshot might be {} if no target logged at this timestep
        if not isinstance(snapshot, dict):
            continue

        for target_id, entry in snapshot.items():
            # Safety check: skip malformed entries
            if not isinstance(entry, dict):
                continue

            tracks[target_id].append({
                "t": t,
                "state": entry.get("state"),
                "cov": entry.get("cov"),
                "exceedFOV": entry.get("exceedFOV", False),
            })

    return tracks

def estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc):
    errors_all_targets = []
    total_trace_cov = []
    KFstate = []
    KFcov = []
    #for tgt_id, track in tracks.items():
    for tgt_id in range(len(all_target_states)):
        #print("Plot errors for target " + str(tgt_id))

        # --- Extract data from track ---
        timesteps = [obs["t"] for obs in tracks[tgt_id]]
        #timesteps = np.arange(len(all_meas[tgt_id]))
        #if not timesteps and tgt_id >= last_env.init_n_targets:   # if not detected and initially unknown
        if tgt_id >= last_env.init_n_targets:   # if not detected and initially unknown
            continue
        all_tgt_meas = all_meas[tgt_id]
        tgt_meas = all_tgt_meas[timesteps, :]
        if tgt_id < last_env.init_n_targets:
            for tgt in last_env.targets.copy():
                    tid = tgt['id']
                    if tid == tgt_id:
                        motion = last_env.motion_model[tgt_id]
                        if motion == "L":
                            f_dyn = lambda x, omega=None: f_cv_cont(x)
                            Fx_dyn = lambda x, omega=None: Fx_cv_cont(x)
                            inputs = {
                                "Rk": R,
                                #"Q": np.eye(2) * 1e-27,
                                "Q": np.eye(2) * 0.,
                                "Po": tgt['P'],
                                #"omega": omega,
                                "f_dyn": f_dyn,
                                "Fx_dyn": Fx_dyn
                            }
                            integrationFcn = KalmanFilter.int_constant_velocity_stm
                            
                        else:
                            omega = last_env.motion_params[tgt_id]
                            f_dyn = lambda x, omega: f_ct_linear(x, omega)
                            Fx_dyn = lambda x, omega: Fx_ct_linear(x, omega)
                            inputs = {
                                "Rk": R,
                                #"Q": np.eye(2) * 5e-13,
                                "Q": np.eye(2) * 0,
                                "Po": tgt['P'],
                                "omega": omega,
                                "f_dyn": f_dyn,
                                "Fx_dyn": Fx_dyn
                            }
                            integrationFcn = KalmanFilter.int_constant_turn_stm_2D
                        break
        else:
            for tgt in last_env.unknown_targets.copy():
                tid = tgt['id']
                if tid == tgt_id:
                    motion = last_env.motion_model[tgt_id]
                    if motion == "L":
                        f_dyn = lambda x, omega=None: f_cv_cont(x)
                        Fx_dyn = lambda x, omega=None: Fx_cv_cont(x)
                        inputs = {
                            "Rk": R,
                            #"Q": np.eye(2) * 1e-27,
                            "Q": np.eye(2) * 0,
                            "Po": tgt['P'],
                            "f_dyn": f_dyn,
                            "Fx_dyn": Fx_dyn
                        }
                        integrationFcn = KalmanFilter.int_constant_velocity_stm
                        
                    else:
                        omega = last_env.motion_params[tgt_id]
                        f_dyn = lambda x, omega: f_ct_linear(x, omega)
                        Fx_dyn = lambda x, omega: Fx_ct_linear(x, omega)
                        inputs = {
                            "Rk": R,
                            #"Q": np.eye(2) * 5e-13,
                            "Q": np.eye(2) * 0,
                            "Po": tgt['P'],
                            "omega": omega,
                            "f_dyn": f_dyn,
                            "Fx_dyn": Fx_dyn
                        }
                        integrationFcn = KalmanFilter.int_constant_turn_stm_2D
                    break

        t_all, Xk_mat, P_mat, resids = KalmanFilter.ckf_predict_update(
                                        Xo_ref = tgt['x'],          # shape (4,)
                                        t_obs  = timesteps,         # shape (L,)
                                        tend   = last_env.max_steps,
                                        obs    = tgt_meas,          # shape (p, L)
                                        intfcn = integrationFcn,
                                        H_fcn  = obsFunc,
                                        inputs = inputs
                                    )
        all_tgt_states = all_target_states[tgt_id]
        tgt_states_with_velocity = all_tgt_states[t_all, :]
        tgt_states = tgt_states_with_velocity.T[:4, :]
        trace_P_pos = np.trace(P_mat[0:2, 0:2, :], axis1=0, axis2=1)
        if timesteps:
            selected_traces = trace_P_pos[timesteps[0]:]
        else:
            selected_traces = trace_P_pos
        total_trace_cov.append(selected_traces)

        """ plt.figure()
        plt.plot(t_all, three_sigma_pos, label="+3σ x")
        plt.plot(t_all, -three_sigma_pos, label="-3σ x")
        plt.scatter(t_all, pos_error, label="Positional error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

        """ # Skip this target if there are no timesteps/measurements
        if not timesteps:   # or: if len(timesteps) == 0:
            continue
        else: """
            
        Xk, Pk, resids = KalmanFilter.ekf(
                        Xo_ref = tgt['x'],          # shape (4,)
                        t_obs  = timesteps,         # shape (L,)
                        obs    = tgt_meas,          # shape (p, L)
                        intfcn = integrationFcn,
                        H_fcn  = obsFunc,   
                        inputs = inputs
                    )
        all_tgt_states = all_target_states[tgt_id]
        tgt_states_with_velocity = all_tgt_states[timesteps, :]
        tgt_states = tgt_states_with_velocity.T[:4, :]
        error = tgt_states - Xk
        errors_all_targets.append(error)
        """ pos_error = np.sqrt((tgt_states[0, :] - Xk[0, :])**2 + (tgt_states[1, :] - Xk[1, :])**2)
        three_sigma_x  = 3.0 * np.sqrt(Pk[0, 0, :])
        three_sigma_y  = 3.0 * np.sqrt(Pk[1, 1, :])
        three_sigma_vx = 3.0 * np.sqrt(Pk[2, 2, :])
        three_sigma_vy = 3.0 * np.sqrt(Pk[3, 3, :])
        three_sigma_pos = 3.0 * np.sqrt(Pk[0,0,:] + Pk[1,1,:])

        t = timesteps """

        KFstate.append(Xk)
        KFcov.append(Pk)

        """ plt.figure()
        plt.plot(t, three_sigma_x, label="+3σ x")
        plt.plot(t, -three_sigma_x, label="-3σ x")
        plt.scatter(t, error[0, :], label="X error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(t, three_sigma_y, label="+3σ x")
        plt.plot(t, -three_sigma_y, label="-3σ x")
        plt.scatter(t, error[1, :], label="Y error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

        """ plt.figure()
        plt.plot(t, three_sigma_pos, label="+3σ x")
        plt.plot(t, -three_sigma_pos, label="-3σ x")
        plt.scatter(t, pos_error, label="Positional error")
        plt.xlabel("Time")
        plt.ylabel("Position uncertainty")
        plt.legend()
        plt.grid(True)
        plt.show() """

    return errors_all_targets, total_trace_cov, KFstate, KFcov

def computeRMSEalgo(heuristic=False, random=False, model=None, env=None, n_episodes=100, sigma_theta=0, sigma_r=0, R=None, Q=None):
    error_episodes = []
    total_error_episodes = []
    for i in range(n_episodes):
        if heuristic or random:
            det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=1, random_policy=random, deterministic_policy=heuristic)
        elif model=="ppo":
            ppo_model = PPO.load("agents/ppo_track_trained_IEEE", env=env)
            ppo_rewards, exceedFOV_ppo, last_env, last_episode_log, illegal_actions_ppo = evaluate_agent_track(env, model=ppo_model, n_episodes=1)

        elif model=="maskableppo":
            maskppo_model = MaskablePPO.load("agents/maskableppo_track_trained_IEEE", env=env)
            maskppo_rewards, exceedFOV_maskppo, last_env, last_episode_log, illegal_actions_maskppo = evaluate_agent_track(env, model=maskppo_model, n_episodes=1, maskable=True)

        else:
            dqn_model = DQN.load("agents/dqn_track_trained_IEEE", env=env)
            dqn_rewards, exceedFOV_dqn, last_env, last_episode_log, illegal_actions_dqn = evaluate_agent_track(env, model=dqn_model, n_episodes=1)

        tracks = extract_tracks_from_log(last_episode_log)
        # Generate truth data
        timesteps = np.arange(last_env.max_steps)
        all_target_states = {}
        all_meas = {}
        for tgt_id in last_env.targets:
            tid = tgt_id["id"]
            truth_states, measurements, _ = generate_truth_states(timesteps, tid, last_env)
            all_target_states[tid] = truth_states
            measurements[:, 0] += np.random.normal(0, sigma_theta, size=len(measurements))
            measurements[:, 1] += np.random.normal(0, sigma_r, size=len(measurements))
            all_meas[tid] = measurements

        errors_all_targets, total_trace_cov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q)

        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodes.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)
        """ total_episode_error = 0
        for tgt_error in total_error_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            total_rmse_target = np.sqrt(np.mean(sq_err))
            total_episode_error = total_episode_error + total_rmse_target """
        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodes.append(total_rmse_all_target)
        env.reset()

    error_episodes = np.array(error_episodes)
    total_error_episodes = np.array(total_error_episodes)

    if len(error_episodes)>0:

        mean_pos_error_all_episodes = sum(error_episodes)/len(error_episodes)
        print("Mean of positional errors over all episodes " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodes], ddof=1))
    if len(total_error_episodes)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodes)/len(total_error_episodes)
        print("Mean of covariance trace over all episodes " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodes], ddof=1))
    return error_episodes, total_error_episodes

def efficiencyOtherPlot():
    n_targets = 5
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=42, mode="track")

    sigma_theta = np.deg2rad(1.0 / 3600.0)  # 1 arcsec bearing noise
    sigma_r = 0.1        # 10 cm range noise

    R = np.diag([sigma_theta**2, sigma_r**2])
     # Generate truth data
    timesteps = np.arange(env.max_steps)
    all_target_states = {}
    all_meas = {}
    for tgt_id in env.targets:
        tid = tgt_id["id"]
        truth_states, measurements, _ = generate_truth_states(timesteps, tid, env)
        all_target_states[tid] = truth_states
        measurements[:, 0] += np.random.normal(0, sigma_theta, size=len(measurements))
        measurements[:, 1] += np.random.normal(0, sigma_r, size=len(measurements))
        all_meas[tid] = measurements
    
    # Extract best cov guess
    estimates = {}
    efficiency = defaultdict(list)
    for k in timesteps:
        for tgt in env.targets:
            if env.motion_model[tgt["id"]] == "T":
                chosen_model = "CT"

            else: 
                chosen_model = "CV"
            state = tgt["x"]
            cov = tgt["P"]

            # Batch LS estimate run in a batch for ever new incoming measurement
            out = batch_estimate_single_target(
                    np.array(timesteps[:k+1]),
                    np.array(all_meas[tgt["id"]][:k+1]),
                    state,
                    cov,
                    R,
                    model=chosen_model,
                    omega = env.motion_params[tgt["id"]]
            )
            estimates[tgt["id"]] = out

        all_meas_trimmed = {
            tgt_id: meas[:k+1]
            for tgt_id, meas in all_meas.items()
        }
        klX, klCov = estimateAndPlot(env.targets, all_target_states, env, all_meas_trimmed, R) 

        for i in range(len(estimates)):
            P_est = estimates[i]["P_mat"][:,:,k]
            det_Pest = np.linalg.det(P_est)

            P_kl = klCov[i][:,:,k]
            det_Pkl = np.linalg.det(P_kl)
            efficiency[i].append(det_Pest / det_Pkl)
    plt.figure()

    for tgt_id, values in efficiency.items():
        plt.plot(values, label=f"Target {tgt_id}")

    plt.xlabel("Timestep")
    plt.ylabel("Efficiency (det_Pest / det_Pkl)")
    plt.title("Efficiency per Target")
    plt.legend()
    plt.grid(True)
    plt.show()

def repositionEfficiency(values, time):
    max_time = 100
    out = np.full(max_time, np.nan)
    for k in range(len(out)):

        for tid in time:

            t_array = time[tid]

            # check if timestep exists for this target
            idx = np.where(t_array == k)[0]

            if len(idx) > 0:
                i = idx[0]

                eff_val = values[tid][i]

                if np.isnan(out[k]):
                    out[k] = eff_val
                else:
                    out[k] += eff_val
    return out


def efficiencyPlot():
    n_targets = 5
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=42, mode="track")
    n_episodes = 2
    episode_efficiencies_pfov4 = []
    episode_efficiencies_pfov25 = []
    episode_efficiencies_pfov15 = []
    episode_efficiencies_pfov10 = []
    episode_efficiencies_Random = []
    episode_efficiencies_Maskppo = []
    episode_efficiencies_Ppo = []
    episode_efficiencies_Dqn = []



    episode_efficiencies_ig = []
    sigma_theta = np.deg2rad(1.0 / 3600.0)  # 1 arcsec bearing noise
    sigma_r = 0.001        # 1 mm range noise
    sigma = sigma_r

    R = np.diag([sigma**2, sigma**2])

    algorithms = ["Heuristic", "MaskPPO", "PPO", "DQN", "Random"]
    #meanEfficiencies = {alg: [] for alg in algorithms}
    # Before your episode loop, initialize timing lists
    evaluate_times_pfov4 = []
    evaluate_times_pfov25 = []
    evaluate_times_pfov15 = []
    evaluate_times_pfov10 = []
    evaluate_times_ig = []
    evaluate_times_random = []
    evaluate_times_Maskppo = []
    evaluate_times_Ppo = []
    evaluate_times_Dqn = []

    error_episodespFOV = []
    total_error_episodespFOV = []

    error_episodesIG = []
    total_error_episodesIG = []

    error_episodesRandom = []
    total_error_episodesRandom = []

    error_episodesMaskppo = []
    total_error_episodesMaskppo = []

    error_episodesPpo = []
    total_error_episodesPpo = []

    error_episodesDqn = []
    total_error_episodesDqn = []
    for i in range(n_episodes):

        # Generate truth data
        timesteps = np.arange(env.max_steps)
        all_target_states = {}
        all_meas = {}
        obsFunc = MultiTargetEnv.extract_measurement_XY
        for tgt_id in env.targets:
            tid = tgt_id["id"]
            truth_states, measurements, _ = generate_truth_states(timesteps, tid, env, obsFunc)
            all_target_states[tid] = truth_states
            measurements[:, 0] += np.random.normal(0, sigma, size=len(measurements))
            measurements[:, 1] += np.random.normal(0, sigma, size=len(measurements))
            all_meas[tid] = measurements
        
        # Extract best cov guess
        bestCov = []
        estimates = {}
        for tgt in env.targets:
            if env.motion_model[tgt["id"]] == "T":
                chosen_model = "CT"

            else: 
                chosen_model = "CV"
            state = tgt["x"]
            cov = tgt["P"]

            out = batch_estimate_single_target(
                    timesteps,
                    all_meas[tgt["id"]],
                    state,
                    cov,
                    R,
                    model=chosen_model,
                    omega=env.motion_params[tgt["id"]],
                    obsFunc=obsFunc
            )
            bestCov.append((tgt["id"], out["P0_post"]))
            estimates[tgt["id"]] = out

        """ # Heuristic track pFOV
        start = time.perf_counter()
        det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = \
            evaluate_agent_track(env, n_episodes=1, random_policy=False, deterministic_policy=True, fov=4)
        evaluate_times_pfov4.append(time.perf_counter() - start)

        #tracks = constant_obs_all_targets(estimates=estimates)
        tracks = extract_tracks_from_log(last_episode_log)

        klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc)

        efficiency_by_targetKLpFOV, t_by_target = computeEff(klCov, tracks, estimates)

        # ---- Average across targets ----
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLpFOV, t_by_target)
        episode_efficiencies_pfov4.append(mean_eff_targets)

        env = last_env
    
        # Heuristic track pFOV
        start = time.perf_counter()
        det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = \
            evaluate_agent_track(env, n_episodes=1, random_policy=False, deterministic_policy=True, fov=np.sqrt(2.5e-4))
        evaluate_times_pfov25.append(time.perf_counter() - start)

        #tracks = constant_obs_all_targets(estimates=estimates)
        tracks = extract_tracks_from_log(last_episode_log)

        klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc)

        efficiency_by_targetKLpFOV, t_by_target = computeEff(klCov, tracks, estimates)

        # ---- Average across targets ----
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLpFOV, t_by_target)
        episode_efficiencies_pfov25.append(mean_eff_targets)

        env = last_env
    
        # Heuristic track pFOV
        start = time.perf_counter()
        det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = \
            evaluate_agent_track(env, n_episodes=1, random_policy=False, deterministic_policy=True, fov=np.sqrt(1.5e-4))
        evaluate_times_pfov15.append(time.perf_counter() - start)

        #tracks = constant_obs_all_targets(estimates=estimates)
        tracks = extract_tracks_from_log(last_episode_log)

        klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc)

        efficiency_by_targetKLpFOV, t_by_target = computeEff(klCov, tracks, estimates)

        # ---- Average across targets ----
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLpFOV, t_by_target)
        episode_efficiencies_pfov15.append(mean_eff_targets)

        env = last_env """
    
        # Heuristic track pFOV
        start = time.perf_counter()
        det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = \
            evaluate_agent_track(env, n_episodes=1, random_policy=False, deterministic_policy=True, fov=np.sqrt(2.0e-8))
        evaluate_times_pfov10.append(time.perf_counter() - start)

        #tracks = constant_obs_all_targets(estimates=estimates)
        tracks = extract_tracks_from_log(last_episode_log)

        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc)
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodespFOV.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodespFOV.append(total_rmse_all_target)


    

        efficiency_by_targetKLpFOV, t_by_target = computeEff(klCov, tracks, estimates)

        # ---- Average across targets ----
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLpFOV, t_by_target)
        episode_efficiencies_pfov10.append(mean_eff_targets)
        env = last_env

        """ # Heuristic track IG
        start = time.perf_counter()
        det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=1, random_policy=False, deterministic_policy=False, deterministic_policy_alternative=True)
        evaluate_times_ig.append(time.perf_counter() - start)
        tracks = extract_tracks_from_log(last_episode_log)

        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc) 
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodesIG.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodesIG.append(total_rmse_all_target)


        efficiency_by_targetKLig, t_by_target = computeEff(klCov, tracks, estimates)
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLig, t_by_target)
        episode_efficiencies_ig.append(mean_eff_targets) """

        #env = last_env
 
        # ****** Random policy ******
        """ start = time.perf_counter()
        random_rewards, exceedFOV_random, last_env, last_episode_log, illegal_actions_random = evaluate_agent_track(env, n_episodes=1, random_policy=True, deterministic_policy=False)
        evaluate_times_random.append(time.perf_counter() - start)
        tracks = extract_tracks_from_log(last_episode_log)
        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc)
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodesRandom.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodesRandom.append(total_rmse_all_target)
        efficiency_by_targetKLRandom, t_by_target = computeEff(klCov, tracks, estimates)
        # ---- Average across targets ----
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLRandom, t_by_target)
        episode_efficiencies_Random.append(mean_eff_targets)
 
        env.reset() """
        #eff_sum = np.sum(list(efficiency_by_targetKLHeuristic.values()), axis=0)
    
        #plot_efficiency(efficiency_by_target, t_by_target)
        """ plot_efficiency(efficiency_by_targetKLHeuristic, t_by_target)
        plot_efficiency_all(efficiency_by_targetKLHeuristic, t_by_target) """
        # ****** Maskable PPO policy ******
        """ maskppo_model = MaskablePPO.load("agents/maskableppo_track_trained_IEEE_singlepFOV", env=env)
        start = time.perf_counter()
        maskppo_rewards, exceedFOV_maskppo, last_env, last_episode_log, illegal_actions_maskppo = evaluate_agent_track(env, model=maskppo_model, n_episodes=1, deterministic_policy=False, maskable=True)
        evaluate_times_Maskppo.append(time.perf_counter() - start)
        tracks = extract_tracks_from_log(last_episode_log)
        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc) 
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodesMaskppo.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodesMaskppo.append(total_rmse_all_target)


        efficiency_by_targetKLMaskPPO, t_by_target = computeEff(klCov, tracks, estimates)
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLMaskPPO, t_by_target)
        episode_efficiencies_Maskppo.append(mean_eff_targets)

        env = last_env
        # ****** PPO policy ******
        ppo_model = PPO.load("agents/ppo_track_trained_IEEE_singlepFOV", env=env)
        start = time.perf_counter()
        ppo_rewards, exceedFOV_ppo, last_env, last_episode_log, illegal_actions_ppo = evaluate_agent_track(env, model=ppo_model, n_episodes=1, deterministic_policy=False)
        evaluate_times_Ppo.append(time.perf_counter() - start)
        tracks = extract_tracks_from_log(last_episode_log)
        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc) 
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodesPpo.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodesPpo.append(total_rmse_all_target)


        efficiency_by_targetKLPPO, t_by_target = computeEff(klCov, tracks, estimates)
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLPPO, t_by_target)
        episode_efficiencies_Ppo.append(mean_eff_targets)

        env = last_env
        # ****** DQN policy ******
        dqn_model = DQN.load("agents/dqn_track_trained_IEEE_singlepFOV", env=env)
        start = time.perf_counter()
        dqn_rewards, exceedFOV_dqn, last_env, last_episode_log, illegal_actions_dqn = evaluate_agent_track(env, model=dqn_model, n_episodes=1)
        evaluate_times_Dqn.append(time.perf_counter() - start)
        tracks = extract_tracks_from_log(last_episode_log)
        errors_all_targets, total_trace_cov, klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, obsFunc) 
        episode_error = 0
        for tgt_error in errors_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            if len(sq_err) == 0:
                continue    # target has not been tracked at all
            rmse_target = np.sqrt(np.mean(sq_err))
            episode_error = episode_error + rmse_target
        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            if not np.isnan(rmse_all_target): 
                error_episodesDqn.append(rmse_all_target)
            else:
                print("error is nan")

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)

        if len(total_trace_cov)>0:
            total_rmse_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_error_episodesDqn.append(total_rmse_all_target)


        efficiency_by_targetKLDQN, t_by_target = computeEff(klCov, tracks, estimates)
        mean_eff_targets = repositionEfficiency(efficiency_by_targetKLDQN, t_by_target)
        episode_efficiencies_Dqn.append(mean_eff_targets) """

        env.reset()
        # ****** Random policy ******
        """ random_rewards, exceedFOV_random, last_env, last_episode_log, illegal_actions_random = evaluate_agent_track(env, n_episodes=1, random_policy=True, deterministic_policy=False)
        tracks = extract_tracks_from_log(last_episode_log)
        klX, klCov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q) 
        efficiency_by_targetKLRandom, t_by_target = computeEff(klCov, tracks, ls_cov_dict)
        #plot_efficiency_all(efficiency_by_targetKL, t_by_target)
        eff_sum = np.sum(list(efficiency_by_targetKLRandom.values()), axis=0)
        if meanEfficiencies["Random"] is None:
            meanEfficiencies["Random"] = eff_sum.copy()
        else:
            meanEfficiencies["Random"] += (eff_sum - meanEfficiencies["Random"]) / (i + 1) """
            
    #plot_mean_efficiencies(meanEfficiencies)
    """ episode_efficiencies_pfov4 = np.array(episode_efficiencies_pfov4)
    episode_efficiencies_pfov25 = np.array(episode_efficiencies_pfov25)
    episode_efficiencies_pfov15 = np.array(episode_efficiencies_pfov15) """
    episode_efficiencies_pfov10 = np.array(episode_efficiencies_pfov10)
    #episode_efficiencies_ig = np.array(episode_efficiencies_ig)
    #episode_efficiencies_Random = np.array(episode_efficiencies_Random)


    """ mean_pfov4 = np.mean(episode_efficiencies_pfov4, axis=0)
    std_pfov4 = np.std(episode_efficiencies_pfov4, axis=0)
    mean_pfov25 = np.mean(episode_efficiencies_pfov25, axis=0)
    std_pfov25 = np.std(episode_efficiencies_pfov25, axis=0)
    mean_pfov15 = np.mean(episode_efficiencies_pfov15, axis=0)
    std_pfov15 = np.std(episode_efficiencies_pfov15, axis=0) """
    mean_pfov10 = np.mean(episode_efficiencies_pfov10, axis=0)
    std_pfov10 = np.std(episode_efficiencies_pfov10, axis=0)
    """ mean_ig = np.mean(episode_efficiencies_ig, axis=0)
    std_ig = np.std(episode_efficiencies_ig, axis=0) """
    """ mean_Random = np.mean(episode_efficiencies_Random, axis=0)
    std_Random = np.std(episode_efficiencies_Random, axis=0) """
    mean_Maskppo = np.mean(episode_efficiencies_Maskppo, axis=0)
    std_Maskppo = np.std(episode_efficiencies_Maskppo, axis=0)
    mean_Ppo = np.mean(episode_efficiencies_Ppo, axis=0)
    std_Ppo = np.std(episode_efficiencies_Ppo, axis=0)
    mean_Dqn = np.mean(episode_efficiencies_Dqn, axis=0)
    std_Dqn = np.std(episode_efficiencies_Dqn, axis=0)

    # Time statistics
    mean_time_pfov = np.mean(evaluate_times_pfov10)
    std_time_pfov  = np.std(evaluate_times_pfov10)

    """ mean_time_ig   = np.mean(evaluate_times_ig)
    std_time_ig    = np.std(evaluate_times_ig) """

    """ mean_time_random   = np.mean(evaluate_times_random)
    std_time_random    = np.std(evaluate_times_random) """

    mean_time_Maskppo   = np.mean(evaluate_times_Maskppo)
    std_time_Maskppo    = np.std(evaluate_times_Maskppo)
    mean_time_Ppo   = np.mean(evaluate_times_Ppo)
    std_time_Ppo    = np.std(evaluate_times_Ppo)
    mean_time_Dqn   = np.mean(evaluate_times_Dqn)
    std_time_Dqn    = np.std(evaluate_times_Dqn)

    print(f"evaluate_agent_track (pFOV) — mean: {mean_time_pfov:.4f}s, std: {std_time_pfov:.4f}s")
    #print(f"evaluate_agent_track (IG)   — mean: {mean_time_ig:.4f}s,   std: {std_time_ig:.4f}s")
    #print(f"evaluate_agent_track (Random)   — mean: {mean_time_random:.4f}s,   std: {std_time_random:.4f}s")
    print(f"evaluate_agent_track (MaskPPO)   — mean: {mean_time_Maskppo:.4f}s,   std: {std_time_Maskppo:.4f}s")
    print(f"evaluate_agent_track (PPO)   — mean: {mean_time_Ppo:.4f}s,   std: {std_time_Ppo:.4f}s")
    print(f"evaluate_agent_track (DQN)   — mean: {mean_time_Dqn:.4f}s,   std: {std_time_Dqn:.4f}s")

    error_episodespFOV = np.array(error_episodespFOV)
    total_error_episodespFOV = np.array(total_error_episodespFOV)

    if len(error_episodespFOV)>0:

        mean_pos_error_all_episodes = sum(error_episodespFOV)/len(error_episodespFOV)
        """ print("Mean of positional errors over all episodes pFOV " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodespFOV], ddof=1)) """
    if len(total_error_episodespFOV)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodespFOV)/len(total_error_episodespFOV)
        print("Mean of covariance trace over all episodes pFOV " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodespFOV], ddof=1))

    """ error_episodesIG = np.array(error_episodesIG)
    total_error_episodesIG = np.array(total_error_episodesIG)
    if len(error_episodesIG)>0:

        mean_pos_error_all_episodes = sum(error_episodesIG)/len(error_episodesIG)
        print("Mean of positional errors over all episodes IG " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodesIG], ddof=1))
    if len(total_error_episodesIG)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodesIG)/len(total_error_episodesIG)
        print("Mean of covariance trace over all episodes IG " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodesIG], ddof=1)) """

    """ error_episodesRandom = np.array(error_episodesRandom)
    total_error_episodesRandom = np.array(total_error_episodesRandom)
    if len(error_episodesRandom)>0:

        mean_pos_error_all_episodes = sum(error_episodesRandom)/len(error_episodesRandom)
        print("Mean of positional errors over all episodes Random " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodesRandom], ddof=1))
    if len(total_error_episodesRandom)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodesRandom)/len(total_error_episodesRandom)
        print("Mean of covariance trace over all episodes Random " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodesRandom], ddof=1)) """

    error_episodesMaskppo = np.array(error_episodesMaskppo)
    total_error_episodesMaskppo = np.array(total_error_episodesMaskppo)
    if len(error_episodesMaskppo)>0:

        mean_pos_error_all_episodes = sum(error_episodesMaskppo)/len(error_episodesMaskppo)
        """ print("Mean of positional errors over all episodes MaskPPO " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodesMaskppo], ddof=1)) """
    if len(total_error_episodesMaskppo)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodesMaskppo)/len(total_error_episodesMaskppo)
        print("Mean of covariance trace over all episodes MaskPPO " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodesMaskppo], ddof=1))

    error_episodesPpo = np.array(error_episodesPpo)
    total_error_episodesPpo = np.array(total_error_episodesPpo)
    if len(error_episodesPpo)>0:

        mean_pos_error_all_episodes = sum(error_episodesPpo)/len(error_episodesPpo)
        """ print("Mean of positional errors over all episodes PPO " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodesPpo], ddof=1)) """
    if len(total_error_episodesPpo)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodesPpo)/len(total_error_episodesPpo)
        print("Mean of covariance trace over all episodes PPO " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodesPpo], ddof=1))

    error_episodesDqn = np.array(error_episodesDqn)
    total_error_episodesDqn = np.array(total_error_episodesDqn)
    if len(error_episodesDqn)>0:

        mean_pos_error_all_episodes = sum(error_episodesDqn)/len(error_episodesDqn)
        """ print("Mean of positional errors over all episodes DQN " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodesDqn], ddof=1)) """
    if len(total_error_episodesDqn)>0:
        mean_pos_total_error_all_episodes = sum(total_error_episodesDqn)/len(total_error_episodesDqn)
        print("Mean of covariance trace over all episodes DQN " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_error_episodesDqn], ddof=1))

    plt.figure()

    """ plt.plot(timesteps, mean_pfov4, label="Heuristic pFOV4")
    plt.fill_between(timesteps, mean_pfov4 - std_pfov4, mean_pfov4 + std_pfov4, alpha=0.3)

    plt.plot(timesteps, mean_pfov25, label="Heuristic pFOV25")
    plt.fill_between(timesteps, mean_pfov25 - std_pfov25, mean_pfov25 + std_pfov25, alpha=0.3)

    plt.plot(timesteps, mean_pfov15, label="Heuristic pFOV15")
    plt.fill_between(timesteps, mean_pfov15 - std_pfov15, mean_pfov15 + std_pfov15, alpha=0.3) """

    plt.plot(timesteps, mean_pfov10, label="Heuristic pFOV sqrt(2.0e-8)")
    plt.fill_between(timesteps, mean_pfov10 - std_pfov10, mean_pfov10 + std_pfov10, alpha=0.3)

    """ plt.plot(timesteps, mean_ig, label="Heuristic IG")
    plt.fill_between(timesteps, mean_ig - std_ig, mean_ig + std_ig, alpha=0.3) """

    """ plt.plot(timesteps, mean_Random, label="Random")
    plt.fill_between(timesteps, mean_Random - std_Random, mean_Random + std_Random, alpha=0.3) """

    """ plt.plot(timesteps, mean_Maskppo, label="MaskPPO")
    plt.fill_between(timesteps, mean_Maskppo - std_Maskppo, mean_Maskppo + std_Maskppo, alpha=0.3)

    plt.plot(timesteps, mean_Ppo, label="PPO")
    plt.fill_between(timesteps, mean_Ppo - std_Ppo, mean_Ppo + std_Ppo, alpha=0.3)

    plt.plot(timesteps, mean_Dqn, label="DQN")
    plt.fill_between(timesteps, mean_Dqn - std_Dqn, mean_Dqn + std_Dqn, alpha=0.3) """

    plt.xlabel("Time")
    plt.ylabel("Efficiency")
    plt.title("Average Efficiency over Episodes")
    plt.legend()
    plt.grid()

    plt.show()

def plot_mean_efficiencies(meanEfficiencies):
    """
    Plots mean efficiency curves for all algorithms stored in meanEfficiencies.

    Parameters
    ----------
    meanEfficiencies : dict
        Dictionary of form:
        {
            "AlgorithmName": np.array(shape=(T,)),
            ...
        }
    """

    plt.figure()

    for alg_name, values in meanEfficiencies.items():
        if values is None:
            continue

        values = np.asarray(values)
        x = np.arange(len(values))
        plt.plot(x, values, label=alg_name)

    plt.xlabel("Time Step")
    plt.ylabel("Mean Efficiency")
    plt.title("Mean Efficiency over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def computeEff(klCov, tracks, ls_cov_dict):
    efficiency_by_target = {}
    efficiency_by_targetKL = {}
    t_by_target = {}

    # ODE tolerances
    ode_tol = 1e-12
    for tgt_id, track in tracks.items():
        tvec = [obs["t"] for obs in track]
        #tvec = np.arange(100)
        #covEst = [obs["cov"] for obs in track]
        covEstKL = klCov[tgt_id]

        targets = ls_cov_dict[tgt_id]
        P0 = targets["P0_post"][:4, :4]
        if targets["model"] == "CT":
            integrationFcn = KalmanFilter.int_constant_turn_stm_2D
            inputs = {
                        "Rk": None,
                        "Q": np.eye(2) * 5e-13,
                        "Po": P0,
                        "omega": targets["omega"]
                    }
        else:
            integrationFcn = KalmanFilter.int_constant_velocity_stm
            inputs = {
                        "Rk": None,
                        "Q": np.eye(2) * 5e-27,
                        "Po": P0
                    }
        P = targets["P_mat"]
        efficiency = []
        efficiencyKL = []
        propPLS = []
        
        n = 4
        phi0_v = np.eye(n).reshape(n * n)
        int0 = np.hstack((targets["Xref"][:4,0], phi0_v))
        for k in range(len(tvec)):
            if tvec[k] == 0:
                P_prop = P0
                Xref = targets["Xref"][:,0].copy()
            else:
                sol = solve_ivp(
                    fun=lambda tau, y: integrationFcn(tau, y, inputs),
                    t_span=(0, tvec[k]),  # always from t0
                    y0=int0,                    # always from initial state + identity STM
                    rtol=ode_tol,
                    atol=ode_tol
                )
                xout = sol.y[:, -1]
                Xref = xout[:n]
                phi = xout[n:].reshape((n, n))

                P_prop = phi @ P0 @ phi.T

            #P_est = np.array(covEst[k])
            P_est_KL = np.array(covEstKL[:,:,k])
            #P_ls = np.array(P[:4,:4,k])
            P_ls = P_prop

            det_ls  = np.linalg.det(P_ls)
            #det_est = np.linalg.det(P_est)
            det_estKL = np.linalg.det(P_est_KL)

            # Avoid division by zero
            if det_ls <= 0:
                efficiency.append(np.nan)
                efficiencyKL.append(np.nan)
            else:
                #efficiency.append(det_ls / det_est)
                efficiencyKL.append(det_ls / det_estKL)
                propPLS.append(det_ls)

        #efficiency_by_target[tgt_id] = np.array(efficiency)
        efficiency_by_targetKL[tgt_id] = np.array(efficiencyKL)
        t_by_target[tgt_id] = np.array(tvec)
        """ propPLS = np.array(propPLS)
        P_xx = propPLS[:, 0, 0]
        P_yy = propPLS[:, 1, 1]
        three_sigma_pos = 3.0 * np.sqrt(P_xx + P_yy)
        detP = np.det """

        """ plt.plot(propPLS)
        plt.xlabel("Timestep")
        plt.ylabel("Determinant")
        plt.title("Covariance")
        plt.grid(True)
        plt.show() """
    return efficiency_by_targetKL, t_by_target

def plot_efficiency(efficiency_by_target, t_by_target):

    for tgt_id in efficiency_by_target.keys():

        plt.figure()

        plt.plot(
            t_by_target[tgt_id],
            efficiency_by_target[tgt_id]
        )

        plt.xlabel("Time")
        plt.ylabel("Efficiency (det P_LS / det P_est)")
        plt.title(f"Efficiency over Time - Target {tgt_id}")
        plt.grid(True)

        plt.show()

def plot_efficiency_all(efficiency_by_target, t_by_target):

    plt.figure()

    for tgt_id in efficiency_by_target.keys():

        plt.scatter(
            t_by_target[tgt_id],
            efficiency_by_target[tgt_id],
            label=f"Target {tgt_id}"
        )

    plt.xlabel("Time")
    plt.ylabel("Efficiency (det P_LS / det P_est)")
    plt.title("Efficiency over Time - All Targets")
    #plt.axhline(1)        # reference line
    plt.yscale("log") 
    plt.legend()
    plt.grid(True)

    plt.show()

def rmsePlot():
    n_targets = 5
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    obs = env.reset()
    n_episodes = 100

    sigma_theta = np.deg2rad(1.0 / 3600.0)  # 1 arcsec bearing noise
    sigma_r = 0.1        # 10 cm range noise

    R = np.diag([sigma_theta**2, sigma_r**2])
    Q = np.eye(2) * 1e-17

    heuristic = False
    random = False
    model = "maskableppo"
    print("Maskable PPO starts")
    errors_maskppo, total_errors_maskppo = computeRMSEalgo(heuristic, random, model, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = True
    random = False
    model = None
    print("Heuristic starts")
    errors_heuristic, total_errors_heuristic = computeRMSEalgo(heuristic, random, model, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "dqn"
    print("DQN starts")
    errors_dqn, total_errors_dqn = computeRMSEalgo(heuristic, random, model, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "ppo"
    print("PPO starts")
    errors_ppo, total_errors_ppo = computeRMSEalgo(heuristic, random, model, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = True
    model = None
    print("Random starts")
    errors_random, total_errors_random = computeRMSEalgo(heuristic, random, model, env, n_episodes, sigma_theta, sigma_r, R, Q)


    results = {
        "Random": errors_random,
        "PPO": errors_ppo,
        "Heuristic": errors_heuristic,
        "DQN" : errors_dqn,
        "Maskable PPO" : errors_maskppo
    }
    plot_violin(results, ylabel="Positional Error")

    results = {
        "Random": total_errors_random,
        "PPO": total_errors_ppo,
        "Heuristic": total_errors_heuristic,
        "DQN" : total_errors_dqn,
        "Maskable PPO": total_errors_maskppo
    }
    plot_violin(results, ylabel="Total Positional Error")
    


def kalmanPlots():
    n_targets = 5
    seed=42
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=seed, mode="track")
    obs = env.reset(seed=seed)
    n_episodes = 1

    sigma_theta = np.deg2rad(1.0 / 3600.0)  # 1 arcsec bearing noise
    sigma_r = 0.1        # 10 cm range noise

    R = np.diag([sigma_theta**2, sigma_r**2])
    Q = np.eye(2) * 1e-17

    # Generate truth data
    timesteps = np.arange(env.max_steps)
    all_target_states = {}
    all_meas = {}
    for tgt_id in env.targets:
        tid = tgt_id["id"]
        truth_states, measurements, _ = generate_truth_states(timesteps, tid, env)
        all_target_states[tid] = truth_states
        measurements[:, 0] += np.random.normal(0, sigma_theta, size=len(measurements))
        measurements[:, 1] += np.random.normal(0, sigma_r, size=len(measurements))
        all_meas[tid] = measurements

    det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=n_episodes, random_policy=False, deterministic_policy=True, seed=seed)
    tracks = extract_tracks_from_log(last_episode_log)
    estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q)

    dqn_model = DQN.load("agents/dqn_track_trained_IEEE", env=env)
    env = last_env
    dqn_rewards, exceedFOV_dqn, last_env, last_episode_log, illegal_actions_dqn = evaluate_agent_track(env, model=dqn_model, n_episodes=n_episodes)
    tracks = extract_tracks_from_log(last_episode_log)
    estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q)

    maskppo_model = MaskablePPO.load("agents/maskableppo_track_trained_IEEE", env=env)
    env = last_env

    
    #obs = env.reset()


def main():
    # ****** Test with random policy ******
    n_targets = 5
    seeds = [42, 123, 321]
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    n_episodes = 2

    #visualize_initial_positions(env)

    # Run random policy
    #positions, covariances = run_random_policy_search(env, n_steps=10)
    #positions, covariances = run_random_policy_track(env, n_steps=100)
    #positions, covariances = run_random_policy_combined(env, n_steps=10)

    """ plot_positions(positions, env) 

    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode="search")

    # Load trained model
    model = PPO.load("agents/ppo_search_trained", env=env)
    #model = DQN.load("agents/dqn_search_trained", env=env)

    # Visualize trained agent
    visualize_trained_agent(env, model, n_steps=30) """
    
#def anotherMethod():
    

    """ env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="search")
    n_episodes = 1

    # ****** Search ******
    # ****** Random policy ******
    random_rewards, random_detections = evaluate_agent_search(env, n_episodes=n_episodes, random_policy=True, seed=seeds)
    print("Reward")
    print(sum(random_rewards)/len(random_rewards))
    print(np.std([np.mean(arr) for arr in random_rewards], ddof=1))
    print("Detections")
    print(sum(random_detections)/len(random_detections))
    print(np.std([np.mean(arr) for arr in random_detections], ddof=1))

    # ****** PPO agent ******
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode="search")
    ppo_model = PPO.load("agents/ppo_search_trained_slowTargets_obsSpace4Channels", env=env)
    ppo_rewards, ppo_detections = evaluate_agent_search(env, model=ppo_model, n_episodes=n_episodes, seed=seeds)
    print("Reward")
    print(sum(ppo_rewards)/len(ppo_rewards))
    print(np.std([np.mean(arr) for arr in ppo_rewards], ddof=1))
    print("Detections")
    print(sum(ppo_detections)/len(ppo_detections))
    print(np.std([np.mean(arr) for arr in ppo_detections], ddof=1))

    # ****** DQN agent ******
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode="search")
    dqn_model = DQN.load("agents/dqn_search_trained_IEEE2", env=env)
    dqn_rewards, dqn_detections = evaluate_agent_search(env, model=dqn_model, n_episodes=n_episodes, seed=seeds)
    print("Reward")
    print(sum(dqn_rewards)/len(dqn_rewards))
    print(np.std([np.mean(arr) for arr in dqn_rewards], ddof=1))
    print("Detections")
    print(sum(dqn_detections)/len(dqn_detections))
    print(np.std([np.mean(arr) for arr in dqn_detections], ddof=1))

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": random_rewards,
        "PPO": ppo_rewards,
        "DQN": dqn_rewards
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        "Random": random_detections,
        "PPO": ppo_detections,
        "DQN": dqn_detections
    }
    plot_violin(detection_results, ylabel="Number of Detections") """

    # ****** Track ******
    """ obs = env.reset()
    maskppo_model = MaskablePPO.load("agents/maskableppo_track_trained_IEEE", env=env)
    maskppo_rewards, exceedFOV_maskppo, last_env, last_episode_log, illegal_actions_maskppo = evaluate_agent_track(env, model=maskppo_model, n_episodes=n_episodes, deterministic_policy=False, maskable=True)
    print("Illegal action")
    print(sum(illegal_actions_maskppo)/len(illegal_actions_maskppo))
    print(np.std([np.mean(arr) for arr in illegal_actions_maskppo], ddof=1))
    print("Lost targets")
    print(sum(exceedFOV_maskppo)/len(exceedFOV_maskppo))
    print(np.std([np.mean(arr) for arr in exceedFOV_maskppo], ddof=1))
    print("Reward")
    print(sum(maskppo_rewards)/len(maskppo_rewards))
    print(np.std([np.mean(arr) for arr in maskppo_rewards], ddof=1)) """

    # ****** Deterministic policy pFOV ******
    det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=n_episodes, random_policy=False, deterministic_policy=True, deterministic_policy_alternative=False)
    print("Illegal action")
    print(sum(illegal_actions_det)/len(illegal_actions_det))
    print(np.std([np.mean(arr) for arr in illegal_actions_det], ddof=1))
    print("Lost targets")
    print(sum(exceedFOV_det)/len(exceedFOV_det))
    print(np.std([np.mean(arr) for arr in exceedFOV_det], ddof=1))
    print("Reward")
    print(sum(det_rewards)/len(det_rewards))
    print(np.std([np.mean(arr) for arr in det_rewards], ddof=1))

    # ****** Deterministic policy IG******
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    det_rewards, exceedFOV_det, last_env, last_episode_log, illegal_actions_det = evaluate_agent_track(env, n_episodes=n_episodes, random_policy=False, deterministic_policy=False, deterministic_policy_alternative=True)
    print("Illegal action")
    print(sum(illegal_actions_det)/len(illegal_actions_det))
    print(np.std([np.mean(arr) for arr in illegal_actions_det], ddof=1))
    print("Lost targets")
    print(sum(exceedFOV_det)/len(exceedFOV_det))
    print(np.std([np.mean(arr) for arr in exceedFOV_det], ddof=1))
    print("Reward")
    print(sum(det_rewards)/len(det_rewards))
    print(np.std([np.mean(arr) for arr in det_rewards], ddof=1))

    # ****** Random policy ******
    """ env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    # Reset environment ONCE and plot initial positions right after
    obs = env.reset()
    random_rewards, exceedFOV_random, last_env, last_episode_log, illegal_actions_random = evaluate_agent_track(env, n_episodes=n_episodes, random_policy=True, deterministic_policy=False)
    print("Illegal action")
    print(sum(illegal_actions_random)/len(illegal_actions_random))
    print(np.std([np.mean(arr) for arr in illegal_actions_random], ddof=1))
    print("Lost targets")
    print(sum(exceedFOV_random)/len(exceedFOV_random))
    print(np.std([np.mean(arr) for arr in exceedFOV_random], ddof=1))
    print("Reward")
    print(sum(random_rewards)/len(random_rewards))
    print(np.std([np.mean(arr) for arr in random_rewards], ddof=1)) """
    """ tracks = extract_tracks_from_log(last_episode_log)
    estimates = estimate_all_targets_from_tracks(tracks, last_env)

    truth_by_tgt, est_by_tgt, cov_by_tgt, errors_by_tgt, sigma_by_tgt, t_by_tgt = \
        process_estimates(estimates, last_env)
    
    for i in range(last_env.n_targets):
        plot_errors_and_sigmas(
            errors_by_tgt[i],
            sigma_by_tgt[i],
            t=t_by_tgt[i],          
            target_id=i
        ) """
 
    """ # ****** PPO agent ******
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    # Reset environment ONCE and plot initial positions right after
    obs = env.reset()
    ppo_model = PPO.load("agents/ppo_track_trained_IEEE", env=env)

    ppo_rewards, exceedFOV_ppo, last_env, last_episode_log, illegal_actions_ppo = evaluate_agent_track(env, model=ppo_model, n_episodes=n_episodes, deterministic_policy=False)
    print("Illegal action")
    print(sum(illegal_actions_ppo)/len(illegal_actions_ppo))
    print(np.std([np.mean(arr) for arr in illegal_actions_ppo], ddof=1))
    print("Lost targets")
    print(sum(exceedFOV_ppo)/len(exceedFOV_ppo))
    print(np.std([np.mean(arr) for arr in exceedFOV_ppo], ddof=1))
    print("Reward")
    print(sum(ppo_rewards)/len(ppo_rewards))
    print(np.std([np.mean(arr) for arr in ppo_rewards], ddof=1)) """
    """ visualize_initial_positions(last_env)
    print(last_env.motion_model) """
    """ tracks = extract_tracks_from_log(last_episode_log)
    estimates = estimate_all_targets_from_tracks(tracks, last_env)

    truth_by_tgt, est_by_tgt, cov_by_tgt, errors_by_tgt, sigma_by_tgt, t_by_tgt = \
        process_estimates(tracks, estimates, last_env)
    
    for i in range(last_env.n_targets):
        plot_errors_and_sigmas(
            errors_by_tgt[i],
            sigma_by_tgt[i],
            t=t_by_tgt[i],          
            target_id=i
        ) """
 

    """ # ****** DQN agent ******
    env = MultiTargetEnv(n_targets=n_targets, n_unknown_targets=100, seed=None, mode="track")
    obs = env.reset()
    dqn_model = DQN.load("agents/dqn_track_trained_IEEE", env=env)

    dqn_rewards, exceedFOV_dqn, last_env, last_episode_log, illegal_actions_dqn = evaluate_agent_track(env, model=dqn_model, n_episodes=n_episodes)
    print("Illegal action")
    print(sum(illegal_actions_dqn)/len(illegal_actions_dqn))
    print(np.std(illegal_actions_dqn, ddof=1))
    print(illegal_actions_dqn)
    print("Lost targets")
    print(sum(exceedFOV_dqn)/len(exceedFOV_dqn))
    print(np.std([np.mean(arr) for arr in exceedFOV_dqn], ddof=1))
    print("Reward")
    print(sum(dqn_rewards)/len(dqn_rewards))
    print(np.std([np.mean(arr) for arr in dqn_rewards], ddof=1))
    visualize_initial_positions(last_env)
    print(last_env.motion_model) 
    tracks = extract_tracks_from_log(last_episode_log) """
    """ estimates = estimate_all_targets_from_tracks(tracks, last_env)

    truth_by_tgt, est_by_tgt, cov_by_tgt, errors_by_tgt, sigma_by_tgt, t_by_tgt = \
        process_estimates(tracks, estimates, last_env) """
    
    """ for i in range(last_env.n_targets):
        plot_errors_and_sigmas(
            errors_by_tgt[i],
            sigma_by_tgt[i],
            t=t_by_tgt[i],          
            target_id=i
        ) """
 

    """ # ****** Plot reward distributions ******
    reward_results = {
        "Random": random_rewards,
        "PPO": ppo_rewards,
        "Heuristic": det_rewards,
        "DQN" : dqn_rewards
    }
    plot_violin(reward_results, ylabel="Episode Reward")
    #exceedFOV_dqn = [5]
    plot_means_lost_targets(exceedFOV_ppo, n_targets, exceedFOV_dqn, n_targets, exceedFOV_random, n_targets, exceedFOV_det, n_targets)
 """
 
if __name__ == "__main__":
    #main()
    #kalmanPlots()
    #rmsePlot()
    efficiencyPlot()
    #efficiencyOtherPlot()