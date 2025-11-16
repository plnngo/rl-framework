import copy
from stable_baselines3 import DQN, PPO
from LSBatchFilter import estimate_all_targets_from_tracks
from multi_target_env import MultiTargetEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from matplotlib.patches import Ellipse, Rectangle

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

    # Known targets (blue)
    for tgt in env.targets:
        ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", label="Known Target")
        plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="blue", alpha=0.3)

    # Unknown targets (orange)
    for utgt in env.unknown_targets:
        ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target")
        plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="orange", alpha=0.3)

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

    for step in range(n_steps):
        valid_ids = np.flatnonzero(env.known_mask)
        if len(valid_ids) == 0:
            print("No known targets available for tracking at step", step)
            break

        action = int(env.rng.choice(valid_ids))
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1:02d}: TRACK target {action}, Reward={reward:.4f}")

        """ fig, ax = plt.subplots(figsize=(8, 8))
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
        ax.legend(unique.values(), unique.keys(), loc="upper right") """

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    # --- Show all figures together at the end ---
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

def extract_target_state_cov(obs, target_idx, env):
    """
    Extract both the d-dimensional state vector and its corresponding dxd covariance
    matrix for a given target index from the flat observation vector `obs`.

    Parameters
    ----------
    obs : np.ndarray
        Flat observation vector returned by the environment.
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
    per = env.obs_dim_per_target
    d = env.d_state
    start = int(target_idx * per)

    # Slice corresponding to this target
    target_slice = obs[start:start + per]

    # Extract state vector (first d_state elements)
    x = target_slice[:d].astype(float)

    # Extract packed Cholesky (next cholesky_size elements)
    ch_pack = target_slice[d:d + env.cholesky_size].astype(float)

    # Handle degenerate or uninitialized covariance
    if np.allclose(ch_pack, 0.0) or np.any(np.isnan(ch_pack)):
        P = env.P0.copy()
    else:
        L = _unpack_cholesky(ch_pack, d)
        P = L @ L.T
        P = 0.5 * (P + P.T)  # ensure symmetry

    return x, P

def analyse_tracking_task(obs, target_idx, env, confidence=0.95):
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
    mask_offset = int(env.max_targets * env.obs_dim_per_target)
    mask_val = obs[mask_offset + int(target_idx)]
    if mask_val <= 0.5:
        return False

    x, P = extract_target_state_cov(obs, target_idx, env)
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

def evaluate_agent_track(env, model=None, n_episodes=100, random_policy=False):
    rewards = []
    exceedFOV = []

    # for logging the final episode
    last_episode_log = {}
    last_env = None

    for ep in trange(n_episodes, desc="Evaluating"):
        obs, _ = env.reset()
        exceed_count = 0
        exceed_target = []

        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_env = copy.deepcopy(env)
            episode_log = {}        # create a temporary log if this is the last episode
        done = False
        total_reward = 0.0
        t = 0  # timestep counter

        while not done:
            # --- Choose action ---
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            # --- Step environment ---
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # --- Analyse targets ---
            if ep == n_episodes - 1:
                episode_log[t] = {}

            for tgt in range(env.n_targets):
                exceed, x, P = analyse_tracking_task(obs, tgt, env, confidence=0.95)
                if exceed and tgt not in exceed_target:
                    exceed_target.append(tgt)

                # if this is the last episode, store everything
                if ep == n_episodes - 1 and tgt == info["target_id"]:
                    episode_log[t][tgt] = {
                        "id": tgt,
                        "state": x.copy(),
                        "cov": P.copy(),
                        "exceedFOV": bool(exceed),
                    }

            t += 1  # increment timestep

        rewards.append(total_reward)
        exceedFOV.append(exceed_target)
        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_episode_log = episode_log

    return rewards, exceedFOV, last_env, last_episode_log

def evaluate_agent_search(env, model=None, n_episodes=100, random_policy=False):
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
    detections = []
    detection_count = 0
    detect_count3 = 0

    for ep in trange(n_episodes, desc="Evaluating"):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        detect_count2 = 0

        # Initialize previous mask to track detection changes
        prev_mask = env.get_action_mask().copy()

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Compare action mask with previous one to detect new trackable targets
            if reward>9:
                detect_count3 = detect_count3 + 1
            curr_mask = info["action_mask"]
            # Count newly enabled tracking actions (assuming they correspond to detections)
            new_detections = np.sum((curr_mask == 1) & (prev_mask == 0))
            detection_count += int(new_detections)
            prev_mask = curr_mask.copy()

        rewards.append(total_reward)
        detections.append(detect_count2)

    return rewards, detections, detect_count3


def plot_violin(results_dict, ylabel="Episode Reward"):
    """
    Plots a violin plot comparing metrics (e.g., rewards or detections) across agents.
    """
    colors = {
        "PPO": "orange",
        "DQN": "green",
        "Random": "red"
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

def plot_means_lost_targets(ppo, dqn, random):
    """
    Each input (ppo, dqn, random) is a list/array of subarrays.
    For each method:
        - Measure the length of each subarray
        - Compute the mean length
        - Compute SEM
    Then plot the results.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # Convert to numpy arrays of lengths
    ppo_lengths = np.array([len(arr) for arr in ppo])
    dqn_lengths = np.array([len(arr) for arr in dqn])
    random_lengths = np.array([len(arr) for arr in random])

    # Compute means
    means = np.array([
        ppo_lengths.mean(),
        dqn_lengths.mean(),
        random_lengths.mean()
    ])

    # Compute SEMs
    sem = np.array([
        ppo_lengths.std(ddof=1) / np.sqrt(len(ppo_lengths)),
        dqn_lengths.std(ddof=1) / np.sqrt(len(dqn_lengths)),
        random_lengths.std(ddof=1) / np.sqrt(len(random_lengths))
    ])

    # Plot
    labels = ["PPO", "DQN", "Random"]
    x = np.arange(len(labels))

    plt.figure(figsize=(6, 4))
    plt.bar(x, means, yerr=sem, capsize=8)
    plt.xticks(x, labels)
    plt.ylabel("Mean Subarray Length")
    plt.title("Mean Length of Subarrays with SEM Uncertainty")

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
                R
            )

            # Difference: updated EKF - predicted
            diffs[t][tgt_id] = x_upd - x_pred

            # If we have a logged state, compare as well
            if t in last_episode_log and tgt_id in last_episode_log[t]:
                x_log = last_episode_log[t][tgt_id]["state"]
                compared[t][tgt_id] = x_upd - x_log

    return diffs, compared

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

if __name__ == "__main__":
    """ # ****** Test with random policy ******
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=5, seed=42, mode="track")

    # Reset environment ONCE and plot initial positions right after
    obs = env.reset()
    visualize_initial_positions(env)

    # Run random policy
    #positions, covariances = run_random_policy_search(env, n_steps=10)
    positions, covariances = run_random_policy_track(env, n_steps=300)
    #positions, covariances = run_random_policy_combined(env, n_steps=10)

    plot_positions(positions, env) """

    """ env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode="search")

    # Load trained model
    model = PPO.load("agents/ppo_search_trained", env=env)
    #model = DQN.load("agents/dqn_search_trained", env=env)

    # Visualize trained agent
    visualize_trained_agent(env, model, n_steps=30) """
    
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=None, mode="track")
    n_episodes = 2

    # ****** Search ******
    """ # ****** Random policy ******
    random_rewards, random_detections, detect_count2random = evaluate_agent_search(env, n_episodes=n_episodes, random_policy=True)
    print(detect_count2random)

    # ****** PPO agent ******
    ppo_model = PPO.load("agents/ppo_search_trained", env=env)
    ppo_rewards, ppo_detections, detect_count2ppo = evaluate_agent_search(env, model=ppo_model, n_episodes=n_episodes)
    print(detect_count2ppo)

    # ****** DQN agent ******
    dqn_model = DQN.load("agents/dqn_search_trained", env=env)
    dqn_rewards, dqn_detections, detect_count2dqn = evaluate_agent_search(env, model=dqn_model, n_episodes=n_episodes)
    print(detect_count2dqn)

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
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot total detection ******
    detection_results = {
        "Random": detect_count2random,
        "PPO": detect_count2ppo,
        "DQN": detect_count2dqn
    }
    plot_detection_bar_chart(detection_results) """

    # ****** Track ******
    # ****** Random policy ******
    random_rewards, exceedFOV_random, last_env, last_episode_log = evaluate_agent_track(env, n_episodes=n_episodes, random_policy=True)
    print(exceedFOV_random)

    """ # ****** PPO agent ******
    ppo_model = PPO.load("agents/ppo_track_trained", env=env)
    ppo_rewards, exceedFOV_ppo, last_env, last_episode_log = evaluate_agent_track(env, model=ppo_model, n_episodes=n_episodes)
    print(exceedFOV_ppo)

    # ****** DQN agent ******
    dqn_model = DQN.load("agents/dqn_track_trained", env=env)
    dqn_rewards, exceedFOV_dqn, last_env, last_episode_log = evaluate_agent_track(env, model=dqn_model, n_episodes=n_episodes)
    print(exceedFOV_dqn)

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": random_rewards,
        "PPO": ppo_rewards,
        "DQN": dqn_rewards
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    plot_means_lost_targets(exceedFOV_ppo, exceedFOV_dqn, exceedFOV_random) """

    #visualize_initial_positions(last_env)
    print(last_env.motion_model)
    tracks = extract_tracks_from_log(last_episode_log)
    estimates = estimate_all_targets_from_tracks(tracks)

    #Print real motion models
 

       

    # ****** Generate truth data ******
    #truthData = propagate_known_targets_over_episode(last_env)

    # Compute state differences
    #diffs = compute_state_differences_with_ekf(truthData, last_episode_log, last_env, )
    

