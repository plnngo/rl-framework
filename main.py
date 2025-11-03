from stable_baselines3 import DQN, PPO
from multi_target_env import MultiTargetEnv
import numpy as np
import matplotlib.pyplot as plt
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

    # --- Draw reward window ---
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
    ax.add_patch(reward_rect)

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

    for step in range(n_steps):
       # Sample a valid action from the discrete space
        # Only track valid known targets
        valid_ids = np.flatnonzero(env.known_mask)
        if len(valid_ids) == 0:
            print("No known targets available for tracking at step", step)
            break

        # Choose a random valid target
        action = int(env.rng.choice(valid_ids))
        obs, reward, done, truncated, info = env.step(action)

        print(f"Step {step+1:02d}: TRACK target {action}, Reward={reward:.4f}")

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw all grid cells for context
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

        # Plot known (blue) + unknown (orange) markers
        for tgt in env.targets:
            if tgt["id"] == action:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="red", s=120, marker="*", label="Tracked Target")
            else:
                ax.scatter(tgt["x"][0], tgt["x"][1], c="blue", s=40, marker="o", label="Known Target" if step == 0 else "")
        for utgt in env.unknown_targets:
            ax.scatter(utgt["x"][0], utgt["x"][1], c="orange", label="Unknown Target" if step == 0 else "")

        # Draw uncertainty ellipses 
        for tgt in env.targets:
            if tgt["id"] == action:
                plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="red", alpha=0.7)
            else:
                plot_cov_ellipse(tgt["P"][:2, :2], tgt["x"][:2], ax, edgecolor="blue", alpha=0.3)
        for utgt in env.unknown_targets:
            plot_cov_ellipse(utgt["P"][:2, :2], utgt["x"][:2], ax, edgecolor="red", alpha=0.3)

        ax.set_xlim(-env.space_size / 2, env.space_size / 2)
        ax.set_ylim(-env.space_size / 2, env.space_size / 2)
        ax.set_aspect("equal", "box")
        ax.set_title(f"Step {step+1}: TRACK target {action}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)

        # Combine legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right")

        plt.pause(1)
        plt.close(fig)  # Show each step's figure

        positions.append([tgt['x'][:2] for tgt in env.targets + env.unknown_targets])
        covariances.append([tgt['P'][:2, :2] for tgt in env.targets + env.unknown_targets])

        if done or truncated:
            break

    plt.show()
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
        
        # --- Draw reward window ---
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
        ax.add_patch(reward_rect)

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
        plt.pause(1)
        plt.close(fig)

        if done:
            obs, _ = env.reset()


if __name__ == "__main__":
    # ****** Test with random policy ******
    """ env = MultiTargetEnv(n_targets=5, n_unknown_targets=3, seed=42, mode="track")

    # Reset environment ONCE and plot initial positions right after
    obs = env.reset()
    visualize_initial_positions(env)

    # Run random policy
    #positions, covariances = run_random_policy_search(env, n_steps=10)
    #positions, covariances = run_random_policy_track(env, n_steps=55)
    #positions, covariances = run_random_policy_combined(env, n_steps=10) """

    env = MultiTargetEnv(n_targets=5, n_unknown_targets=15, seed=42, mode="search")

    # Load trained model
    model = PPO.load("ppo2_sensor_tasking_search_gamma09_steps50000.zip", env=env)
    #model = DQN.load("dqn_sensor_tasking_search_gamma09820_steps50000.zip", env=env)

    # Visualize trained agent
    visualize_trained_agent(env, model, n_steps=50)
