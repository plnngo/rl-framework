import copy
import time
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import DQN, PPO
from tqdm import trange
import KalmanFilter
from LSBatchFilter import Fx_ct, Fx_ct_linear, Fx_cv_cont, f_ct, f_ct_linear, f_cv_cont, generate_truth_states, propagate_state_and_stm
import MCTSenv
from MacroEnv import MacroEnv, _sync_envs
from deterministic_macro import select_best_action_pFOV
from main import analyse_tracking_task, estimateAndPlot, extract_target_state_cov, extract_tracks_from_log, plot_detection_bar_chart, plot_means_lost_targets, plot_violin
from multi_seed_training_macro import MacroRandomSeedEnv
from multi_target_env import MultiTargetEnv

def computeRMSEalgo(seeds=None, heuristic=False, random=False, model=None, trackModel=None, env=None, n_episodes=100, sigma_theta=0, sigma_r=0, R=None, Q=None):
    error_episodes = []
    total_traceCov_episodes = []
    for i in range(n_episodes):
        if heuristic or random:
            det_rewards, detection_count, exceedFOV_det, last_env, last_episode_log, known, last_tasks_log, det_illegal = evaluate_agent_macro(seeds=seeds, env=env, model=None, n_episodes=1, random_policy=random, deterministic_policy=heuristic)
        elif model=="ppo":
            if trackModel == "dqn":
                ppo_model = PPO.load("agents/ppo_macro_trained_dqn_track", env=env)
            else:
                ppo_model = PPO.load("agents/ppo_macro_trained", env=env)

            ppo_rewards, detection_count, exceedFOV_ppo, last_env, last_episode_log, known, last_tasks_log, ppo_illegal = evaluate_agent_macro(seeds=seeds, env=env, model=ppo_model, n_episodes=1)

        else:
            if trackModel == "dqn":
               dqn_model = DQN.load("agents/dqn_macro_trained_dqn_track", env=env)
            else:
               dqn_model = DQN.load("agents/dqn_macro_trained", env=env)

            dqn_rewards, detection_count, exceedFOV_dqn, last_env, last_episode_log, known, last_tasks_log, dqn_illegal = evaluate_agent_macro(seeds=seeds, env=env, model=dqn_model, n_episodes=1)
        #max_id = max((t["id"] for t in env.targets), default=None)
        tracks = extract_tracks_from_log(last_episode_log, env.max_targets)
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
        for tgt_id in last_env.unknown_targets:
            tid = tgt_id["id"]
            truth_states, measurements, _ = generate_truth_states(timesteps, tid, last_env)
            all_target_states[tid] = truth_states
            measurements[:, 0] += np.random.normal(0, sigma_theta, size=len(measurements))
            measurements[:, 1] += np.random.normal(0, sigma_r, size=len(measurements))
            all_meas[tid] = measurements

        errors_all_targets, total_trace_cov = estimateAndPlot(tracks, all_target_states, last_env, all_meas, R, Q)

        episode_error = 0.0

        for tgt_error in errors_all_targets:
            # Skip empty target errors
            if tgt_error.size == 0:
                continue

            # tgt_error shape: (state_dim, T_i)
            pos_error = tgt_error[:2, :]            # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)   # (T_i,)
            rmse_target = np.sqrt(np.mean(sq_err))

            episode_error += rmse_target

        if len(errors_all_targets)>0:
            rmse_all_target = episode_error/len(errors_all_targets)
            error_episodes.append(rmse_all_target)

        total_episode_trace_cov = sum(np.sum(arr) for arr in total_trace_cov)
        """ total_episode_error = 0
        for tgt_error in total_error_all_targets:
            pos_error = tgt_error[:2, :]                  # (2, T_i)
            sq_err = np.sum(pos_error**2, axis=0)          # (T_i,)
            total_rmse_target = np.sqrt(np.mean(sq_err))
            total_episode_error = total_episode_error + total_rmse_target """
        if len(total_trace_cov)>0:
            total_trace_all_target = total_episode_trace_cov/len(total_trace_cov)
            total_traceCov_episodes.append(total_trace_all_target)
        env.reset()

    error_episodes = np.array(error_episodes)
    total_traceCov_episodes = np.array(total_traceCov_episodes)

    if len(error_episodes)>0:
        mean_pos_error_all_episodes = sum(error_episodes)/len(error_episodes)
        print("Mean of positional errors over all episodes " + str(mean_pos_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in error_episodes], ddof=1))
    if len(total_traceCov_episodes)>0:
        mean_pos_total_error_all_episodes = sum(total_traceCov_episodes)/len(total_traceCov_episodes)
        print("Mean of covariance trace over all episodes " + str(mean_pos_total_error_all_episodes) + " +- ")
        print(np.std([np.mean(arr) for arr in total_traceCov_episodes], ddof=1))
    return error_episodes, total_traceCov_episodes

def rmsePlot():

    seeds = [42, 123, 321]
    n_episodes = 100
    n_targets = 5
    n_unknown_targets =100

    #DQN tracker
    trackModel="dqn"
    env = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker=trackModel)
    obs = env.reset()

    sigma_theta = np.deg2rad(1.0 / 3600.0)  # 1 arcsec bearing noise
    sigma_r = 0.1        # 10 cm range noise

    R = np.diag([sigma_theta**2, sigma_r**2])
    Q = np.eye(2) * 1e-17

    heuristic = True
    random = False
    model = None
    print("Heuristic starts")
    errors_heuristic, total_errors_heuristic = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "dqn"
    print("DQN starts")
    errors_dqn, total_errors_dqn = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "ppo"
    print("PPO starts")
    errors_ppo, total_errors_ppo = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = True
    model = None
    print("Random starts")
    errors_random, total_errors_random = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)

    #Heuristic tracker
    trackModel = None
    env = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=None)
    obs = env.reset()

    heuristic = True
    random = False
    model = None
    print("Heuristic starts")
    errors_heuristic_heuristicTrack, total_errors_heuristic_heuristicTrack = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "dqn"
    print("DQN starts")
    errors_dqn_heuristicTrack, total_errors_dqn_heuristicTrack = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = False
    model = "ppo"
    print("PPO starts")
    errors_ppo_heuristicTrack, total_errors_ppo_heuristicTrack = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)
    heuristic = False
    random = True
    model = None
    print("Random starts")
    errors_random_heuristicTrack, total_errors_random_heuristicTrack = computeRMSEalgo(seeds, heuristic, random, model, trackModel, env, n_episodes, sigma_theta, sigma_r, R, Q)


    results = {
        "Random": errors_random,
        "PPO": errors_ppo,
        "Heuristic": errors_heuristic,
        "DQN" : errors_dqn
    }
    plot_violin(results, ylabel="Positional Error")

    results = {
        "Random": total_errors_random,
        "PPO": total_errors_ppo,
        "Heuristic": total_errors_heuristic,
        "DQN" : total_errors_dqn
    }
    plot_violin(results, ylabel="Total Covariance Trace")

    results = {
        "Random": errors_random_heuristicTrack,
        "PPO": errors_ppo_heuristicTrack,
        "Heuristic": errors_heuristic_heuristicTrack,
        "DQN" : errors_dqn_heuristicTrack
    }
    plot_violin(results, ylabel="Positional Error")

    results = {
        "Random": total_errors_random_heuristicTrack,
        "PPO": total_errors_ppo_heuristicTrack,
        "Heuristic": total_errors_heuristic_heuristicTrack,
        "DQN" : total_errors_dqn_heuristicTrack
    }
    plot_violin(results, ylabel="Total Covariance Trace")

def evaluate_agent_macro(seeds=None, env=None, model=None, n_episodes=100, random_policy=False, deterministic_policy=False):
    rewards = []
    detect_count3 = 0
    detection_count = []
    exceedFOV = []
    illegal = []
    all_init_cond_targets = []

    sigma_r = 0.001        # 1 mm range noise
    sigma = sigma_r

    R = np.diag([sigma**2, sigma**2])
    obsFunc = MultiTargetEnv.extract_measurement_XY

    # for logging the final episode
    last_episode_log = []
    last_tasks_log = []

    # average number of known targets
    known = []

    for ep in trange(n_episodes, desc="Evaluating",  leave=False):
        seed = int(np.random.choice(seeds))
        obs, _ = env.reset(seed=seed)
        episode_log = {}        # create a temporary log if this is the last episode
        """ if ep == n_episodes - 1:
            episode_task_log = {} """
        episode_task_log = {}
        
        done = False
        total_reward = 0.0
        detections = env.init_n_targets
        t = 0  # timestep counter
        illegalActs = 0
        init_cond_targets = copy.deepcopy(env.targets)
        init_cond_unknownTargets = copy.deepcopy(env.unknown_targets)
        all_init_cond_targets.append(init_cond_targets + init_cond_unknownTargets)

        while not done:

            if isinstance(model, MCTSenv.MCTS):
                obs, reward, done, truncated, info, action, micro = run_mcts_step(env, model, obs)
            elif deterministic_policy:
                action = 0
                #action = select_best_action_pFOV(env, dt=1.0, boundary=env.boundary) #boundary=np.sqrt(1.0e-6)
                obs, reward, done, truncated, info = env.step(action)

            else:
                if random_policy:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)

            total_reward += reward

            # evaluate searching task
            """ known_targets = sum(info["action_mask"]["micro_track"])
            if known_targets>detections:
                detect_count3 = detect_count3 + (known_targets - detections)
                detections = known_targets """

            # evaluate tracking task
            episode_log[t] = {}
            """ if ep == n_episodes - 1:
                episode_task_log[t] = {"timestamp": t, "action": int(action)} """
            episode_task_log[t] = {"timestamp": t, "action": int(action)}
            
            
            for tgt in env.targets:
                #print(env.base_env.targets[tgt]["id"])
                if info.get("invalid_action") or info.get("catastrophic_loss") or not info["target_id"]:
                    if info.get("invalid_action"):
                        illegalActs += 1
                    continue
                #exceed, x, P = analyse_tracking_task(tgt["id"], env, confidence=0.95)
                x, P = extract_target_state_cov(tgt["id"], env)

                # if this is the last episode, store everything
                if tgt["id"] in info["target_id"]:
                    episode_log[t][tgt["id"]] = {
                        "id": tgt["id"],
                        "state": x.copy(),
                        "cov": P.copy()
                    }

            t += 1  # increment timestep
        rewards.append(total_reward)
        detection_count.append(env.detect_counter)
        exceedFOV.append(env.lost_counter)
        known.append(env.n_targets)
        illegal.append(illegalActs)
        last_episode_log.append(episode_log)

        # --- For last episode, store deep copy of env ---
        #if ep == n_episodes - 1:
        last_tasks_log.append(episode_task_log)

    # Perform estimation algorithm
    allTraceOverAllEpisodes = []
    for episode_log, init_cond in zip(last_episode_log, all_init_cond_targets):
        tracks = extract_tracks_from_log(episode_log)
        allTraces = 0

        for target_id, track in tracks.items():
            init_entry = next(
                (tgt for tgt in init_cond if tgt["id"] == target_id),
                None
            )

            if init_entry is None:
                print(f"Warning: target_id {target_id} not found in initial conditions, skipping.")
                continue

            x0 = init_entry["x"]
            P0 = init_entry["P"]

            propagated = propagate_to_track_timesteps(track, x0, P0, target_id, env, obsFunc)

            # t_obs: (L,) array of timesteps
            timesteps = np.array([entry["t"] for entry in propagated])

            # obs: (L, p) array of measurements  <-- ekf indexes as obs[k, :]
            tgt_meas = np.array([entry["measurement"] for entry in propagated])

            # Xo_ref: initial reference state
            Xo_ref = x0.copy()

            # Motion model dependent dynamics functions
            if env.motion_model[target_id] == "T":
                f_dyn  = f_ct_linear
                Fx_dyn = Fx_ct_linear
            else:
                f_dyn  = f_cv_cont
                Fx_dyn = Fx_cv_cont

            inputs = {
                "Rk":    R,                          # measurement noise covariance (p, p)
                "Q":     np.eye(2) * 0,                           # process noise covariance
                "Po":    P0,                          # initial state covariance (n, n)
                "f_dyn": f_dyn,
                "Fx_dyn": Fx_dyn,
            }

            # Optionally add omega for turning targets
            if env.motion_model[target_id] == "T":
                inputs["omega"] = env.motion_params[target_id]

            Xk, Pk, resids = KalmanFilter.ekf(
                Xo_ref = Xo_ref,
                t_obs  = timesteps,
                obs    = tgt_meas,
                intfcn = None,       # unused, ekf uses f_dyn/Fx_dyn from inputs directly
                H_fcn  = obsFunc,
                inputs = inputs
            )

            # --- Propagate last EKF estimate to t=100 ---

            # Extract last estimate and its timestep
            t_last = timesteps[-1]
            x_last = Xk[:, -1].copy()      # shape (n,)
            P_last = Pk[:, :, -1].copy()   # shape (n, n)

            n = x_last.shape[0]
            Phi0 = np.eye(n).reshape(-1)

            # Only propagate if last timestep is before t=100
            if t_last < 100:
                x_prop, Phi = propagate_state_and_stm(
                    t0=t_last,
                    t1=100,
                    x0=x_last,
                    Phi0_vec=Phi0,
                    f_dyn=f_dyn,
                    Fx_dyn=Fx_dyn,
                    n=n,
                    omega=inputs.get("omega", 0)
                )
                Phi = Phi.reshape(n, n)
                P_prop = Phi @ P_last @ Phi.T
            else:
                # Last update is already at t=100, no propagation needed
                x_prop = x_last
                P_prop = P_last

            # Extract trace from propagated covariance
            traceCov = np.trace(P_prop)
            allTraces += traceCov

        allTraceOverAllEpisodes.append(allTraces)

    print(allTraceOverAllEpisodes)
    print(rewards)
    print(known)
    print(exceedFOV)
    return rewards, detection_count, exceedFOV, allTraceOverAllEpisodes, last_episode_log, known, last_tasks_log, illegal


def propagate_to_track_timesteps(track, x0, P0, target_id, env, H_fcn):
    """
    Propagate x0 and P0 to the timesteps saved in track, and extract measurements.

    Args:
        track:      list of dicts with {"t": timestep, "state": ..., "cov": ...}
        x0:         initial state vector
        P0:         initial covariance matrix
        target_id:  target ID (used to select motion model)
        env:        environment (for motion_model and motion_params)
        H_fcn:      measurement function returning (H, measurement)

    Returns:
        propagated: list of dicts with {"t": timestep, "x": propagated state, 
                                        "P": propagated covariance,
                                        "H": observation matrix,
                                        "measurement": measurement vector}
    """

    n = x0.shape[0]
    Phi0 = np.eye(n).reshape(-1)

    t_vec = [entry["t"] for entry in track]

    propagated = []

    # --- Propagate from t=0 to first timestep in track ---
    x_curr = x0.copy()
    P_curr = P0.copy()
    t_first = t_vec[0]

    if t_first > 0:
        if env.motion_model[target_id] == "T":
            x_curr, Phi = propagate_state_and_stm(
                0.0, t_first, x_curr, Phi0,
                f_ct_linear, Fx_ct_linear, n
            )
        else:
            x_curr, Phi = propagate_state_and_stm(
                0.0, t_first, x_curr, Phi0,
                f_cv_cont, Fx_cv_cont, n
            )
        Phi = Phi.reshape(n, n)
        P_curr = Phi @ P_curr @ Phi.T

    H, Gk = H_fcn(x_curr[:4])
    propagated.append({
        "t": t_first,
        "x": x_curr.copy(),
        "P": P_curr.copy(),
        "H": H,
        "measurement": Gk
    })

    # --- Propagate through remaining timesteps in track ---
    for k in range(len(t_vec) - 1):
        t0 = t_vec[k]
        t1 = t_vec[k + 1]

        if env.motion_model[target_id] == "T":
            x_next, Phi = propagate_state_and_stm(
                t0, t1, x_curr, Phi0,
                f_ct_linear, Fx_ct_linear, n, env.motion_params[target_id]
            )
        else:
            x_next, Phi = propagate_state_and_stm(
                t0, t1, x_curr, Phi0,
                f_cv_cont, Fx_cv_cont, n
            )

        Phi = Phi.reshape(n, n)
        P_next = Phi @ P_curr @ Phi.T

        H, Gk = H_fcn(x_next[:4])
        propagated.append({
            "t": t1,
            "x": x_next.copy(),
            "P": P_next.copy(),
            "H": H,
            "measurement": Gk
        })

        x_curr = x_next
        P_curr = P_next

    return propagated

def plot_pareto(detection_count, exceedFOV):
    """
    detection_count: list of objective 1 values (to maximize)
    exceedFOV:       list of objective 2 values (to minimize)
    """
    assert len(detection_count) == len(exceedFOV), "Lists must have same length"
    points = list(zip(detection_count, exceedFOV))

    # --- compute Pareto front (maximize detection_count, minimize exceedFOV) ---
    pareto_idx = []
    for i, (d_i, e_i) in enumerate(points):
        dominated = False
        for j, (d_j, e_j) in enumerate(points):
            if j == i:
                continue
            # j dominates i if: >= detections AND <= exceedFOV
            # and strictly better in at least one
            if (d_j >= d_i and e_j <= e_i) and (d_j > d_i or e_j < e_i):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    # sort Pareto points by exceedFOV for a nicer line plot
    pareto_points = sorted([(exceedFOV[i], detection_count[i]) for i in pareto_idx])

    # --- plotting ---
    plt.figure()
    # all solutions
    plt.scatter(exceedFOV, detection_count, c="lightgray", label="All runs")
    # Pareto front
    if pareto_points:
        xs, ys = zip(*pareto_points)
        plt.plot(xs, ys, "-o", color="tab:red", label="Pareto front")

    plt.xlabel("Number of exceed FOV events")
    plt.ylabel("Detection count")
    plt.title("MCTS Pareto front: detections vs exceed-FOV")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pareto_idx, pareto_points

def plot_pareto_multi(algorithms, color_map=None):
    """
    algorithms: list of tuples (detection_count, exceedFOV, label)
        detection_count: list of objective 1 values (maximize)
        exceedFOV:       list of objective 2 values (minimize)
        label:           string identifying the algorithm
        
    Example call:
    plot_pareto_multi([
        (det_count1, exceed1, 'Algo1'),
        (det_count2, exceed2, 'Algo2'),
        (det_count3, exceed3, 'Algo3')
    ])
    """
    
    if color_map is None:
        color_map = {}

    # --- build global list of points with algorithm index ---
    all_points = []
    for a_idx, (det_list, fov_list, label) in enumerate(algorithms):
        assert len(det_list) == len(fov_list)
        for d, f in zip(det_list, fov_list):
            all_points.append((d, f, label, a_idx))

    # --- compute Pareto front ---
    pareto_idx = []
    for i, (d_i, e_i, _, _) in enumerate(all_points):
        dominated = False
        for j, (d_j, e_j, _, _) in enumerate(all_points):
            if j == i:
                continue
            if (d_j >= d_i and e_j <= e_i) and (d_j > d_i or e_j < e_i):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(i)

    # sorted only for drawing the line (but remember original indices)
    pareto_sorted = sorted(pareto_idx, key=lambda i: all_points[i][1])

    # --- plotting ---
    plt.figure()

    # fallback colormap (only used if label not in color_map)
    cmap = plt.cm.get_cmap("tab10", len(algorithms))

    # 1) scatter all points for each algorithm
    for a_idx, (det_list, fov_list, label) in enumerate(algorithms):
        color = color_map.get(label, cmap(a_idx))

        plt.scatter(
            fov_list,
            det_list,
            color=color,
            label=label,
            alpha=0.7
        )

    # 2) connect Pareto points with a black line
    xs = [all_points[i][1] for i in pareto_sorted]
    ys = [all_points[i][0] for i in pareto_sorted]
    plt.plot(xs, ys, "-", color="black", linewidth=2, label="Pareto front")

    # 3) re-plot the Pareto points individually in their algorithm colors
    for i in pareto_sorted:
        d, e, label, a_idx = all_points[i]
        color = color_map.get(label, cmap(a_idx))

        plt.scatter(
            e,
            d,
            color=color,
            edgecolor="black",
            s=70,
            linewidth=0.8
        )

    plt.xlabel("Number of exceed-FOV events")
    plt.ylabel("Detection count")
    plt.title("Pareto Front Comparison Across Algorithms")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pareto_idx, pareto_sorted



def run_mcts_step(env, mcts, obs):
    """
    Runs one MCTS planning step and applies the chosen macro+micro action.
    Returns: next_obs, reward, done, truncated, info, macro, micro
    """

    # Plan with MCTS
    mcts.reset_tree(obs)
    best_child_node = mcts.choose()

    macro = best_child_node.macro_action
    micro = best_child_node.micro_action

    # Execute the action exactly as in your run_episode code
    if macro == 0:     # SEARCH
        next_obs, reward_vec, done, truncated, info = env.search_env.env.step(micro)
        _sync_envs(env.search_env.env, env)
        _sync_envs(env.search_env.env, env.track_env.env)
        trackingNeeded, trackingReallyNeeded = env._compute_track_reward(env)
        if (sum(trackingNeeded) + sum(trackingReallyNeeded))>=1:
            # tracking nedded instead of 
            reward = -0.5 * (sum(trackingNeeded) + sum(trackingReallyNeeded))
        else:
            reward = 1
    else:              # TRACK
        next_obs, reward_vec, done, truncated, info = env.track_env.env.step(micro)
        _sync_envs(env.track_env.env, env)
        _sync_envs(env.track_env.env, env.search_env.env)
        trackingNeeded, trackingReallyNeeded = env._compute_track_reward(env)
        if any(trackingNeeded):
            reward = 2.0
        elif any(trackingReallyNeeded):
            reward = 3.0
        else:
            reward = -1.0

    # Re-root MCTS for next iteration
    mcts.root = mcts.re_root(best_child_node)

    return next_obs, reward, done, truncated, info, macro, micro

def plot_task_log(task_log, title="Action Timeline"):
    # Extract ordered timesteps
    timesteps = sorted(task_log.keys())
    # Extract actions in the correct order
    actions = [task_log[t]["action"] for t in timesteps]

    plt.figure(figsize=(10, 3))
    plt.step(timesteps, actions, where='post')
    plt.scatter(timesteps, actions)
    plt.yticks([0, 1], ["Search", "Track"])
    plt.xlabel("Timestep")
    plt.ylabel("Action")
    plt.title(title)
    plt.grid(True)
    
    # Create output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "evaluated")
    os.makedirs(output_dir, exist_ok=True)

    # --- sanitize ylabel for filename ---
    safe_ylabel = re.sub(r'[^a-zA-Z0-9]+', '_', title).strip('_').lower()

    filename = f"timeline_{safe_ylabel}.pdf"
    filepath = os.path.join(output_dir, filename)

    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def main():
    seeds = [42, 123, 321]
    n_episodes = 10
    n_targets = 5
    n_unknown_targets =100
    boundary = np.sqrt(1e-2)

    """ envMcts = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    mcts = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.95)
    rewardsMcts, detection_countMcts, exceedFOVMcts, last_envMcts, last_episode_logMcts, knownMcts, last_tasks_logMcts = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts, n_episodes=n_episodes, random_policy=True, deterministic_policy=False)  """

    # Heuristic tracker
    tracker = "heuristic"
    """ envHeuristic = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=tracker)
    start = time.perf_counter()
    rewardsHeuristicHeuristic, detection_countHeuristicHeuristic, exceedFOVHeuristicHeuristic, allTracesHeuristicHeuristic, last_episode_logHeuristic, knownHeuristicHeuristic, last_tasks_logHeuristicHeuristic, illegal_heuristicHeuristic = evaluate_agent_macro(seeds=seeds, env=envHeuristic, model=None, n_episodes=n_episodes, random_policy=False, deterministic_policy=True)
    print(time.perf_counter() - start)
    print("Reward - Heuristic macro - Heuristic tracker")
    print(sum(rewardsHeuristicHeuristic)/len(rewardsHeuristicHeuristic))
    print(np.std([np.mean(arr) for arr in rewardsHeuristicHeuristic], ddof=1))
    print("Detections")
    print(sum(detection_countHeuristicHeuristic)/len(detection_countHeuristicHeuristic))
    print(np.std([np.mean(arr) for arr in detection_countHeuristicHeuristic], ddof=1))
    print("Lost")
    print(sum(exceedFOVHeuristicHeuristic)/len(exceedFOVHeuristicHeuristic))
    print(np.std([np.mean(arr) for arr in exceedFOVHeuristicHeuristic], ddof=1))
    print("Known")
    print(sum(knownHeuristicHeuristic)/len(knownHeuristicHeuristic))
    print(np.std([np.mean(arr) for arr in knownHeuristicHeuristic], ddof=1))
    print("Illegal")
    print(sum(illegal_heuristicHeuristic)/len(illegal_heuristicHeuristic))
    print(np.std([np.mean(arr) for arr in illegal_heuristicHeuristic], ddof=1))
    print("Covariance trace sum")
    print(sum(allTracesHeuristicHeuristic) / len(allTracesHeuristicHeuristic))
    print(np.std(allTracesHeuristicHeuristic, ddof=1)) """


    # Mask PPO tracker
    """ tracker = "maskableppo"
    envHeuristic = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=tracker)
    start = time.perf_counter()
    rewardsHeuristicMaskable, detection_countHeuristicMaskable, exceedFOVHeuristicMaskable, allTracesHeuristicMaskable, last_episode_logHeuristicMaskable, knownHeuristicMaskable, last_tasks_logHeuristicMaskable, illegal_heuristicMaskable = evaluate_agent_macro(seeds=seeds, env=envHeuristic, model=None, n_episodes=n_episodes, random_policy=False, deterministic_policy=True)
    print(time.perf_counter() - start)
    print("Reward - Heuristic macro - MaskablePPO 30 targets")
    print(sum(rewardsHeuristicMaskable)/len(rewardsHeuristicMaskable))
    print(np.std([np.mean(arr) for arr in rewardsHeuristicMaskable], ddof=1))
    print("Detections")
    print(sum(detection_countHeuristicMaskable)/len(detection_countHeuristicMaskable))
    print(np.std([np.mean(arr) for arr in detection_countHeuristicMaskable], ddof=1))
    print("Lost")
    print(sum(exceedFOVHeuristicMaskable)/len(exceedFOVHeuristicMaskable))
    print(np.std([np.mean(arr) for arr in exceedFOVHeuristicMaskable], ddof=1))
    print("Known")
    print(sum(knownHeuristicMaskable)/len(knownHeuristicMaskable))
    print(np.std([np.mean(arr) for arr in knownHeuristicMaskable], ddof=1))
    print("Illegal")
    print(sum(illegal_heuristicMaskable)/len(illegal_heuristicMaskable))
    print(np.std([np.mean(arr) for arr in illegal_heuristicMaskable], ddof=1))
    print("Covariance trace sum")
    print(sum(allTracesHeuristicMaskable) / len(allTracesHeuristicMaskable))
    print(np.std(allTracesHeuristicMaskable, ddof=1)) """

    """ envRandom = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=tracker)
    start = time.perf_counter()
    rewardsRandom, detection_countRandom, exceedFOVRandom, allTracesRandom, last_episode_logRandom, knownRandom, last_tasks_logRandom, illegal_random = evaluate_agent_macro(seeds=seeds, env=envRandom, model=None, n_episodes=n_episodes, random_policy=True, deterministic_policy=False)
    print(time.perf_counter() - start)
    print("Reward - Random macro - Heuristic tracker")
    print(sum(rewardsRandom)/len(rewardsRandom))
    print(np.std([np.mean(arr) for arr in rewardsRandom], ddof=1))
    print("Detections")
    print(sum(detection_countRandom)/len(detection_countRandom))
    print(np.std([np.mean(arr) for arr in detection_countRandom], ddof=1))
    print("Lost")
    print(sum(exceedFOVRandom)/len(exceedFOVRandom))
    print(np.std([np.mean(arr) for arr in exceedFOVRandom], ddof=1))
    print("Known")
    print(sum(knownRandom)/len(knownRandom))
    print(np.std([np.mean(arr) for arr in knownRandom], ddof=1))
    print("Illegal")
    print(sum(illegal_random)/len(illegal_random))
    print(np.std([np.mean(arr) for arr in illegal_random], ddof=1))
    print("Covariance trace sum")
    print(sum(allTracesRandom) / len(allTracesRandom))
    print(np.std(allTracesRandom, ddof=1))

    envPPO = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=tracker)
    ppo_model = PPO.load("agents/ppo_macro_trained_heuristic_track_perfectSearch_rewardNorm", env=envPPO)
    rewardsPPO, detection_countPPO, exceedFOVPPO, allTracesPpo, last_episode_logPPO, knownPPO, last_tasks_logPPO, illegal_ppo = evaluate_agent_macro(seeds=seeds, env=envPPO, model=ppo_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print("Reward - PPO macro - Heuristic tracker")
    print(sum(rewardsPPO)/len(rewardsPPO))
    print(np.std([np.mean(arr) for arr in rewardsPPO], ddof=1))
    print("Detections")
    print(sum(detection_countPPO)/len(detection_countPPO))
    print(np.std([np.mean(arr) for arr in detection_countPPO], ddof=1))
    print("Lost")
    print(sum(exceedFOVPPO)/len(exceedFOVPPO))
    print(np.std([np.mean(arr) for arr in exceedFOVPPO], ddof=1))
    print("Known")
    print(sum(knownPPO)/len(knownPPO))
    print(np.std([np.mean(arr) for arr in knownPPO], ddof=1))
    print("Illegal")
    print(sum(illegal_ppo)/len(illegal_ppo))
    print(np.std([np.mean(arr) for arr in illegal_ppo], ddof=1))
    print("Covariance trace sum")
    print(sum(allTracesPpo) / len(allTracesPpo))
    print(np.std(allTracesPpo, ddof=1)) """

    envDQN = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=tracker, boundary=boundary)
    dqn_model = DQN.load("agents/dqn_macro_trained_heuristic_track_predState", env=envDQN)
    start = time.perf_counter()
    rewardsDQN, detection_countDQN, exceedFOVDQN, allTracesDqn, last_episode_logDQN, knownDQN, last_tasks_logDQN, illegal_dqn = evaluate_agent_macro(seeds=seeds, env=envDQN, model=dqn_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print(time.perf_counter() - start)
    print("Reward - DQN macro - Heuristic tracker")
    print(sum(rewardsDQN)/len(rewardsDQN))
    print(np.std([np.mean(arr) for arr in rewardsDQN], ddof=1))
    print("Detections")
    print(sum(detection_countDQN)/len(detection_countDQN))
    print(np.std([np.mean(arr) for arr in detection_countDQN], ddof=1))
    print("Lost")
    print(sum(exceedFOVDQN)/len(exceedFOVDQN))
    print(np.std([np.mean(arr) for arr in exceedFOVDQN], ddof=1))
    print("Known")
    print(sum(knownDQN)/len(knownDQN))
    print(np.std([np.mean(arr) for arr in knownDQN], ddof=1))
    print("Illegal")
    print(sum(illegal_dqn)/len(illegal_dqn))
    print(np.std([np.mean(arr) for arr in illegal_dqn], ddof=1))
    print("Covariance trace sum")
    print(sum(allTracesDqn) / len(allTracesDqn))
    print(np.std(allTracesDqn, ddof=1))

    #plot_pareto(detection_countMcts, exceedFOVMcts)

    # ****** Plot reward distributions ******
    reward_results = {
        #"Random": rewardsRandom,
        #"PPO": rewardsHeuristicMaskable,
        #"Heuristic": rewardsHeuristic,
        #"HeuristicHeuristic": rewardsHeuristicHeuristic,
        "DQN": rewardsDQN
        #"MCTS": rewardsMcts
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        #"Random": detection_countRandom,
        #"PPO": detection_countHeuristicMaskable,
        #"Heuristic": detection_countHeuristic,
        #"HeuristicHeuristic": detection_countHeuristicHeuristic,
        "DQN": detection_countDQN
        #"MCTS": detection_countMcts
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    """ illegal_results = {
        "Random": illegal_random,
        "PPO": illegal_ppo,
        "Heuristic": illegal_heuristic,
        "DQN": illegal_dqn,
        #"MCTS": detection_countMcts
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions") """

    # ****** Plot total target lost ******
    labels = ["PPO", "DQN", "Random", "Heuristic"]
    #plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=[], knownDQN=[], random=exceedFOVRandom, knownRandom=knownRandom, det=[], knowndet=[])
    #plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=exceedFOVDQN, knownDQN=knownDQN, random=exceedFOVRandom, knownRandom=knownRandom, det=exceedFOVHeuristic, knowndet=knownHeuristic, labels=labels)

    # ****** Plot time spent on task ******
    #plot_task_log(last_tasks_logRandom, title="Random Policy - Action Timeline")
    #plot_task_log(last_tasks_logPPO, title="PPO - Action Timeline")
    plot_task_log(last_tasks_logDQN[-1], title="DQN - Action Timeline")
    #plot_task_log(last_tasks_logHeuristic, title="Heuristic - Action Timeline")
    #plot_task_log(last_tasks_logHeuristicHeuristic, title="HeuristicHeuristic - Action Timeline")
    #plot_task_log(last_tasks_logHeuristicMaskable, title="HeuristicMaskable - Action Timeline")

    


    # DQN tracker
    """ tracker = "dqn"
    envHeuristic = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker=tracker)
    rewardsHeuristic, detection_countHeuristic, exceedFOVHeuristic, last_envHeuristic, last_episode_logHeuristic, knownHeuristic, last_tasks_logHeuristic, illegal_heuristic = evaluate_agent_macro(seeds=seeds, env=envHeuristic, model=None, n_episodes=n_episodes, random_policy=False, deterministic_policy=True)
    print("Reward - Heuristic macro - DQN tracker")
    print(sum(rewardsHeuristic)/len(rewardsHeuristic))
    print(np.std([np.mean(arr) for arr in rewardsHeuristic], ddof=1))
    print("Detections")
    print(sum(detection_countHeuristic)/len(detection_countHeuristic))
    print(np.std([np.mean(arr) for arr in detection_countHeuristic], ddof=1))
    print("Lost")
    print(sum(exceedFOVHeuristic)/len(exceedFOVHeuristic))
    print(np.std([np.mean(arr) for arr in exceedFOVHeuristic], ddof=1))
    print("Known")
    print(sum(knownHeuristic)/len(knownHeuristic))
    print(np.std([np.mean(arr) for arr in knownHeuristic], ddof=1))
    print("Illegal")
    print(sum(illegal_heuristic)/len(illegal_heuristic))
    print(np.std([np.mean(arr) for arr in illegal_heuristic], ddof=1))

    envRandom = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker=tracker)
    rewardsRandom, detection_countRandom, exceedFOVRandom, last_envRandom, last_episode_logRandom, knownRandom, last_tasks_logRandom, illegal_random = evaluate_agent_macro(seeds=seeds, env=envRandom, model=None, n_episodes=n_episodes, random_policy=True, deterministic_policy=False)
    print("Reward - Random macro - DQN tracker")
    print(sum(rewardsRandom)/len(rewardsRandom))
    print(np.std([np.mean(arr) for arr in rewardsRandom], ddof=1))
    print("Detections")
    print(sum(detection_countRandom)/len(detection_countRandom))
    print(np.std([np.mean(arr) for arr in detection_countRandom], ddof=1))
    print("Lost")
    print(sum(exceedFOVRandom)/len(exceedFOVRandom))
    print(np.std([np.mean(arr) for arr in exceedFOVRandom], ddof=1))
    print("Known")
    print(sum(knownRandom)/len(knownRandom))
    print(np.std([np.mean(arr) for arr in knownRandom], ddof=1))
    print("Illegal")
    print(sum(illegal_random)/len(illegal_random))
    print(np.std([np.mean(arr) for arr in illegal_random], ddof=1))

    envPPO = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker=tracker)
    ppo_model = PPO.load("agents/ppo_macro_trained_dqn_track", env=envPPO)
    rewardsPPO, detection_countPPO, exceedFOVPPO, last_envPPO, last_episode_logPPO, knownPPO, last_tasks_logPPO, illegal_ppo = evaluate_agent_macro(seeds=seeds, env=envPPO, model=ppo_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print("Reward - PPO macro - DQN tracker")
    print(sum(rewardsPPO)/len(rewardsPPO))
    print(np.std([np.mean(arr) for arr in rewardsPPO], ddof=1))
    print("Detections")
    print(sum(detection_countPPO)/len(detection_countPPO))
    print(np.std([np.mean(arr) for arr in detection_countPPO], ddof=1))
    print("Lost")
    print(sum(exceedFOVPPO)/len(exceedFOVPPO))
    print(np.std([np.mean(arr) for arr in exceedFOVPPO], ddof=1))
    print("Known")
    print(sum(knownPPO)/len(knownPPO))
    print(np.std([np.mean(arr) for arr in knownPPO], ddof=1))
    print("Illegal")
    print(sum(illegal_ppo)/len(illegal_ppo))
    print(np.std([np.mean(arr) for arr in illegal_ppo], ddof=1))

    envDQN = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker=tracker)
    dqn_model = DQN.load("agents/dqn_macro_trained_dqn_track", env=envDQN)
    rewardsDQN, detection_countDQN, exceedFOVDQN, last_envDQN, last_episode_logDQN, knownDQN, last_tasks_logDQN, illegal_dqn = evaluate_agent_macro(seeds=seeds, env=envDQN, model=dqn_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print("Reward - DQN macro - DQN tracker")
    print(sum(rewardsDQN)/len(rewardsDQN))
    print(np.std([np.mean(arr) for arr in rewardsDQN], ddof=1))
    print("Detections")
    print(sum(detection_countDQN)/len(detection_countDQN))
    print(np.std([np.mean(arr) for arr in detection_countDQN], ddof=1))
    print("Lost")
    print(sum(exceedFOVDQN)/len(exceedFOVDQN))
    print(np.std([np.mean(arr) for arr in exceedFOVDQN], ddof=1))
    print("Known")
    print(sum(knownDQN)/len(knownDQN))
    print(np.std([np.mean(arr) for arr in knownDQN], ddof=1))
    print("Illegal")
    print(sum(illegal_dqn)/len(illegal_dqn))
    print(np.std([np.mean(arr) for arr in illegal_dqn], ddof=1))

    #plot_pareto(detection_countMcts, exceedFOVMcts)

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": rewardsRandom,
        "PPO": rewardsPPO,
        "Heuristic": rewardsHeuristic,
        "DQN": rewardsDQN
        #"MCTS": rewardsMcts
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        "Random": detection_countRandom,
        "PPO": detection_countPPO,
        "Heuristic": detection_countHeuristic,
        "DQN": detection_countDQN,
        #"MCTS": detection_countMcts
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    illegal_results = {
        "Random": illegal_random,
        "PPO": illegal_ppo,
        "Heuristic": illegal_heuristic,
        "DQN": illegal_dqn,
        #"MCTS": detection_countMcts
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions")

    # ****** Plot total target lost ******
    #plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=[], knownDQN=[], random=exceedFOVRandom, knownRandom=knownRandom, det=[], knowndet=[])
    plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=exceedFOVDQN, knownDQN=knownDQN, random=exceedFOVRandom, knownRandom=knownRandom, det=exceedFOVHeuristic, knowndet=knownHeuristic, labels=labels)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logRandom, title="Random Policy - Action Timeline")
    plot_task_log(last_tasks_logPPO, title="PPO - Action Timeline")
    plot_task_log(last_tasks_logDQN, title="DQN - Action Timeline")
    plot_task_log(last_tasks_logHeuristic, title="Heuristic - Action Timeline")
    #plot_task_log(last_tasks_logMcts, title="MCTS - Action Timeline")


    # ****** Non-dominated solution plot******
    plot_pareto_multi(
    [
        (detection_countPPO, exceedFOVPPO, 'PPO'),
        (detection_countDQN, exceedFOVDQN, 'DQN'),
        #(detection_countMcts, exceedFOVMcts, 'MCTS'),
        (detection_countHeuristic, exceedFOVHeuristic, 'Heuristic'),
        (detection_countRandom, exceedFOVRandom, 'Random')
    ],
    color_map={
        'PPO': 'blue',
        'DQN': 'yellow',
        'MCTS': 'purple',
        'Random': 'red',
        'Heuristic': 'green',
    }
    )


    # Heuristic tracker
    envHeuristic = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=None)
    rewardsHeuristic, detection_countHeuristic, exceedFOVHeuristic, last_envHeuristic, last_episode_logHeuristic, knownHeuristic, last_tasks_logHeuristic, illegal_heuristic = evaluate_agent_macro(seeds=seeds, env=envHeuristic, model=None, n_episodes=n_episodes, random_policy=False, deterministic_policy=True)
    print("Reward - Heuristic macro - Heuristic tracker")
    print(sum(rewardsHeuristic)/len(rewardsHeuristic))
    print(np.std([np.mean(arr) for arr in rewardsHeuristic], ddof=1))
    print("Detections")
    print(sum(detection_countHeuristic)/len(detection_countHeuristic))
    print(np.std([np.mean(arr) for arr in detection_countHeuristic], ddof=1))
    print("Lost")
    print(sum(exceedFOVHeuristic)/len(exceedFOVHeuristic))
    print(np.std([np.mean(arr) for arr in exceedFOVHeuristic], ddof=1))
    print("Known")
    print(sum(knownHeuristic)/len(knownHeuristic))
    print(np.std([np.mean(arr) for arr in knownHeuristic], ddof=1))
    print("Illegal")
    print(sum(illegal_heuristic)/len(illegal_heuristic))
    print(np.std([np.mean(arr) for arr in illegal_heuristic], ddof=1))

    envRandom = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True,  tracker=None)
    rewardsRandom, detection_countRandom, exceedFOVRandom, last_envRandom, last_episode_logRandom, knownRandom, last_tasks_logRandom, illegal_random = evaluate_agent_macro(seeds=seeds, env=envRandom, model=None, n_episodes=n_episodes, random_policy=True, deterministic_policy=False)
    print("Reward - Random macro - Heuristic tracker")
    print(sum(rewardsRandom)/len(rewardsRandom))
    print(np.std([np.mean(arr) for arr in rewardsRandom], ddof=1))
    print("Detections")
    print(sum(detection_countRandom)/len(detection_countRandom))
    print(np.std([np.mean(arr) for arr in detection_countRandom], ddof=1))
    print("Lost")
    print(sum(exceedFOVRandom)/len(exceedFOVRandom))
    print(np.std([np.mean(arr) for arr in exceedFOVRandom], ddof=1))
    print("Known")
    print(sum(knownRandom)/len(knownRandom))
    print(np.std([np.mean(arr) for arr in knownRandom], ddof=1))
    print("Illegal")
    print(sum(illegal_random)/len(illegal_random))
    print(np.std([np.mean(arr) for arr in illegal_random], ddof=1))

    envPPO = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=None)
    ppo_model = PPO.load("agents/ppo_macro_trained", env=envPPO)
    rewardsPPO, detection_countPPO, exceedFOVPPO, last_envPPO, last_episode_logPPO, knownPPO, last_tasks_logPPO, illegal_ppo = evaluate_agent_macro(seeds=seeds, env=envPPO, model=ppo_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print("Reward - PPO macro - Heuristic tracker")
    print(sum(rewardsPPO)/len(rewardsPPO))
    print(np.std([np.mean(arr) for arr in rewardsPPO], ddof=1))
    print("Detections")
    print(sum(detection_countPPO)/len(detection_countPPO))
    print(np.std([np.mean(arr) for arr in detection_countPPO], ddof=1))
    print("Lost")
    print(sum(exceedFOVPPO)/len(exceedFOVPPO))
    print(np.std([np.mean(arr) for arr in exceedFOVPPO], ddof=1))
    print("Known")
    print(sum(knownPPO)/len(knownPPO))
    print(np.std([np.mean(arr) for arr in knownPPO], ddof=1))
    print("Illegal")
    print(sum(illegal_ppo)/len(illegal_ppo))
    print(np.std([np.mean(arr) for arr in illegal_ppo], ddof=1))

    envDQN = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=None)
    dqn_model = DQN.load("agents/dqn_macro_trained", env=envDQN)
    rewardsDQN, detection_countDQN, exceedFOVDQN, last_envDQN, last_episode_logDQN, knownDQN, last_tasks_logDQN, illegal_dqn = evaluate_agent_macro(seeds=seeds, env=envDQN, model=dqn_model, n_episodes=n_episodes, random_policy=False, deterministic_policy=False)
    print("Reward - DQN macro - Heuristic tracker")
    print(sum(rewardsDQN)/len(rewardsDQN))
    print(np.std([np.mean(arr) for arr in rewardsDQN], ddof=1))
    print("Detections")
    print(sum(detection_countDQN)/len(detection_countDQN))
    print(np.std([np.mean(arr) for arr in detection_countDQN], ddof=1))
    print("Lost")
    print(sum(exceedFOVDQN)/len(exceedFOVDQN))
    print(np.std([np.mean(arr) for arr in exceedFOVDQN], ddof=1))
    print("Known")
    print(sum(knownDQN)/len(knownDQN))
    print(np.std([np.mean(arr) for arr in knownDQN], ddof=1))
    print("Illegal")
    print(sum(illegal_dqn)/len(illegal_dqn))
    print(np.std([np.mean(arr) for arr in illegal_dqn], ddof=1))

    #plot_pareto(detection_countMcts, exceedFOVMcts)

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": rewardsRandom,
        "PPO": rewardsPPO,
        "Heuristic": rewardsHeuristic,
        "DQN": rewardsDQN
        #"MCTS": rewardsMcts
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        "Random": detection_countRandom,
        "PPO": detection_countPPO,
        "Heuristic": detection_countHeuristic,
        "DQN": detection_countDQN,
        #"MCTS": detection_countMcts
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    illegal_results = {
        "Random": illegal_random,
        "PPO": illegal_ppo,
        "Heuristic": illegal_heuristic,
        "DQN": illegal_dqn,
        #"MCTS": detection_countMcts
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions")

    # ****** Plot total target lost ******
    #plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=[], knownDQN=[], random=exceedFOVRandom, knownRandom=knownRandom, det=[], knowndet=[])
    plot_means_lost_targets(ppo=exceedFOVPPO, knownPPO=knownPPO, dqn=exceedFOVDQN, knownDQN=knownDQN, random=exceedFOVRandom, knownRandom=knownRandom, det=exceedFOVHeuristic, knowndet=knownHeuristic, labels=labels)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logRandom, title="Random Policy - Action Timeline")
    plot_task_log(last_tasks_logPPO, title="PPO - Action Timeline")
    plot_task_log(last_tasks_logDQN, title="DQN - Action Timeline")
    plot_task_log(last_tasks_logHeuristic, title="Heuristic - Action Timeline") """

def mctsMain():
    seeds = [42, 123, 321]
    n_episodes = 1
    n_targets = 5
    n_unknown_targets =100
    rollout_depth = 50
    n_simulations = 1000

    # heuristic tracker
    envMcts = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=True, tracker=None)
    mcts095 = MCTSenv.MCTS(env=envMcts, rollout_depth=rollout_depth, n_simulations=n_simulations, gamma=0.95)
    rewardsMcts, detection_countMcts095, exceedFOVMcts095, last_envMcts, last_episode_logMcts, knownMcts095, last_tasks_logMcts095, illegal_Mcts095 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts095, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.95 gamma - Heuristic tracker")
    print(sum(detection_countMcts095)/len(detection_countMcts095))
    print(np.std([np.mean(arr) for arr in detection_countMcts095], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts095)/len(exceedFOVMcts095))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts095], ddof=1))
    print("Known")
    print(sum(knownMcts095)/len(knownMcts095))
    print(np.std([np.mean(arr) for arr in knownMcts095], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts095)/len(illegal_Mcts095))
    print(np.std([np.mean(arr) for arr in illegal_Mcts095], ddof=1))

    mcts05 = MCTSenv.MCTS(env=envMcts, rollout_depth=rollout_depth, n_simulations=n_simulations, gamma=0.5)
    rewardsMcts, detection_countMcts05, exceedFOVMcts05, last_envMcts, last_episode_logMcts, knownMcts05, last_tasks_logMcts05, illegal_Mcts05 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts05, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.5 gamma - Heuristic tracker")
    print(sum(detection_countMcts05)/len(detection_countMcts05))
    print(np.std([np.mean(arr) for arr in detection_countMcts05], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts05)/len(exceedFOVMcts05))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts05], ddof=1))
    print("Known")
    print(sum(knownMcts05)/len(knownMcts05))
    print(np.std([np.mean(arr) for arr in knownMcts05], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts05)/len(illegal_Mcts05))
    print(np.std([np.mean(arr) for arr in illegal_Mcts05], ddof=1))

    mcts01 = MCTSenv.MCTS(env=envMcts, rollout_depth=rollout_depth, n_simulations=n_simulations, gamma=0.1)
    rewardsMcts, detection_countMcts01, exceedFOVMcts01, last_envMcts, last_episode_logMcts, knownMcts01, last_tasks_logMcts01, illegal_Mcts01 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts01, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.1 gamma - Heuristic tracker")
    print(sum(detection_countMcts01)/len(detection_countMcts01))
    print(np.std([np.mean(arr) for arr in detection_countMcts01], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts01)/len(exceedFOVMcts01))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts01], ddof=1))
    print("Known")
    print(sum(knownMcts01)/len(knownMcts01))
    print(np.std([np.mean(arr) for arr in knownMcts01], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts01)/len(illegal_Mcts01))
    print(np.std([np.mean(arr) for arr in illegal_Mcts01], ddof=1))

    # ****** Plot detection distributions ******
    detection_results = {
        "0.95": detection_countMcts095,
        "0.5": detection_countMcts05,
        "0.1": detection_countMcts01
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    illegal_results = {
        "0.95": illegal_Mcts095,
        "0.5": illegal_Mcts05,
        "0.1": illegal_Mcts01
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions")

    # ****** Plot total target lost ******
    labels = ["0.95", "0.5", "0.1", " "]
    plot_means_lost_targets(ppo=exceedFOVMcts095, knownPPO=knownMcts095, dqn=exceedFOVMcts05, knownDQN=knownMcts05, random=exceedFOVMcts01, knownRandom=knownMcts01, det=[], knowndet=[], labels=labels)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logMcts095, title="Gamma 0.95 - Action Timeline")
    plot_task_log(last_tasks_logMcts05, title="Gamma 0.5 - Action Timeline")
    plot_task_log(last_tasks_logMcts01, title="Gamma 0.1 - Action Timeline")

    # mask ppo tracker
    """ envMcts = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker="maskppo")
    mcts095 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.95)
    rewardsMcts, detection_countMcts095, exceedFOVMcts095, last_envMcts, last_episode_logMcts, knownMcts095, last_tasks_logMcts095, illegal_Mcts095 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts095, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.95 gamma - maskppo tracker")
    print(sum(detection_countMcts095)/len(detection_countMcts095))
    print(np.std([np.mean(arr) for arr in detection_countMcts095], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts095)/len(exceedFOVMcts095))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts095], ddof=1))
    print("Known")
    print(sum(knownMcts095)/len(knownMcts095))
    print(np.std([np.mean(arr) for arr in knownMcts095], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts095)/len(illegal_Mcts095))
    print(np.std([np.mean(arr) for arr in illegal_Mcts095], ddof=1))

    mcts05 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.5)
    rewardsMcts, detection_countMcts05, exceedFOVMcts05, last_envMcts, last_episode_logMcts, knownMcts05, last_tasks_logMcts05, illegal_Mcts05 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts05, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.5 gamma - maskppo tracker")
    print(sum(detection_countMcts05)/len(detection_countMcts05))
    print(np.std([np.mean(arr) for arr in detection_countMcts05], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts05)/len(exceedFOVMcts05))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts05], ddof=1))
    print("Known")
    print(sum(knownMcts05)/len(knownMcts05))
    print(np.std([np.mean(arr) for arr in knownMcts05], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts05)/len(illegal_Mcts05))
    print(np.std([np.mean(arr) for arr in illegal_Mcts05], ddof=1)) """

    """ mcts01 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.1)
    rewardsMcts, detection_countMcts01, exceedFOVMcts01, last_envMcts, last_episode_logMcts, knownMcts01, last_tasks_logMcts01, illegal_Mcts01 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts01, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.1 gamma - maskppo tracker")
    print(sum(detection_countMcts01)/len(detection_countMcts01))
    print(np.std([np.mean(arr) for arr in detection_countMcts01], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts01)/len(exceedFOVMcts01))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts01], ddof=1))
    print("Known")
    print(sum(knownMcts01)/len(knownMcts01))
    print(np.std([np.mean(arr) for arr in knownMcts01], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts01)/len(illegal_Mcts01))
    print(np.std([np.mean(arr) for arr in illegal_Mcts01], ddof=1))

    # ****** Plot detection distributions ******
    detection_results = {
        "0.95": detection_countMcts095,
        "0.5": detection_countMcts05,
        "0.1": detection_countMcts01
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    illegal_results = {
        "0.95": illegal_Mcts095,
        "0.5": illegal_Mcts05,
        "0.1": illegal_Mcts01
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions")

    # ****** Plot total target lost ******
    labels = ["0.95", "0.5", "0.1", " "]
    plot_means_lost_targets(ppo=exceedFOVMcts095, knownPPO=knownMcts095, dqn=exceedFOVMcts05, knownDQN=knownMcts05, random=exceedFOVMcts01, knownRandom=knownMcts01, det=[], knowndet=[], labels=labels)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logMcts095, title="Gamma 0.95 - Action Timeline")
    plot_task_log(last_tasks_logMcts05, title="Gamma 0.5 - Action Timeline")
    plot_task_log(last_tasks_logMcts01, title="Gamma 0.1 - Action Timeline") """

    # dqn tracker
    """ envMcts = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)), heuristicTracker=False, tracker="dqn")
    mcts095 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.95)
    rewardsMcts, detection_countMcts095, exceedFOVMcts095, last_envMcts, last_episode_logMcts, knownMcts095, last_tasks_logMcts095, illegal_Mcts095 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts095, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.95 gamma - DQN tracker")
    print(sum(detection_countMcts095)/len(detection_countMcts095))
    print(np.std([np.mean(arr) for arr in detection_countMcts095], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts095)/len(exceedFOVMcts095))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts095], ddof=1))
    print("Known")
    print(sum(knownMcts095)/len(knownMcts095))
    print(np.std([np.mean(arr) for arr in knownMcts095], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts095)/len(illegal_Mcts095))
    print(np.std([np.mean(arr) for arr in illegal_Mcts095], ddof=1))

    mcts05 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.5)
    rewardsMcts, detection_countMcts05, exceedFOVMcts05, last_envMcts, last_episode_logMcts, knownMcts05, last_tasks_logMcts05, illegal_Mcts05 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts05, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.5 gamma - DQN tracker")
    print(sum(detection_countMcts05)/len(detection_countMcts05))
    print(np.std([np.mean(arr) for arr in detection_countMcts05], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts05)/len(exceedFOVMcts05))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts05], ddof=1))
    print("Known")
    print(sum(knownMcts05)/len(knownMcts05))
    print(np.std([np.mean(arr) for arr in knownMcts05], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts05)/len(illegal_Mcts05))
    print(np.std([np.mean(arr) for arr in illegal_Mcts05], ddof=1))

    mcts01 = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.1)
    rewardsMcts, detection_countMcts01, exceedFOVMcts01, last_envMcts, last_episode_logMcts, knownMcts01, last_tasks_logMcts01, illegal_Mcts01 = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts01, n_episodes=n_episodes, random_policy=True, deterministic_policy=False) 
    print("Detections - 0.1 gamma - DQN tracker")
    print(sum(detection_countMcts01)/len(detection_countMcts01))
    print(np.std([np.mean(arr) for arr in detection_countMcts01], ddof=1))
    print("Lost")
    print(sum(exceedFOVMcts01)/len(exceedFOVMcts01))
    print(np.std([np.mean(arr) for arr in exceedFOVMcts01], ddof=1))
    print("Known")
    print(sum(knownMcts01)/len(knownMcts01))
    print(np.std([np.mean(arr) for arr in knownMcts01], ddof=1))
    print("Illegal")
    print(sum(illegal_Mcts01)/len(illegal_Mcts01))
    print(np.std([np.mean(arr) for arr in illegal_Mcts01], ddof=1))

    # ****** Plot detection distributions ******
    detection_results = {
        "0.95": detection_countMcts095,
        "0.5": detection_countMcts05,
        "0.1": detection_countMcts01
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot illegal actions distributions ******
    illegal_results = {
        "0.95": illegal_Mcts095,
        "0.5": illegal_Mcts05,
        "0.1": illegal_Mcts01
    }
    plot_violin(illegal_results, ylabel="Number of Illegal Actions")

    # ****** Plot total target lost ******
    labels = ["0.95", "0.5", "0.1", " "]
    plot_means_lost_targets(ppo=exceedFOVMcts095, knownPPO=knownMcts095, dqn=exceedFOVMcts05, knownDQN=knownMcts05, random=exceedFOVMcts01, knownRandom=knownMcts01, det=[], knowndet=[], labels=labels)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logMcts095, title="Gamma 0.95 - Action Timeline")
    plot_task_log(last_tasks_logMcts05, title="Gamma 0.5 - Action Timeline")
    plot_task_log(last_tasks_logMcts01, title="Gamma 0.1 - Action Timeline") """
if __name__ == "__main__":
    main()
    #rmsePlot()
    #mctsMain()


    