import copy
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from tqdm import trange
import MCTSenv
from MacroEnv import MacroEnv, _sync_envs
from main import analyse_tracking_task, plot_detection_bar_chart, plot_means_lost_targets, plot_violin
from multi_seed_training_macro import MacroRandomSeedEnv
from multi_target_env import MultiTargetEnv

def evaluate_agent_macro(seeds=None, env=None, model=None, n_episodes=100, random_policy=False, n_targets=5, n_unknown_targets=100):
    rewards = []
    detect_count3 = 0
    detection_count = []
    exceedFOV = []

    # for logging the final episode
    last_episode_log = {}
    last_tasks_log = {}
    last_env = None

    # average number of known targets
    known = []

    for ep in trange(n_episodes, desc="Evaluating",  leave=False):
        obs, _ = env.reset(seeds)
        exceed_target = []
        if ep == n_episodes - 1:
            last_env = copy.deepcopy(env)
            episode_log = {}        # create a temporary log if this is the last episode
            episode_task_log = {}
        done = False
        total_reward = 0.0
        detections = env.init_n_targets
        t = 0  # timestep counter

        while not done:

            if isinstance(model, MCTSenv.MCTS):
                next_obs, reward, done, truncated, info, action, micro = run_mcts_step(env, model, obs)
            else:
                if random_policy:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=False)
                next_obs, reward, done, truncated, info = env.step(action)

            total_reward += reward

            # evaluate searching task
            known_targets = sum(info["action_mask"]["micro_track"])
            if known_targets>detections:
                detect_count3 = detect_count3 + (known_targets - detections)
                detections = known_targets

            # evaluate tracking task
            if ep == n_episodes - 1:
                episode_log[t] = {}
                episode_task_log[t] = {"timestamp": t, "action": int(action)}
            if ep == n_episodes - 1:
                for tgt in range(env.base_env.n_targets):
                    #print(env.base_env.targets[tgt]["id"])
                    exceed, x, P = analyse_tracking_task(next_obs, env.base_env.targets[tgt]["id"], env.base_env, confidence=0.95)
                    """ if exceed and tgt not in exceed_target:
                        exceed_target.append(tgt)
    """
                    # if this is the last episode, store everything
                    if tgt == info["target_id"]:
                        episode_log[t][tgt] = {
                            "id": tgt,
                            "state": x.copy(),
                            "cov": P.copy()
                        }

            t += 1  # increment timestep
        rewards.append(total_reward)
        detection_count.append(env.base_env.detect_counter)
        exceedFOV.append(env.base_env.lost_counter)
        known.append(env.base_env.n_targets)
        # --- For last episode, store deep copy of env ---
        if ep == n_episodes - 1:
            last_episode_log = episode_log
            last_tasks_log = episode_task_log
    return rewards, detection_count, detect_count3, exceedFOV, last_env, last_episode_log, known, last_tasks_log



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

def plot_pareto_multi(algorithms):
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
    pareto_sorted = sorted(pareto_idx, key=lambda i: all_points[i][1])  # by exceedFOV (x-axis)

    # --- plotting ---
    plt.figure()

    # consistent colors for each algorithm
    cmap = plt.cm.get_cmap("tab10", len(algorithms))

    # 1) scatter all points for each algorithm
    for a_idx, (det_list, fov_list, label) in enumerate(algorithms):
        plt.scatter(
            fov_list, det_list,
            color=cmap(a_idx),
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
        plt.scatter(e, d, color=cmap(a_idx), edgecolor="black", s=70, linewidth=0.8)

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
        _sync_envs(env.search_env.env, env.base_env)
        _sync_envs(env.search_env.env, env.track_env.env)
        trackingNeeded, trackingReallyNeeded = env._compute_track_reward(env.base_env)
        if (sum(trackingNeeded) + sum(trackingReallyNeeded))>=1:
            # tracking nedded instead of 
            reward = -0.5 * (sum(trackingNeeded) + sum(trackingReallyNeeded))
        else:
            reward = 1
    else:              # TRACK
        next_obs, reward_vec, done, truncated, info = env.track_env.env.step(micro)
        _sync_envs(env.track_env.env, env.base_env)
        _sync_envs(env.track_env.env, env.search_env.env)
        trackingNeeded, trackingReallyNeeded = env._compute_track_reward(env.base_env)
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
    plt.show()

if __name__ == "__main__":
    seeds = [42, 123, 321]
    n_episodes = 1
    n_targets = 5
    n_unknown_targets =100

    envMcts = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    mcts = MCTSenv.MCTS(env=envMcts, rollout_depth=5, gamma=0.95)
    rewardsMcts, detection_countMcts, detect_count3Mcts, exceedFOVMcts, last_envMcts, last_episode_logMcts, knownMcts, last_tasks_logMcts = evaluate_agent_macro(seeds=seeds, env=envMcts, model=mcts, n_episodes=n_episodes, random_policy=True, n_targets=n_targets, n_unknown_targets=n_unknown_targets)
    plot_pareto(detection_countMcts, exceedFOVMcts)

    envRandom = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    rewardsRandom, detection_countRandom, detect_count3Random, exceedFOVRandom, last_envRandom, last_episode_logRandom, knownRandom, last_tasks_logRandom = evaluate_agent_macro(seeds=seeds, env=envRandom, model=None, n_episodes=n_episodes, random_policy=True, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    envPPO = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    ppo_model = PPO.load("agents/ppo_macro_shorttrained", env=envPPO)
    rewardsPPO, detection_countPPO, detect_count3PPO, exceedFOVPPO, last_envPPO, last_episode_logPPO, knownPPO, last_tasks_logPPO = evaluate_agent_macro(seeds=seeds, env=envPPO, model=ppo_model, n_episodes=n_episodes, random_policy=False, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    envDQN = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    dqn_model = DQN.load("agents/dqn_macro_shorttrained", env=envDQN)
    rewardsDQN, detection_countDQN, detect_count3DQN, exceedFOVDQN, last_envDQN, last_episode_logDQN, knownDQN, last_tasks_logDQN = evaluate_agent_macro(seeds=seeds, env=envDQN, model=dqn_model, n_episodes=n_episodes, random_policy=False, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": rewardsRandom,
        "PPO": rewardsPPO,
        "DQN": rewardsDQN,
        "MCTS": rewardsMcts
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        "Random": detection_countRandom,
        "PPO": detection_countPPO,
        "DQN": detection_countDQN,
        "MCTS": detection_countMcts
    }
    plot_violin(detection_results, ylabel="Number of Detections")

    # ****** Plot total detection ******
    """ detection_results = {
        "Random": detect_count3Random,
        "PPO": detect_count3PPO,
        "DQN": detect_count3DQN
    }
    plot_detection_bar_chart(detection_results) """

    # ****** Plot total target lost ******
    plot_means_lost_targets(exceedFOVPPO, knownPPO, exceedFOVDQN, knownDQN, exceedFOVRandom, knownRandom, exceedFOVMcts, knownMcts)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logRandom, title="Random Policy - Action Timeline")
    plot_task_log(last_tasks_logPPO, title="PPO - Action Timeline")
    plot_task_log(last_tasks_logDQN, title="DQN - Action Timeline")
    plot_task_log(last_tasks_logMcts, title="MCTS - Action Timeline")

    # ****** Non-dominated solution plot******
    plot_pareto_multi([
        (detection_countPPO, exceedFOVPPO, 'PPO'),
        (detection_countDQN, exceedFOVDQN, 'DQN'),
        (detection_countMcts, exceedFOVMcts, 'MCTS'),
        (detection_countRandom, exceedFOVRandom, 'Random')
    ])



    