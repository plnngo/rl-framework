import copy
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines3 import DQN, PPO
from tqdm import trange
from MacroEnv import MacroEnv
from main import analyse_tracking_task, plot_detection_bar_chart, plot_means_lost_targets, plot_violin
from multi_seed_training_macro import MacroRandomSeedEnv
from multi_target_env import MultiTargetEnv

def evaluate_agent_macro(env=None, model=None, n_episodes=100, random_policy=False, n_targets=5, n_unknown_targets=100):
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
        obs, _ = env.reset()
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
    envRandom = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    rewardsRandom, detection_countRandom, detect_count3Random, exceedFOVRandom, last_envRandom, last_episode_logRandom, knownRandom, last_tasks_logRandom = evaluate_agent_macro(env=envRandom, model=None, n_episodes=n_episodes, random_policy=True, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    envPPO = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    ppo_model = PPO.load("agents/ppo_macro_trained", env=envPPO)
    rewardsPPO, detection_countPPO, detect_count3PPO, exceedFOVPPO, last_envPPO, last_episode_logPPO, knownPPO, last_tasks_logPPO = evaluate_agent_macro(env=envPPO, model=ppo_model, n_episodes=n_episodes, random_policy=False, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    envDQN = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
    dqn_model = DQN.load("agents/dqn_macro_trained", env=envDQN)
    rewardsDQN, detection_countDQN, detect_count3DQN, exceedFOVDQN, last_envDQN, last_episode_logDQN, knownDQN, last_tasks_logDQN = evaluate_agent_macro(env=envDQN, model=dqn_model, n_episodes=n_episodes, random_policy=False, n_targets=n_targets, n_unknown_targets=n_unknown_targets)

    # ****** Plot reward distributions ******
    reward_results = {
        "Random": rewardsRandom,
        "PPO": rewardsPPO,
        "DQN": rewardsDQN
    }
    plot_violin(reward_results, ylabel="Episode Reward")

    # ****** Plot detection distributions ******
    detection_results = {
        "Random": detection_countRandom,
        "PPO": detection_countPPO,
        "DQN": detection_countDQN
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
    plot_means_lost_targets(exceedFOVPPO, knownPPO, exceedFOVDQN, knownDQN, exceedFOVRandom, knownRandom)

    # ****** Plot time spent on task ******
    plot_task_log(last_tasks_logRandom, title="Random Policy - Action Timeline")
    plot_task_log(last_tasks_logPPO, title="PPO - Action Timeline")
    plot_task_log(last_tasks_logDQN, title="DQN - Action Timeline")



    