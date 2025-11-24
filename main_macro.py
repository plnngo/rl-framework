import numpy as np
from stable_baselines3 import PPO
from tqdm import trange
from MacroEnv import MacroEnv
from multi_seed_training_macro import MacroRandomSeedEnv
from multi_target_env import MultiTargetEnv

def evaluate_agent_macro(seeds=None, model=None, n_episodes=100, random_policy=False, n_targets=5, n_unknown_targets=100):
    rewards = []
    detect_count3 = 0
    detection_count = []

    for ep in trange(n_episodes, desc="Evaluating",  leave=False):
        env = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        detections = env.init_n_targets

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            known_targets = sum(info["action_mask"]["micro_track"])
            if known_targets>detections:
                detect_count3 = detect_count3 + (known_targets - detections)
                detections = known_targets
        rewards.append(total_reward)
        detection_count.append(detections)
    return rewards, detection_count, detect_count3

if __name__ == "__main__":
    seeds = [42, 123, 321]
    n_episodes = 2
    n_targets = 5
    n_unknown_targets =100
    evaluate_agent_macro(seeds=seeds, model=None, n_episodes=n_episodes, random_policy=True, n_targets=n_targets, n_unknown_targets=n_unknown_targets)
    