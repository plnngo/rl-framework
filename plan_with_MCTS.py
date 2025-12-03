import numpy as np

import MCTSenv
from MacroEnv import _sync_envs
from multi_seed_training_macro import MacroRandomSeedEnv
from multi_target_env import MultiTargetEnv
from IPython.display import display

def run_episode(env, mcts, max_steps=20, visualize=False):
    """
    Run one MCTS-driven planning episode.
    
    env: your environment instance
    mcts: initialized MCTS object
    max_steps: maximum number of steps per episode
    visualize: whether to generate a Graphviz tree at each step
    """
    obs = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        print(f"\n=== Step {step} ===")

        # Plan with MCTS
        print("Running MCTS planning...")
        mcts.reset_tree(obs)           # resets or sets root to current observation
        best_child_node= mcts.choose() # runs search, returns encoded action
        
        # Decode hierarchical action for display
        #macro, micro_s, micro_t = env.decode_action(best_action)
        macro = best_child_node.macro_action
        if macro == 0:
            micro_s = best_child_node.micro_action
            micro_t = None
        else:
            micro_s = None
            micro_t = best_child_node.micro_action
        print(f"Chosen Action: Macro={macro}, Micro_S={micro_s}, Micro_T={micro_t}")

        # Apply action to environment
        if macro == 0:
            next_obs, rewards, done, truncated, info = env.search_env.env.step(micro_s) 
            _sync_envs(env.search_env.env, env.base_env)
            _sync_envs(env.search_env.env, env.track_env.env)
        else:
            next_obs, rewards, done, truncated, info = env.track_env.env.step(micro_t) 
            _sync_envs(env.track_env.env, env.base_env)
            _sync_envs(env.track_env.env, env.search_env.env)

        print(f"Reward: {rewards}")
        #print(f"Done: {done}, Info: {info}")

        # Update MCTS root for next iteration (re-rooting)
        mcts.root = mcts.re_root(best_child_node)

        obs = next_obs
        step += 1

    print("Episode finished!")
    return obs

def main():
    # Create environment
    seeds = [42, 123, 321]
    n_targets = 5
    n_unknown_targets =100
    env = MacroRandomSeedEnv._make_env(n_targets,n_unknown_targets,seed=int(np.random.choice(seeds)))

    mcts = MCTSenv.MCTS(env=env, rollout_depth=5, gamma=0.95)

    final_state = run_episode(env, mcts, max_steps=10, visualize=False)


if __name__ == "__main__":
    main()