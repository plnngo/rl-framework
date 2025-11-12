import numpy as np

import MCTSenv
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
        best_action = mcts.choose() # runs search, returns encoded action
        
        # Decode hierarchical action for display
        macro, micro_s, micro_t = env.decode_action(best_action)
        print(f"Chosen Action: Macro={macro}, Micro_S={micro_s}, Micro_T={micro_t}")

        # Optional: visualize MCTS tree
        if visualize:
            src = mcts.visualize_mcts_tree(mcts.root, max_depth=5)
            display(src)  # Shows the tree inline without needing Graphviz executable

        # Apply action to environment
        next_obs, reward, done, truncated, info = env.step(best_action)

        print(f"Reward: {reward}")
        print(f"Done: {done}, Info: {info}")

        # Update MCTS root for next iteration (re-rooting)
        mcts.root = mcts.re_root(best_action)

        obs = next_obs
        step += 1

    print("Episode finished!")
    return obs

def main():
    # Create environment
    env = MultiTargetEnv(n_targets=5, n_unknown_targets=100, seed=42, mode="search")
    mcts = MCTSenv.MCTS(env=env, rollout_depth=5, gamma=0.95)

    final_state = run_episode(env, mcts, max_steps=10, visualize=True)


if __name__ == "__main__":
    main()