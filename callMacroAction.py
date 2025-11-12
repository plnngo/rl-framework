# -------------------------
# TEST MACRO ENVIRONMENT
# -------------------------
from stable_baselines3 import PPO
from MacroEnv import MacroEnv


if __name__ == "__main__":
    # 1. Instantiate macro environment with dummy micro agents
    macro_env = MacroEnv(
        n_targets=5,
        n_unknown_targets=3,
        fov_size=2.0,
        search_agent=PPO.load("agents/ppo2_sensor_tasking_search_gamma09148472308668416_steps60000", env=env),
        track_agent=DummyMicroAgent(n_actions=5)     # number of known targets
    )

    # 2. Reset macro environment
    obs, info = macro_env.reset()
    print("Initial macro observation shape:", obs.shape)

    # 3. First macro step: search (macro_action=0)
    next_obs, reward, done, truncated, info = macro_env.step(macro_action=0)
    print("\n--- Macro Step 1: Search ---")
    print("Macro reward:", reward)
    print("Next observation shape:", next_obs.shape)

    # 4. Second macro step: track (macro_action=1)
    next_obs, reward, done, truncated, info = macro_env.step(macro_action=1)
    print("\n--- Macro Step 2: Track ---")
    print("Macro reward:", reward)
    print("Next observation shape:", next_obs.shape)