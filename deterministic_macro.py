from MacroEnv import _sync_envs, unwrap_env
from multi_target_env import MultiTargetEnv, compute_fov_prob_single
import numpy as np

def select_best_action_pFOV(env, dt=1.0, boundary=4):

    # Make copy of tracking enviornment
    # evaluate the enviornment under a tracking action
    # if sum of pFOV of predicted obs is smaller than threshold, return 1
    # evaluate the environment under searching objective
    # if sum of pFOV of predicted obs is smaller than threshold, return 1
    # only if sum of pFOV of predicted obs within searching task is larger than threshold, return 0
    # predict tracking action
    #action_masks = track_env.action_masks()

    # Synchronise enviornements 
    real_search_env = unwrap_env(env.search_env)
    real_track_env  = unwrap_env(env.track_env)
    _sync_envs(env, real_search_env)
    _sync_envs(env, real_track_env)

    probSum = 0
    for tgt in real_search_env.targets.copy():

            idx = tgt['id']  # global index
            model = real_search_env.motion_model[idx]
            param = real_search_env.motion_params[idx]

            # retrieve predictions by propagating each known target
            predState, predCov = MultiTargetEnv.propagate_target_2D(tgt['x'], tgt['P'], tgt.get('Q', real_search_env.Q0), dt=real_search_env.dt, rng=real_search_env.rng, motion_model=model, motion_param=param)
            #trace_reward += np.trace(predCov)
            prob = compute_fov_prob_single(boundary, predState, predCov)
            probSum += prob

    return 0 if probSum == sum(real_search_env.known_mask) else 1


