from multi_target_env import MultiTargetEnv, compute_fov_prob_single
import numpy as np

def select_best_action_pFOV(env, dt=1.0, boundary=4):

    # Make copy of tracking enviornment
    search_env = env.search_env.env
    # evaluate the enviornment under a tracking action
    # if sum of pFOV of predicted obs is smaller than threshold, return 1
    # evaluate the environment under searching objective
    # if sum of pFOV of predicted obs is smaller than threshold, return 1
    # only if sum of pFOV of predicted obs within searching task is larger than threshold, return 0
    # predict tracking action
    #action_masks = track_env.action_masks()
    probSum = 0
    for tgt in search_env.targets.copy():

            idx = tgt['id']  # global index
            model = search_env.motion_model[idx]
            param = search_env.motion_params[idx]

            # retrieve predictions by propagating each known target
            predState, predCov = MultiTargetEnv.propagate_target_2D(tgt['x'], tgt['P'], tgt.get('Q', search_env.Q0), dt=search_env.dt, rng=search_env.rng, motion_model=model, motion_param=param)
            #trace_reward += np.trace(predCov)
            prob = compute_fov_prob_single(boundary, predState, predCov)
            probSum += prob

    return 0 if probSum == sum(search_env.known_mask) else 1


