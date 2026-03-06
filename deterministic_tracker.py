import numpy as np
from multi_target_env import MultiTargetEnv, compute_fov_prob_single


def select_best_action_pFOV(env, dt=None):
    """
    Returns the target ID that yields the highest information gain (KL divergence)
    when selecting it as the next action.

    Parameters
    ----------
    env : object
        Environment-like object containing targets, models, parameters, rng, etc.
    R : ndarray
        Measurement noise covariance used in the EKF update.
    dt : float or None
        Optional override of env.dt.

    Returns
    -------
    best_target_id : int
        ID of the target with highest information gain.
    best_ig : float
        The corresponding information gain value.
    best_update : dict
        Contains the updated state and covariance (x, P) for the best action.
    """
    dt = dt if dt is not None else env.dt

    best_target_id = None
    best_ig = -float('inf')
    highest_prob = -float('inf')
    best_update = None

    obsFunc = MultiTargetEnv.extract_measurement_XY

    # Pre-compute propagations for all targets
    propagated = {}
    for tgt in env.targets:
        idx = tgt['id']
        model = env.motion_model[idx]
        param = env.motion_params[idx]
        x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
            tgt['x'], tgt['P'], tgt.get('Q', env.Q0),
            dt=dt,
            rng=env.rng,
            motion_model=model,
            motion_param=param
        )
        propagated[idx] = (x_pred, P_pred)

    # Pre-sum all predicted probs
    total_pred_prob = sum(
        compute_fov_prob_single(env.fov_size, x_p, P_p)
        for x_p, P_p in propagated.values()
    )
    # Main loop: choose one target as the sensing action 
    for tgt in env.targets:
        idx = tgt['id']
        x_pred, P_pred = propagated[idx]
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R, obsFunc)

        ig = MultiTargetEnv.compute_kl_divergence(x_pred, P_pred, x_upd, P_upd)

        # Swap out this target's predicted prob for its updated prob
        prob_this_pred = compute_fov_prob_single(env.fov_size, x_pred, P_pred)
        prob_this_upd  = compute_fov_prob_single(env.fov_size, x_upd,  P_upd)
        prob = total_pred_prob - prob_this_pred + prob_this_upd

        # Keep the best
        """ if ig > best_ig:
            best_ig = ig
            best_target_id = idx
            best_update = {"x": x_upd, "P": P_upd} """
        if prob > highest_prob:
            best_ig = ig
            highest_prob = prob
            best_target_id = idx
            best_update = {"x": x_upd, "P": P_upd}

    return best_target_id, best_ig, best_update


def select_best_action_IG(env, dt=None):
    """
    Returns the target ID that yields the highest information gain (KL divergence)
    when selecting it as the next action.

    Parameters
    ----------
    env : object
        Environment-like object containing targets, models, parameters, rng, etc.
    R : ndarray
        Measurement noise covariance used in the EKF update.
    dt : float or None
        Optional override of env.dt.

    Returns
    -------
    best_target_id : int
        ID of the target with highest information gain.
    best_ig : float
        The corresponding information gain value.
    best_update : dict
        Contains the updated state and covariance (x, P) for the best action.
    """
    dt = dt if dt is not None else env.dt

    best_target_id = None
    best_ig = -float('inf')
    best_update = None

    obsFunc = MultiTargetEnv.extract_measurement_XY

    # Loop through all possible sensing actions: choose one target
    for tgt in env.targets:
        idx = tgt['id']
        model = env.motion_model[idx]
        param = env.motion_params[idx]

        # 1. Propagate the target forward
        x_pred, P_pred = MultiTargetEnv.propagate_target_2D(
            tgt['x'], tgt['P'], tgt.get('Q', env.Q0),
            dt=dt,
            rng=env.rng,
            motion_model=model,
            motion_param=param
        )

        # 2. Perform the hypothetical EKF update (sensing action = choose idx)
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R, MultiTargetEnv.extract_measurement_bearingRange)

        # 3. Compute information gain (KL divergence)/ complementary probability
        ig = MultiTargetEnv.compute_kl_divergence(x_pred, P_pred, x_upd, P_upd)

        if ig > best_ig:
            best_ig = ig
            best_target_id = idx
            best_update = {"x": x_upd, "P": P_upd}

    return best_target_id, best_ig, best_update

def select_best_macro_action(env):
    obs = env._get_obs()
    known_obs = obs[env.known_mask]
    return 1 if np.any(known_obs <= env.threshold_fov) else 0