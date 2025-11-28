from multi_target_env import MultiTargetEnv, compute_kl_divergence


def select_best_action(env, dt=None):
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
        x_upd, P_upd = MultiTargetEnv.ekf_update(x_pred, P_pred, env.R)

        # 3. Compute information gain (KL divergence)
        ig = compute_kl_divergence(x_pred, P_pred, x_upd, P_upd)

        # 4. Keep the best
        if ig > best_ig:
            best_ig = ig
            best_target_id = idx
            best_update = {"x": x_upd, "P": P_upd}

    return best_target_id, best_ig, best_update