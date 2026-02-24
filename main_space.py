import numpy as np

import KalmanFilter
from multi_target_env import MultiTargetEnv
from track_sat_env import TrackSatEnv


def select_best_action(env):
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
    best_target_id = None
    best_ig = -float('inf')
    highest_risk = -float('inf')
    best_update = None

    # set up observation model
    observationFcn = KalmanFilter.extract_measurement

    sigma_theta = np.deg2rad(1.0 / 3600.0)
    sigma_r = 0.1

    R = np.diag([sigma_theta**2, sigma_theta**2, sigma_r**2])
    
    # propagate all targets
    currentDate = env.initialDate.shiftedBy(env.step_count)
    dt = 1.

    # Loop through all possible sensing actions: choose one target
    for index, (name, sat) in enumerate(env.satellites.items()):
        
        # 1. Propagate the target forward
        x_pred, P_pred = TrackSatEnv.propagate_sat_keplerian_force(sat["state"], sat["cov"], currentDate, dt, env.frame)

        # 2. Perform the hypothetical EKF update (sensing action = choose idx)
        x_upd, P_upd = MultiTargetEnv.ekf_update(sat["state"], sat["cov"], R, observationFcn)

        # 3. Compute information gain (KL divergence)/ complementary probability
        ig = MultiTargetEnv.compute_kl_divergence(x_pred, P_pred, x_upd, P_upd)
        Hk, Gk = observationFcn(x_upd)
        S = Hk @ P_upd @ Hk.T
        prob = MultiTargetEnv.compute_fov_prob_full(S, env.fov, env.fov)
        risk = 1-prob

        # 4. Keep the best
        if ig > best_ig:
            best_ig = ig
            best_target_id = index
            best_target_name = name
            best_update = {"x": x_upd, "P": P_upd}
        """ if risk > highest_risk:
            best_ig = ig
            highest_risk = risk
            best_target_id = index
            best_target_name = name
            best_update = {"x": x_upd, "P": P_upd} """

    return best_target_id, best_target_name, best_ig, best_update

def main():
    env = TrackSatEnv(d_state=6, max_steps=100)

    done = False

    while not done:
        #action, best_target_name, best_ig, best_update = select_best_action(env) 
        action = 0
        obs, reward, done, truncated, info = env.step(action)


if __name__ == "__main__":
    main()