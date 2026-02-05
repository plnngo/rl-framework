import numpy as np
from scipy.integrate import solve_ivp

@staticmethod
def ckf(Xo_ref, t_obs, obs, intfcn, H_fcn, inputs):
    """
    Least Squares Conventional Kalman Filter (CKF)

    Parameters
    ----------
    Xo_ref : (n,) ndarray
        Reference state at t=0
    t_obs : (L,) ndarray
        Observation times
    obs : (p, L) ndarray
        Observations
    intfcn : callable
        Dynamics + STM ODE function: f(t, state, inputs)
    H_fcn : callable
        Measurement model: H_fcn(Xref, inputs, t)
    inputs : object or dict
        Must contain Rk, Q, Po

    Returns
    -------
    Xk_mat : (n, L) ndarray
        Estimated states
    P_mat : (n, n, L) ndarray
        State covariance matrices
    resids : (p, L) ndarray
        Post-fit residuals
    """

    # Dimensions
    L = obs.shape[0]
    p = obs.shape[1]
    n = Xo_ref.size

    # Inputs
    Rk = inputs["Rk"]
    Q  = inputs["Q"]
    Po_bar = inputs["Po"]

    # Initialization
    xo_bar = np.zeros(n)
    xhat = xo_bar.copy()
    P = Po_bar.copy()
    Xref = Xo_ref.copy()

    resids = np.zeros((p, L))
    Xk_mat = np.zeros((n, L))
    Xref_mat = np.zeros((n, L))
    P_mat = np.zeros((n, n, L))
    #Pbk_mat = np.zeros((n, n, L))

    # STM initialization
    phi0 = np.eye(n)
    phi0_v = phi0.reshape(n * n)

    # ODE tolerances
    ode_tol = 1e-12

    # Kalman filter loop
    for k in range(L):

        t = t_obs[k]
        t_prior = 0.0 if k == 0 else t_obs[k - 1]
        delta_t = t - t_prior

        Yk = obs[k,:]

        # Save priors
        Xref_prior = Xref.copy()
        xhat_prior = xhat.copy()
        P_prior = P.copy()

        # -------------------------
        # Step B: Integrate Xref & STM
        # -------------------------
        int0 = np.hstack((Xref_prior, phi0_v))

        if t == t_prior:
            xout = int0
        else:
            sol = solve_ivp(
                fun=lambda tau, y: intfcn(tau, y, inputs),
                t_span=(t_prior, t),
                y0=int0,
                rtol=ode_tol,
                atol=ode_tol
            )
            xout = sol.y[:, -1]

        Xref = xout[:n]
        phi_v = xout[n:]
        phi = phi_v.reshape((n, n))

        # -------------------------
        # Step C: Time update
        # -------------------------
        Gamma = np.block([
            [(delta_t**2 / 2) * np.eye(2)],
            [delta_t * np.eye(2)]
        ])

        xbar = phi @ xhat_prior
        Pbar = phi @ P_prior @ phi.T + Gamma @ Q @ Gamma.T

        # -------------------------
        # Step D: Measurement update
        # -------------------------
        theta, r, Hk_til= H_fcn(Xref)
        Gk = np.array([theta, r])
        yk = Yk - Gk

        S = Hk_til @ Pbar @ Hk_til.T + Rk
        Kk = Pbar @ Hk_til.T @ np.linalg.inv(S)

        #Bk = yk - Hk_til @ xbar
        P_bk = S

        # -------------------------
        # Step E: State & covariance update
        # -------------------------
        xhat = xbar + Kk @ (yk - Hk_til @ xbar)

        I = np.eye(n)
        P = (I - Kk @ Hk_til) @ Pbar @ (I - Kk @ Hk_til).T + Kk @ Rk @ Kk.T

        # Post-fit residuals
        Xk = Xref + xhat
        resids[:,k] = yk - Hk_til @ xhat

        # Save outputs
        Xref_mat[:, k] = Xref
        Xk_mat[:, k] = Xk
        P_mat[:, :, k] = P
        #Pbk_mat[:, :, k] = P_bk

    return Xk_mat, P_mat, resids

def ckf_predict_update(Xo_ref, t_obs, tend, obs, intfcn, H_fcn, inputs):
    import numpy as np
    from scipy.integrate import solve_ivp

    # Dimensions
    n = Xo_ref.size
    p = obs.shape[1]

    # Inputs
    Rk = inputs["Rk"]
    Q  = inputs["Q"]
    Po = inputs["Po"]

    # Build dense time grid
    t_all = np.arange(tend)
    L_all = len(t_all)

    # Map measurements to times
    meas_dict = {t_obs[k]: obs[k, :] for k in range(len(t_obs))}

    # Initialization
    xhat = np.zeros(n)
    P = Po.copy()
    Xref = Xo_ref.copy()

    # Storage
    Xk_mat = np.zeros((n, L_all))
    P_mat = np.zeros((n, n, L_all))
    resids = np.full((p, L_all), np.nan)

    # STM initialization
    phi0 = np.eye(n).reshape(n * n)

    ode_tol = 1e-12

    # Main loop over *all* time steps
    for k in range(L_all):

        t = t_all[k]
        t_prior = t if k == 0 else t_all[k - 1]
        dt = t - t_prior

        # -------------------------
        # Propagation
        # -------------------------
        int0 = np.hstack((Xref, phi0))

        if dt > 0:
            sol = solve_ivp(
                fun=lambda tau, y: intfcn(tau, y, inputs),
                t_span=(t_prior, t),
                y0=int0,
                rtol=ode_tol,
                atol=ode_tol
            )
            xout = sol.y[:, -1]
        else:
            xout = int0

        Xref = xout[:n]
        phi = xout[n:].reshape((n, n))

        # Process noise (2D CV)
        Gamma = np.block([
            [(dt**2 / 2) * np.eye(2)],
            [dt * np.eye(2)]
        ])

        xbar = phi @ xhat
        Pbar = phi @ P @ phi.T + Gamma @ Q @ Gamma.T

        # -------------------------
        # Measurement update (if available)
        # -------------------------
        if t in meas_dict:
            Yk = meas_dict[t]

            theta, r, Hk_til = H_fcn(Xref)
            Gk = np.array([theta, r])
            yk = Yk - Gk

            S = Hk_til @ Pbar @ Hk_til.T + Rk
            Kk = Pbar @ Hk_til.T @ np.linalg.inv(S)

            xhat = xbar + Kk @ (yk - Hk_til @ xbar)
            I = np.eye(n)
            P = (I - Kk @ Hk_til) @ Pbar @ (I - Kk @ Hk_til).T + Kk @ Rk @ Kk.T

            resids[:, k] = yk - Hk_til @ xhat

        else:
            # No measurement → prediction only
            xhat = xbar
            P = Pbar

        # -------------------------
        # Save estimate
        # -------------------------
        Xk_mat[:, k] = Xref + xhat
        P_mat[:, :, k] = P

    return t_all, Xk_mat, P_mat, resids

def int_constant_velocity_stm(t, X, inputs):
    """
    Continuous-time constant velocity model with STM integration

    State ordering:
        X = [x, y, vx, vy, Phi(0,0), ..., Phi(3,3)]
    """

    n = 4

    # Continuous-time system matrix
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

    # Break out state and STM
    x = X[:n]
    phi = X[n:].reshape((n, n))

    # State derivative
    dx = A @ x

    # STM derivative
    dphi = A @ phi

    # Pack output
    dX = np.zeros_like(X)
    dX[:n] = dx
    dX[n:] = dphi.reshape(n * n)

    return dX

def int_constant_turn_stm_2D(t, X, inputs):
    """
    ODE for 2D constant‑turn + STM.
    inputs must contain 'omega' (turn rate).
    """
    n = 4
    x = X[:n].reshape(-1, 1)
    Phi = X[n:].reshape(n, n)

    omega = inputs["omega"]  # or inputs.omega if it's an object

    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, -omega],
        [0, 0, omega, 0]
    ])

    dxdt = A @ x
    dPhi_dt = A @ Phi

    dX = np.zeros(4 + 16)
    dX[:4] = dxdt.flatten()
    dX[4:] = dPhi_dt.flatten()

    return dX