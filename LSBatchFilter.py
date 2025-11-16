import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

from multi_target_env import MultiTargetEnv

# ----------------------
# Continuous-time dynamics and Jacobians
# ----------------------
def f_cv_cont(x):
    # state: [px,py,vx,vy]
    px, py, vx, vy = x
    return np.array([vx, vy, 0.0, 0.0])

def Fx_cv_cont(x):
    Fx = np.zeros((4,4))
    Fx[0,2] = 1.0
    Fx[1,3] = 1.0
    return Fx

def f_ct_no_heading_cont(x):
    # state: [px,py,vx,vy,omega]
    px, py, vx, vy, omega = x
    return np.array([vx, vy, -omega * vy, omega * vx, 0.0])

def Fx_ct_no_heading_cont(x):
    px, py, vx, vy, omega = x
    Fx = np.zeros((5,5))
    Fx[0,2] = 1.0
    Fx[1,3] = 1.0
    Fx[2,3] = -omega
    Fx[2,4] = -vy
    Fx[3,2] = omega
    Fx[3,4] = vx
    return Fx

# ----------------------
# Integrate state + STM
# ----------------------
def _dynamics_with_stm(t, z, f_dyn, Fx_dyn, n):
    x = z[:n]
    Phi_vec = z[n:]
    Phi = Phi_vec.reshape((n,n))
    dx = f_dyn(x)
    Fx = Fx_dyn(x)
    dPhi = Fx @ Phi
    return np.concatenate([dx, dPhi.reshape(-1)])

# ----------------------
# Batch estimator for one target (Batch LS)
# ----------------------
def batch_estimate_single_target(
    t_obs,
    y_obs,
    x0_ref,
    P0,
    R,
    model="CV",
    max_iters=20,
    tol=3e-5,
    rtol_ivp=1e-6,
    atol_ivp=1e-8,
    lambda_init=1e-3,
    lambda_factor=10.0
):
    """
    Batch estimator (Levenberg-Marquardt) for one target.
    """
    # Select dynamics
    if model.upper() == "CV":
        n = 4
        f_dyn = lambda x: f_cv_cont(x)
        Fx_dyn = lambda x: Fx_cv_cont(x)
    elif model.upper() == "CT":
        n = 5
        f_dyn = lambda x: f_ct_no_heading_cont(x)
        Fx_dyn = lambda x: Fx_ct_no_heading_cont(x)
    else:
        raise ValueError("Unknown model: 'CV' or 'CT'")

    L = len(t_obs)
    if L == 0:
        return None

    Xref_mat = np.zeros((n, L))
    P_mat = np.zeros((n, n, L))
    resids = np.zeros((2, L))

    t_obs = np.asarray(t_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    Po_bar = np.asarray(P0).copy()
    invPo_bar = np.linalg.inv(Po_bar)

    x0_ref = np.asarray(x0_ref).copy()
    xo_bar = np.zeros(n)

    lambda_lm = lambda_init

    for itr in range(max_iters):
        print(itr)
        Lambda = invPo_bar.copy()
        N = invPo_bar @ xo_bar
        Xref = x0_ref.copy()
        Phi = np.eye(n).reshape(-1)
        total_res_norm = 0.0

        for k in range(L):
            if k > 0:
                t_span = (t_obs[k-1], t_obs[k])
                z0 = np.concatenate([Xref, Phi])
                sol = solve_ivp(lambda tt, zz: _dynamics_with_stm(tt, zz, f_dyn, Fx_dyn, n),
                                t_span, z0, rtol=rtol_ivp, atol=atol_ivp)
                z_end = sol.y[:, -1]
                Xref = z_end[:n]
                Phi = z_end[n:]

            Phi_mat = Phi.reshape((n, n))
            Xref_mat[:, k] = Xref
            P_mat[:, :, k] = Phi_mat @ Po_bar @ Phi_mat.T

            if n == 4:
                theta, r, Htil = MultiTargetEnv.extract_measurement(Xref[:4])
            else:
                theta, r, Htil4 = MultiTargetEnv.extract_measurement(Xref[:4])
                Htil = np.zeros((2, 5))
                Htil[:, :4] = Htil4

            Gk = np.array([theta, r])
            yk = y_obs[k] - Gk
            resids[:, k] = yk
            total_res_norm += np.linalg.norm(yk)

            Hk = Htil @ Phi_mat
            Lambda += Hk.T @ np.linalg.inv(R) @ Hk
            N += Hk.T @ np.linalg.inv(R) @ yk

        # Apply LM damping
        Lambda_damped = Lambda + lambda_lm * np.eye(n)
        try:
            Lchol = np.linalg.cholesky(Lambda_damped)
            Lchol_inv = np.linalg.inv(Lchol)
            Po = Lchol_inv @ Lchol_inv.T
        except np.linalg.LinAlgError:
            Po = np.linalg.pinv(Lambda_damped)

        xo_hat = Po @ N
        xo_norm = np.linalg.norm(xo_hat)

        # Step clipping
        max_step = 1.0
        if xo_norm > max_step:
            xo_hat = xo_hat / xo_norm * max_step

        # Update reference
        x0_ref_new = x0_ref + xo_hat

        # Forward simulate new residuals to check if LM reduces cost
        # (Optional, can skip for speed)
        # if total_res_norm_new < total_res_norm: decrease lambda, else increase
        # Here we just adapt lambda heuristically
        if xo_norm < tol:
            print(f"Converged at iteration {itr}, xo_norm={xo_norm:.3e}")
            x0_ref = x0_ref_new
            break

        # Update LM damping heuristically
        lambda_lm *= 0.7 if xo_norm < 2*tol else lambda_factor

        x0_ref = x0_ref_new
        xo_bar = xo_bar - xo_hat

    # Final forward pass to compute consistent outputs
    Xref = x0_ref.copy()
    Phi = np.eye(n).reshape(-1)
    for k in range(L):
        if k > 0:
            t_span = (t_obs[k-1], t_obs[k])
            z0 = np.concatenate([Xref, Phi])
            sol = solve_ivp(lambda tt, zz: _dynamics_with_stm(tt, zz, f_dyn, Fx_dyn, n),
                            t_span, z0, rtol=rtol_ivp, atol=atol_ivp)
            z_end = sol.y[:, -1]
            Xref = z_end[:n]
            Phi = z_end[n:]
        Phi_mat = Phi.reshape((n,n))
        Xref_mat[:, k] = Xref
        P_mat[:, :, k] = Phi_mat @ Po_bar @ Phi_mat.T
        if n == 4:
            theta, r, _ = MultiTargetEnv.extract_measurement(Xref[:4])
        else:
            theta, r, _ = MultiTargetEnv.extract_measurement(Xref[:4])
        resids[:, k] = y_obs[k] - np.array([theta, r])

    return {
        "Xref_mat": Xref_mat,
        "P_mat": P_mat,
        "resids": resids,
        "x0_ref": x0_ref,
        "P0": Po_bar,
        "model": model
    }
# Residuals for CV: params = [px0, py0, vx0, vy0]
def residuals_cv(params, t_obs, y_obs, R_sqrt_inv=None):
    x0 = np.asarray(params)
    # Integrate CV dynamics
    states = integrate_and_sample(f_cv_cont, x0, t_obs)  # shape (L,4)
    thetas = []
    rs = []
    for s in states:
        th, rr, _ = MultiTargetEnv.extract_measurement(s)
        thetas.append(th)
        rs.append(rr)
    thetas = np.array(thetas)
    rs = np.array(rs)

    # angle residuals wrapped
    th_res = angle_diff(thetas, y_obs[:,0])
    r_res = rs - y_obs[:,1]

    # Interleave [th0, r0, th1, r1, ...]
    res = np.empty(2 * len(t_obs))
    res[0::2] = th_res
    res[1::2] = r_res

    if R_sqrt_inv is not None:
        # whiten residuals if provided measurement covariance sqrt-inverse
        # R_sqrt_inv is 2x2 such that R_sqrt_inv @ residual_2 = whitened residual
        res_whitened = np.empty_like(res)
        for i in range(len(t_obs)):
            r2 = res[2*i:2*i+2]
            w2 = R_sqrt_inv @ r2
            res_whitened[2*i:2*i+2] = w2
        return res_whitened
    return res
# ------------------------
# Reuse the continuous-time dynamics from before
# CV dynamics (state: [px,py,vx,vy], n=4)
def f_cv_cont(x):
    px, py, vx, vy = x
    return np.array([vx, vy, 0.0, 0.0])

# CT (no-heading) dynamics (state: [px,py,vx,vy,omega], n=5)
def f_ct_no_heading_cont(x):
    px, py, vx, vy, omega = x
    return np.array([vx, vy, -omega * vy, omega * vx, 0.0])

# ------------------------
# helper: measurement prediction for a state vector (4D px,py,vx,vy)
def predict_measurement_from_state(x4):
    theta, r, _ = MultiTargetEnv.extract_measurement(x4)
    return np.array([theta, r])

# ------------------------
# helper: angle residual wrap to [-pi, pi]
def angle_diff(a, b):
    d = a - b
    return np.arctan2(np.sin(d), np.cos(d))

# ------------------------
# integrate dynamics once for a candidate initial state and sample at t_obs
def integrate_and_sample(f_dyn, x0, t_obs, rtol=1e-9, atol=1e-9):
    # integrate from t0 = t_obs[0] to t_obs[-1] and evaluate at t_obs
    t0 = float(t_obs[0])
    tspan = (t0, float(t_obs[-1]))
    # shift times to absolute by integrating from t0; inside solver we return states at absolute times
    def fun(tt, zz):
        return f_dyn(zz)

    # Use solve_ivp with t_eval = t_obs for direct sampling
    sol = solve_ivp(fun, tspan, x0, t_eval=np.asarray(t_obs), rtol=rtol, atol=atol)
    if not sol.success:
        # fallback: try dense output and sample
        sol = solve_ivp(fun, tspan, x0, dense_output=True, rtol=rtol, atol=atol)
        states = sol.sol(t_obs).T
    else:
        states = sol.y.T  # shape (len(t_obs), n)
    return states
# ------------------------
# Residuals for CT: params = [px0, py0, vx0, vy0, omega]
def residuals_ct(params, t_obs, y_obs, R_sqrt_inv=None):
    x0 = np.asarray(params)
    states = integrate_and_sample(f_ct_no_heading_cont, x0, t_obs)  # shape (L,5)
    thetas = []
    rs = []
    for s in states:
        # pass only first 4 entries to measurement function
        th, rr, _ = MultiTargetEnv.extract_measurement(s[:4])
        thetas.append(th)
        rs.append(rr)
    thetas = np.array(thetas)
    rs = np.array(rs)

    th_res = angle_diff(thetas, y_obs[:,0])
    r_res = rs - y_obs[:,1]

    res = np.empty(2 * len(t_obs))
    res[0::2] = th_res
    res[1::2] = r_res

    if R_sqrt_inv is not None:
        res_whitened = np.empty_like(res)
        for i in range(len(t_obs)):
            r2 = res[2*i:2*i+2]
            w2 = R_sqrt_inv @ r2
            res_whitened[2*i:2*i+2] = w2
        return res_whitened
    return res

# fit functions that call least_squares
def fit_initial_state_cv(t_obs, y_obs, R=None, x0_guess=None, bounds=None, verbose=0):
    """
    Fit initial CV state [px0,py0,vx0,vy0] from measurements y_obs at times t_obs.
    """
    if x0_guess is None:
        # rough guess from first measurement (range+bearing -> px,py) and zero velocity
        th0 = y_obs[0,0]; r0 = y_obs[0,1]
        px0 = r0 * np.cos(th0); py0 = r0 * np.sin(th0)
        x0_guess = np.array([px0, py0, 0.0, 0.0])
    if R is not None:
        # compute whitening: R = cov, get inv sqrt via eigen or cholesky
        try:
            L = np.linalg.cholesky(R)
            R_sqrt_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            # fallback using sqrt of diagonal
            R_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(R)))
    else:
        R_sqrt_inv = None

    res = least_squares(residuals_cv, x0_guess, args=(t_obs, y_obs, R_sqrt_inv), bounds=bounds or (-np.inf, np.inf), verbose=verbose)
    return res.x, res

def fit_initial_state_ct(t_obs, y_obs, R=None, x0_guess=None, bounds=None, verbose=0):
    """
    Fit initial CT state [px0,py0,vx0,vy0,omega] from measurements.
    """
    if x0_guess is None:
        th0 = y_obs[0,0]; r0 = y_obs[0,1]
        px0 = r0 * np.cos(th0); py0 = r0 * np.sin(th0)
        x0_guess = np.array([px0, py0, 0.5, 0.0, 0.0])  # moderate speed guess, omega=0
    if R is not None:
        try:
            L = np.linalg.cholesky(R)
            R_sqrt_inv = np.linalg.inv(L)
        except np.linalg.LinAlgError:
            R_sqrt_inv = np.diag(1.0 / np.sqrt(np.diag(R)))
    else:
        R_sqrt_inv = None

    # sensible bounds: keep omega in a reasonable range to help identifiability
    if bounds is None:
        # px,py unbounded, vx,vy within [-50*space, 50*space] (you can tune), omega in [-2*pi,2*pi]
        lower = [-np.inf, -np.inf, -np.inf, -np.inf, -2*np.pi]
        upper = [ np.inf,  np.inf,  np.inf,  np.inf,  2*np.pi]
        bounds = (lower, upper)

    res = least_squares(residuals_ct, x0_guess, args=(t_obs, y_obs, R_sqrt_inv), bounds=bounds, verbose=verbose)
    return res.x, res

# ----------------------
# Wrapper to estimate all targets from tracks
# ----------------------
def estimate_all_targets_from_tracks(tracks, R=None, **batch_kwargs):
    """
    tracks: output of extract_tracks_from_log
    R: measurement noise matrix
    """

    if R is None:
        sigma_theta = np.deg2rad(1.0)
        sigma_range = 0.1
        R = np.diag([sigma_theta**2, sigma_range**2])

    estimates = {}

    for tgt_id, track in tracks.items():

        # --- Extract data from track ---
        timesteps = [obs["t"] for obs in track]
        states = [obs["state"] for obs in track]

        if len(timesteps) == 0:
            continue

        # --- Convert ground-truth states → measurements ---
        y_obs = []
        for x in states:
            th, rr, _ = MultiTargetEnv.extract_measurement(x)
            y_obs.append([th, rr])
        y_obs = np.array(y_obs)

        # --------------------------------------------
        # 1) Fit initial state under CV model
        # --------------------------------------------
        x0_cv, res_cv = fit_initial_state_cv(timesteps, y_obs, R=R, verbose=0)
        cost_cv = 2 * res_cv.cost

        # --------------------------------------------
        # 2) Fit initial state under CT model
        # --------------------------------------------
        x0_ct, res_ct = fit_initial_state_ct(timesteps, y_obs, R=R, verbose=0)
        cost_ct = 2 * res_ct.cost

        # --------------------------------------------
        # 3) Pick best model
        # --------------------------------------------
        if cost_cv < cost_ct:
            chosen_model = "CV"
            x0_ref = x0_cv
            best_cost = cost_cv
        else:
            chosen_model = "CT"
            x0_ref = x0_ct
            best_cost = cost_ct

        print(f"Target {tgt_id}: chosen {chosen_model} with cost {best_cost}")

        # --------------------------------------------
        # 4) Build matching P0 (dimension = len(x0_ref))
        # --------------------------------------------
        dim = len(x0_ref)
        P0 = np.eye(dim) * 1e3    # adjustable

        # --------------------------------------------
        # 5) Run batch estimator with chosen model
        # --------------------------------------------
        out = batch_estimate_single_target(
            timesteps,
            y_obs,
            x0_ref,
            P0,
            R,
            model=chosen_model,
            **batch_kwargs
        )

        estimates[tgt_id] = out

    return estimates
