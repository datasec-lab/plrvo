import numpy as np
from scipy.special import gammaln
from itertools import product
from multiprocessing import Pool
 
# ============================================================
# USER CONFIGURATION
# ============================================================
 
MODEL_SIZE = 109484000       # trainable parameters
DATASET_SIZE = 67349        # N
DELTA = 1e-5
 
THETAS = [0.01]
CLIPS = [0.1]
EPOCHS_LIST = [3]
BATCHES = [1024]
EPSILONS = [1]
 
LAMBDA_MAX = 20
 
K_MIN = 1.0 + 1e-6
K_MAX_INIT = 10.0
K_HARD_CAP = 60_000_000
 
K_REL_TOL = 1e-3
K_MAX_BISECTION_ITERS = 30
 
 
# ============================================================
# PLRV-O SETTINGS
# ============================================================
 
PLRVO_LAMBDA_INITIAL = 20
PLRVO_LAMBDA_HARD_CAP = 20
PLRVO_LAMBDA_BOUNDARY_MARGIN = 0
 
PLRVO_LAMBDA_CALIBRATE = 20
 
PLRVO_K_MIN = 1.0 + 1e-6
PLRVO_K_MAX_INIT = 10.0
PLRVO_K_HARD_CAP = 60000000
 
PLRVO_K_REL_TOL = 1e-3
PLRVO_K_MAX_BISECTION_ITERS = 30
 
 
# ============================================================
# MAJORIZATION VECTOR
# ============================================================
 
def _majorization_chunk(first, last):
    """
    Exact l2-majorization coordinates:
 
        x_i = sqrt(i) - sqrt(i-1)
 
    for i = first+1,...,last
    """
 
    idx = np.arange(
        first + 1,
        last + 1,
        dtype=np.float64,
    )
 
    return np.sqrt(idx) - np.sqrt(idx - 1.0)
 
 
# ============================================================
# PLRV-O PURE LOG MOMENTS
# ============================================================
 
def _plrvo_pure_log_moments(
    k_shape,
    gamma,
    n_params,
    lambda_max,
):
    """
    Compute log E[exp(eta * L)]
    for eta = 0,...,lambda_max+1.
 
    Exact streaming implementation.
 
    All n_params coordinates are included.
    """
 
    eta = np.arange(
        2,
        lambda_max + 2,
        dtype=np.float64,
    )
 
    total = np.zeros_like(eta)
 
    number_of_orders = len(eta)
 
    chunk = max(
        32,
        min(
            n_params,
            2048,
            2_000_000 // max(
                number_of_orders,
                1,
            ),
        ),
    )
 
    for first in range(
        0,
        n_params,
        chunk,
    ):
 
        last = min(
            first + chunk,
            n_params,
        )
 
        x_chunk = _majorization_chunk(
            first,
            last,
        )[:, None]
 
        t1 = (
            gamma
            * x_chunk
            * (eta[None, :] - 1.0)
        )
 
        invalid = t1 >= 1.0
 
        t1_safe = np.minimum(
            t1,
            1.0 - 1e-14,
        )
 
        t2 = (
            gamma
            * x_chunk
            * eta[None, :]
        )
 
        l1 = (
            np.log(
                eta
                / (2.0 * eta - 1.0)
            )[None, :]
            - k_shape
            * np.log1p(-t1_safe)
        )
 
        l2 = (
            np.log(
                (eta - 1.0)
                / (2.0 * eta - 1.0)
            )[None, :]
            - k_shape
            * np.log1p(t2)
        )
 
        m = np.maximum(
            l1,
            l2,
        )
 
        log_g = (
            m
            + np.log(
                np.exp(l1 - m)
                + np.exp(l2 - m)
            )
        )
 
        log_g[invalid] = np.inf
 
        total += np.sum(
            log_g,
            axis=0,
        )
 
    result = np.zeros(
        lambda_max + 2,
        dtype=np.float64,
    )
 
    result[2:] = total
 
    return result
 
 
# ============================================================
# BINOMIAL WEIGHTS
# ============================================================
 
def _log_binomial_weights(
    m,
    q,
):
 
    eta = np.arange(
        m + 1,
        dtype=np.float64,
    )
 
    values = (
        gammaln(m + 1.0)
        - gammaln(eta + 1.0)
        - gammaln(m - eta + 1.0)
        + (m - eta)
        * np.log1p(-q)
        + eta
        * np.log(q)
    )
 
    mx = np.max(values)
 
    return values - (
        mx
        + np.log(
            np.sum(
                np.exp(
                    values - mx
                )
            )
        )
    )
 
 
# ============================================================
# EPSILON CURVE
# ============================================================
 
def _eps_curve_plrvo(
    pure_log_moments,
    q,
    T,
    delta,
    lambda_max,
):
 
    eps_curve = np.empty(
        lambda_max,
        dtype=np.float64,
    )
 
    for lam in range(
        1,
        lambda_max + 1,
    ):
 
        m = lam + 1
 
        selected = pure_log_moments[
            :m + 1
        ]
 
        if np.any(
            np.isposinf(selected)
        ):
 
            alpha = np.inf
 
        else:
 
            w = _log_binomial_weights(
                m,
                q,
            )
 
            mx = np.max(
                w + selected
            )
 
            alpha = max(
                float(
                    mx
                    + np.log(
                        np.sum(
                            np.exp(
                                w
                                + selected
                                - mx
                            )
                        )
                    )
                ),
                0.0,
            )
 
        eps_curve[
            lam - 1
        ] = (
            T * alpha / lam
            + np.log(
                lam
                / (lam + 1.0)
            )
            - (
                np.log(delta)
                + np.log(lam + 1.0)
            ) / lam
        )
 
    return eps_curve
 
 
# ============================================================
# FULL PLRV-O EPSILON
# ============================================================
 
def plrvo_epsilon(
    k_shape,
    gamma,
    q,
    T,
    delta,
    n_params,
    lambda_initial=PLRVO_LAMBDA_INITIAL,
    lambda_hard_cap=PLRVO_LAMBDA_HARD_CAP,
    margin=PLRVO_LAMBDA_BOUNDARY_MARGIN,
):
    """
    Calculate epsilon for fixed:
 
        k
        gamma
        q
        T
        delta
        n_params
 
    """
 
    lambda_max = lambda_initial
 
    boundary_reached = False
 
    while True:
 
        pure = _plrvo_pure_log_moments(
            k_shape,
            gamma,
            n_params,
            lambda_max,
        )
 
        eps_curve = _eps_curve_plrvo(
            pure,
            q,
            T,
            delta,
            lambda_max,
        )
 
        idx = int(
            np.argmin(
                eps_curve
            )
        )
 
        lam_star = idx + 1
 
        if lam_star < (
            lambda_max - margin
        ):
            break
 
        if lambda_max >= (
            lambda_hard_cap
        ):
 
            boundary_reached = True
            break
 
        lambda_max = min(
            lambda_hard_cap,
            2 * lambda_max,
        )
 
    return (
        float(
            eps_curve[idx]
        ),
        lam_star,
        lambda_max,
        boundary_reached,
    )
 
 
# ============================================================
# COARSE EPSILON
# ============================================================
 
def plrvo_epsilon_coarse(
    k_shape,
    gamma,
    q,
    T,
    delta,
    n_params,
    lambda_max=PLRVO_LAMBDA_CALIBRATE,
):
    """
    Cheap fixed-lambda epsilon calculation.
 
    Used during k search.
    """
 
    pure = _plrvo_pure_log_moments(
        k_shape,
        gamma,
        n_params,
        lambda_max,
    )
 
    eps_curve = _eps_curve_plrvo(
        pure,
        q,
        T,
        delta,
        lambda_max,
    )
 
    return float(
        np.min(
            eps_curve
        )
    )
 
 
# ============================================================
# SOLVE FOR K
# ============================================================
 
def find_k_given_epsilon(
    gamma,
    q,
    T,
    eps_target,
    delta,
    n_params,
    k_min=PLRVO_K_MIN,
    k_max_init=PLRVO_K_MAX_INIT,
    k_hard_cap=PLRVO_K_HARD_CAP,
    rel_tol=PLRVO_K_REL_TOL,
    max_iters=PLRVO_K_MAX_BISECTION_ITERS,
):
    """
    Find k such that:
 
        epsilon(k) ~= eps_target
 
    """
 
    # --------------------------------------------------------
    # Check minimum k
    # --------------------------------------------------------
 
    eps_lo = plrvo_epsilon_coarse(
        k_min,
        gamma,
        q,
        T,
        delta,
        n_params,
    )
    
    print(
       f"[k_min check] "
       f"k={k_min:.12g}, "
       f"gamma={gamma:.12g}, "
       f"q={q:.12g}, "
       f"T={T}, "
       f"target_epsilon={eps_target:.12g}, "
       f"epsilon_at_kmin={eps_lo:.12g}",
       flush=True,
    )
 
    if eps_lo > eps_target:
 
        eps_lo_exact, lam_lo, lambda_checked_lo, boundary_lo = (
            plrvo_epsilon(
                k_min,
                gamma,
                q,
                T,
                delta,
                n_params,
            )
        )
 
        return {
            "k": None,
            "eps_achieved": eps_lo_exact,
            "lambda_star": lam_lo,
            "lambda_checked": lambda_checked_lo,
            "boundary_reached": boundary_lo,
            "feasible": False,
        }
 
    # --------------------------------------------------------
    # Find upper bracket
    # --------------------------------------------------------
 
    k_hi = k_max_init
 
    eps_hi = plrvo_epsilon_coarse(
        k_hi,
        gamma,
        q,
        T,
        delta,
        n_params,
    )
 
    tries = 0
 
    while (
        eps_hi < eps_target
        and k_hi < k_hard_cap
        and tries < 80
    ):
 
        k_hi *= 2.0
 
        eps_hi = plrvo_epsilon_coarse(
            k_hi,
            gamma,
            q,
            T,
            delta,
            n_params,
        )
 
        tries += 1
 
    if eps_hi < eps_target:
 
        raise RuntimeError(
            "Could not find k_hi to reach "
            f"eps_target={eps_target} "
            f"for gamma={gamma}, "
            f"q={q}, "
            f"T={T}"
        )
 
    # --------------------------------------------------------
    # Bisection
    # --------------------------------------------------------
 
    lo = k_min
    hi = k_hi
 
    it = 0
 
    while it < max_iters:
 
        mid = 0.5 * (
            lo + hi
        )
 
        eps_mid = plrvo_epsilon_coarse(
            mid,
            gamma,
            q,
            T,
            delta,
            n_params,
        )
 
        if eps_mid >= eps_target:
 
            hi = mid
 
        else:
 
            lo = mid
 
        if (
            hi - lo
            <= rel_tol * hi
        ):
 
            break
 
        it += 1
 
    k = 0.5 * (
        lo + hi
    )
 
    # --------------------------------------------------------
    # Final exact verification
    # --------------------------------------------------------
 
    (
        eps_achieved,
        lambda_star,
        lambda_checked,
        boundary_reached,
    ) = plrvo_epsilon(
        k,
        gamma,
        q,
        T,
        delta,
        n_params,
    )
 
    return {
        "k": k,
        "eps_achieved": eps_achieved,
        "lambda_star": lambda_star,
        "lambda_checked": lambda_checked,
        "boundary_reached": boundary_reached,
        "feasible": True,
    }
 
 
# ============================================================
# ADDED FOR EQUICONFIG (existing functions above are unchanged)
# ============================================================
 
def find_gamma_given_k(
    k_shape,
    q,
    T,
    eps_target,
    delta,
    n_params,
    gamma_lo=1e-9,
    gamma_hi_init=1e-3,
    gamma_hard_cap=0.99,
    rel_tol=1e-3,
    max_iters=80,
    lambda_cal=PLRVO_LAMBDA_CALIBRATE,
):
    """
    Paper convention (Sec. 3.1, App. A.3/C/E): the mixture SHAPE k (> 2) is
    fixed and the scale gamma = C*theta is calibrated so that the accountant
    meets eps_target.  Normalized RMS noise (Eq. 12):
 
        s_P = sqrt(2) / (gamma * sqrt((k-1)(k-2)))
 
    Privacy depends only on (k, gamma), so the ridge is independent of C
    (Prop. 5); the physical theta for clipping bound C is theta = gamma / C.
 
    epsilon increases with gamma, so we bracket and bisect in log-space and
    return the largest gamma whose VERIFIED epsilon is <= eps_target.
    """
    if k_shape <= 2.0:
        raise ValueError("PLRV-O requires k > 2 for finite variance.")
 
    def coarse(g):
        return plrvo_epsilon_coarse(
            k_shape, g, q, T, delta, n_params, lambda_max=lambda_cal
        )
 
    def infeasible(reason):
        return {"gamma": None, "eps_achieved": float("inf"),
                "lambda_star": None, "lambda_checked": None,
                "boundary_reached": False, "feasible": False,
                "reason": reason}
 
    lo = gamma_lo
    eps_lo = coarse(lo)
    if (not np.isfinite(eps_lo)) or eps_lo > eps_target:
        return infeasible("eps(gamma_lo) above target")
 
    hi = gamma_hi_init
    eps_hi = coarse(hi)
    while eps_hi <= eps_target and hi < gamma_hard_cap:
        lo = hi
        hi = min(2.0 * hi, gamma_hard_cap)
        eps_hi = coarse(hi)
    if eps_hi <= eps_target:
        return infeasible("target not reachable below gamma_hard_cap")
 
    for _ in range(max_iters):
        if hi / lo - 1.0 <= rel_tol:
            break
        mid = float(np.sqrt(lo * hi))
        if coarse(mid) <= eps_target:
            lo = mid
        else:
            hi = mid
 
    gamma = lo
    for _ in range(40):
        eps_ach, lam_star, lam_checked, boundary = plrvo_epsilon(
            k_shape, gamma, q, T, delta, n_params
        )
        if np.isfinite(eps_ach) and eps_ach <= eps_target:
            return {"gamma": float(gamma), "eps_achieved": float(eps_ach),
                    "lambda_star": int(lam_star),
                    "lambda_checked": int(lam_checked),
                    "boundary_reached": bool(boundary), "feasible": True,
                    "reason": ""}
        gamma *= 0.995
 
    return infeasible("could not verify eps <= target")
 
 
# ============================================================
# MAIN
# ============================================================
 
def run_one_config(config):
    theta, clip, epochs, batch, target_epsilon = config
 
    q = (
        float(batch)
        / float(DATASET_SIZE)
    )
 
    T = max(
        1,
        int(
            np.ceil(
                epochs / q
            )
        ),
    )
 
    gamma = clip * theta
 
    result = find_k_given_epsilon(
        gamma=gamma,
        q=q,
        T=T,
        eps_target=target_epsilon,
        delta=DELTA,
        n_params=MODEL_SIZE,
        k_min=K_MIN,
        k_max_init=K_MAX_INIT,
        k_hard_cap=K_HARD_CAP,
        rel_tol=K_REL_TOL,
        max_iters=K_MAX_BISECTION_ITERS,
    )
 
    if not result["feasible"]:
        return (
            f"NA,NA,NA,"
            f"{theta:.12g},"
            f"{batch},"
            f"{clip:.12g},"
            f"{epochs},"
            f"{target_epsilon:.12g}"
        )
 
    k = result["k"]
 
    # SNR = C * (k - 1) * theta
    snr = (
        clip
        * (k - 1.0)
        * theta
    )
 
    # Distortion = 1 / ((k - 1) * theta)
    distortion = (
        1.0
        / (
            (k - 1.0)
            * theta
        )
    )
 
    return (
        f"{k:.12g},"
        f"{snr:.12g},"
        f"{distortion:.12g},"
        f"{theta:.12g},"
        f"{batch},"
        f"{clip:.12g},"
        f"{epochs},"
        f"{target_epsilon:.12g}"
    )
 
 
def main():
 
    print(
        "k,snr,distortion,theta,batch,clip,epoch,epsilon",
        flush=True
    )
 
    # Create all configurations
    configs = list(
        product(
            THETAS,
            CLIPS,
            EPOCHS_LIST,
            BATCHES,
            EPSILONS,
        )
    )
 
    print(
        f"# Configurations: {len(configs)}",
        flush=True
    )
 
    # One CPU process per configuration
    num_workers = len(configs)
 
    with Pool(processes=num_workers) as pool:
 
        results = pool.map(
            run_one_config,
            configs
        )
 
    # Print results
    for result in results:
        print(result)
 
 
if __name__ == "__main__":
    main()
