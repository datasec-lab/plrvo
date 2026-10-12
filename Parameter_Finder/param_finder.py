#!/usr/bin/env python3


import csv
import importlib.util
import math
from pathlib import Path

import numpy as np


# ============================================================
# USER CONFIGURATION
# ============================================================

FINDER_FILE = "find_plrvo.py"

MODEL_SIZE = 26000
DATASET_SIZE = 60000
DELTA = 1e-5

BATCH = 128
EPOCHS = 3
EPSILON_TARGET = 1.0


LAMBDA_CALIBRATE = 64
LAMBDA_VERIFY_INITIAL = 64
LAMBDA_VERIFY_HARD_CAP = 128

# Search k logarithmically
K_GRID = np.logspace(
    np.log10(2.001),
    np.log10(8000),
    15
)

# These are training candidates. Change this to desire clip values.
C_GRID = np.array([0.1,0.5,1], dtype=float)

# gamma bisection
GAMMA_LO = 1e-9
GAMMA_REL_TOL = 2e-3
GAMMA_MAX_ITERS = 60

# Output

OUT_CANDIDATES_CSV = "training_candidates.csv"


# ============================================================
# LOAD EXISTING ACCOUNTANT
# ============================================================

def load_finder():
    path = Path(FINDER_FILE).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put this script in the same directory "
            f"as {FINDER_FILE}."
        )

    spec = importlib.util.spec_from_file_location("plrvo_finder", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


finder = load_finder()


# ============================================================
# TRAINING / ACCOUNTING QUANTITIES
# ============================================================

Q = BATCH / DATASET_SIZE
T = max(1, int(math.ceil(EPOCHS / Q)))

print("=" * 72)
print("ORIGINAL PLRV-O PARAMETER SEARCH")
print("=" * 72)
print(f"model size      : {MODEL_SIZE:,}")
print(f"dataset size    : {DATASET_SIZE:,}")
print(f"batch           : {BATCH}")
print(f"epochs          : {EPOCHS}")
print(f"sampling q      : {Q:.12g}")
print(f"steps T         : {T}")
print(f"target epsilon  : {EPSILON_TARGET}")
print(f"delta           : {DELTA}")
print(f"lambda calibrate: {LAMBDA_CALIBRATE}")
print("=" * 72)


# ============================================================
# PRIVACY CALCULATIONS
# ============================================================

def coarse_epsilon(k, gamma):
    """Reuse the accountant in find_plrvo(1).py."""
    return finder.plrvo_epsilon_coarse(
        k,
        gamma,
        Q,
        T,
        DELTA,
        MODEL_SIZE,
        lambda_max=LAMBDA_CALIBRATE,
    )


def verified_epsilon(k, gamma):
    """
    Final verification with a larger order range.

    The existing accountant can expand its order range until the minimizing
    order is interior, or until the specified hard cap is reached.
    """
    return finder.plrvo_epsilon(
        k,
        gamma,
        Q,
        T,
        DELTA,
        MODEL_SIZE,
        lambda_initial=LAMBDA_VERIFY_INITIAL,
        lambda_hard_cap=LAMBDA_VERIFY_HARD_CAP,
        margin=2,
    )


def find_gamma_for_k(k):
    """
    Solve:
        gamma*(k) = max gamma such that epsilon(k,gamma) <= epsilon_target.
    """
    # The Gamma MGF requires gamma * eta * x < 1.
    # Since x_1 = 1, a conservative calibration cap is:
    gamma_cap = 0.90 / (LAMBDA_CALIBRATE + 1.0)

    lo = GAMMA_LO
    eps_lo = coarse_epsilon(k, lo)

    if not np.isfinite(eps_lo) or eps_lo > EPSILON_TARGET:
        return None

    hi = min(1e-3, gamma_cap)
    eps_hi = coarse_epsilon(k, hi)

    # Expand until we cross epsilon_target.
    while eps_hi <= EPSILON_TARGET and hi < gamma_cap:
        lo = hi
        hi = min(2.0 * hi, gamma_cap)
        eps_hi = coarse_epsilon(k, hi)

    # If even the MGF-safe cap is feasible, return the cap as a boundary solution 
    if eps_hi <= EPSILON_TARGET:
        gamma = hi
        eps_v, lam_star, lam_checked, boundary = verified_epsilon(k, gamma)
        return {
            "k": k,
            "gamma": gamma,
            "epsilon": eps_v,
            "lambda_star": lam_star,
            "lambda_checked": lam_checked,
            "lambda_boundary": boundary,
            "gamma_boundary": True,
        }

    # Log-space bisection because gamma can span several orders of magnitude.
    for _ in range(GAMMA_MAX_ITERS):
        if hi / lo - 1.0 <= GAMMA_REL_TOL:
            break

        mid = math.sqrt(lo * hi)
        eps_mid = coarse_epsilon(k, mid)

        if np.isfinite(eps_mid) and eps_mid <= EPSILON_TARGET:
            lo = mid
        else:
            hi = mid

    gamma = lo

    
    for _ in range(20):
        eps_v, lam_star, lam_checked, boundary = verified_epsilon(k, gamma)

        if np.isfinite(eps_v) and eps_v <= EPSILON_TARGET:
            return {
                "k": k,
                "gamma": gamma,
                "epsilon": eps_v,
                "lambda_star": lam_star,
                "lambda_checked": lam_checked,
                "lambda_boundary": boundary,
                "gamma_boundary": False,
            }

        gamma *= 0.98

    return None


# ============================================================
# MAIN K SEARCH
# ============================================================

def main():
    results = []

    print("\nSearching k -> gamma*(k) -> J(k) ...")
    print("-" * 72)

    for i, k in enumerate(K_GRID, 1):

        print(
            f"[{i}/{len(K_GRID)}] Currently searching k = {k:.8g}",
            flush=True
        )
     

        try:
            r = find_gamma_for_k(float(k))
        except Exception as exc:
            print(f"    ERROR: {exc}")
            continue

        if r is None:
            print("    no feasible gamma found")
            continue

      
        # J = C (k-1) theta = (k-1) gamma
        J = (r["k"] - 1.0) * r["gamma"]

        r["J"] = J
        results.append(r)

     

    if not results:
        raise RuntimeError(
            "No feasible k values were found. "
            "Check the accountant settings and lambda range."
        )

    # Rank by the paper's objective.
    results.sort(key=lambda x: x["J"], reverse=True)
    best = results[0]

    print("\n" + "=" * 72)
    print("BEST PRIVACY-UTILITY CANDIDATE")
    print("=" * 72)
    print(f"k*       = {best['k']:.12g}")
    print(f"gamma*   = {best['gamma']:.12g}")
    print(f"J*       = {best['J']:.12g}")
    print(f"epsilon  = {best['epsilon']:.12g}")
    print(f"lambda*  = {best['lambda_star']}")
    print(f"orders   = {best['lambda_checked']}")
    print("=" * 72)

   

    candidate_rows = []

    for C in C_GRID:
        theta = best["gamma"] / C

        # Expected absolute noise per coordinate for Laplace(0,1/u):
        # E|z| = E[1/u] = 1 / ((k-1) theta)
        expected_abs_noise = 1.0 / ((best["k"] - 1.0) * theta)


        # theta > 0.1/(k-1), equivalently expected |z| < 10.
        expected_l1 = expected_abs_noise

        distortion_feasible = expected_l1 < 10.0

        candidate_rows.append([
            f"{best['k']:.12g}",
            f"{best['gamma']:.12g}",
            f"{C:.12g}",
            f"{theta:.12g}",
            f"{expected_abs_noise:.12g}",
            distortion_feasible,
        ])

    with open(OUT_CANDIDATES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "k", "gamma", "C", "theta",
            "expected_abs_noise",
            "paper_distortion_feasible"
        ])
        writer.writerows(candidate_rows)

    print("\nTraining candidates for the same privacy point:")
    print("-" * 72)
    print(
        f"{'k':>12} {'C':>10} {'theta':>14} {'E|z|':>14} "
        f"{'dist<10':>12}"
    )
    for row in candidate_rows:
        print(
            f"{float(row[0]):12.7g} "
            f"{float(row[2]):10.5g} "
            f"{float(row[3]):14.7g} "
            f"{float(row[4]):14.7g} "
            f"{str(row[5]):>10}"
        )

    print("\nFiles written:")
    print(f"  {OUT_CANDIDATES_CSV}")
    


if __name__ == "__main__":
    main()
