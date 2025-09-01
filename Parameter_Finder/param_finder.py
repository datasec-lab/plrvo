import numpy as np
from scipy.optimize import minimize
from scipy.stats import gamma
import matlab.engine
import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()
# ==== Constants ====
target_epsilon = 1  # or whatever you want
tolerance = 0.005       # In the paper, the pseudocode targets \epsilon directly, but that would be too time consuming or possibly impossible, so we target a window of epsilons around the target epsilon
T = 250
clip = 0.1
N = 26000
delta = 1e-5
q = 0.04

# ==== Initial Guess ====
x0 = [80, 8e-2]  # Initial guess: k > 1, theta > 0

epsilon_cache = {}

def get_epsilon_cached(k, theta):
    # Round or stringify for stable dictionary keys (avoid float precision issues)
    key = (round(k, 12), round(theta, 12))
    
    if key in epsilon_cache:
        return epsilon_cache[key]
    
    # Call MATLAB function
    eps_val = float(eng.get_epsilon(float(T),float(clip), float(N), 
                                    float(delta), float(q), float(theta), float(k)))
    
    # Cache and return
    epsilon_cache[key] = eps_val
    return eps_val

# ==== Objective Function ====
def objective(x):
    k, theta = x
    if k <= 1 or theta <= 0:
        return np.inf  # Infeasible region
    return 1 / ((k - 1) * theta)

# ==== Constraint 1: Gamma CDF at 0.1 < 0.001 ====
def gamma_cdf_constraint(x):
    k, theta = x
    if k <= 0 or theta <= 0:
        return -np.inf  # Infeasible
    cdf_val = gamma.cdf(0.1, a=k, scale=theta)
    return 0.001 - cdf_val  # c(x) >= 0

# ==== Constraint 2: get_epsilon(k, theta) ≈ target ± tolerance ====

# Lower bound: get_epsilon(k, theta) ≥ target - tolerance
def epsilon_lower_constraint(x):
    k, theta = x
    return get_epsilon_cached(k, theta) - (target_epsilon - tolerance)

def epsilon_upper_constraint(x):
    k, theta = x
    return (target_epsilon + tolerance) - get_epsilon_cached(k, theta)
# ==== Combine Constraints ====
constraints = [
    {'type': 'ineq', 'fun': gamma_cdf_constraint},
    {'type': 'ineq', 'fun': epsilon_lower_constraint},
    {'type': 'ineq', 'fun': epsilon_upper_constraint},
]

# ==== Run Optimization ====
result = minimize(
    objective,
    x0,
    method='COBYLA',
    constraints=constraints,
    bounds=[(1.001, None), (1e-12, 0.1)],  # k > 1, theta > 0
    options={'maxiter': 100,'disp': True}
)
k, theta = result.x
print("\n--- Optimization Result ---")
print("Optimal k, theta:", result.x)
print("Minimum distortion:", result.fun)
print("final epsilon value:", eng.get_epsilon(float(T),float(clip), float(N), float(delta), float(q), float(theta), float(k)))

# Close MATLAB engine
eng.quit()
