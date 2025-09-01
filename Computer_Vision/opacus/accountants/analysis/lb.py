import math
import numpy as np
import warnings
from scipy.stats import norm, truncnorm, expon
from typing import List, Tuple, Union
from . import rdp as gaussian_analysis
from decimal import Decimal
from scipy.special import binom

def convert_mgf_rdp(args, order, sample_rate, num_steps):
    clip = args["max_grad_norm"]
    return maf(args, order, sample_rate, num_steps)
    
def maf(args, moment, sample_rate, num_steps):
    clip = args["max_grad_norm"]
    M = 0
    b = args['b']
    bias = args['bias']
    bb = b#*(clip/bias)
    for k in range(0, moment):
        if k <= 1:
            rhs = 1
        if k > 1:
            x = k*clip
            y = (k-1)*clip
            A = 0.5*math.exp(y/bb)
            B = (2*k)-1
            B = ((-0.5*math.exp(-x/bb))+(0.5*math.exp(y/bb)))/B;
            Cp=0.5*math.exp(-x/bb);
            rhs=A+B+Cp;
            #print(B)
            
        binom_coeff = binom(moment, k)
        #term = b*((1-sample_rate)**(moment-k))*(sample_rate**k)*M_p(args, k*clip)
        term = binom_coeff*((1-sample_rate)**(moment-k))*(sample_rate**k)*rhs
        M=M+term
        ret = math.log(M)
        #print(M)
    return num_steps*(math.log(M)/moment)
    #return 0

def M_p(args, moment):
    res =1
    if args['gamma']:
        res *= mgf_gamma(moment, args["theta"], args['k'])
    #if args['uniform']:
    #    res *= mgf_uniform(moment, args['a'], args['b'])
    return res
    
def mgf_gamma(moment, theta, k):
    if moment >= 1/theta:
        return math.inf
        raise Exception("moment must be less than 1/theta")
    return pow(1-moment*(theta), -1*k)

def mgf_truncated_normal(l, u, mu, sigma, t):
    alpha = (l - mu) / sigma
    beta = (u - mu) / sigma
    
    num = truncnorm.cdf(beta - sigma * t, alpha, beta, loc=mu, scale=sigma) - truncnorm.cdf(alpha - sigma * t, alpha, beta, loc=mu, scale=sigma)
    den = truncnorm.cdf(beta, alpha, beta, loc=mu, scale=sigma) - truncnorm.cdf(alpha, alpha, beta, loc=mu, scale=sigma)
    
    return np.exp((mu * t + 0.5 * sigma ** 2 * t ** 2)/2) * num / den
    
def mgf_expon(moment, lam):
    paren = 1-(moment*(1/lam))
    if paren == 0:
        return math.inf
    return 1/paren
    
    
def mgf_uniform(moment, a, b):
    n = math.exp(moment*b) - math.exp(moment*a)
    d = moment*(b-a)
    return n/d
    
def convert_mgf_rdp(args, order, sample_rate, num_steps):
    clip = args["max_grad_norm"]
    return maf(args, order, sample_rate, num_steps)
    
def _compute_rdp(args, order, sample_rate, num_steps):
    return convert_mgf_rdp(args, order, sample_rate, num_steps)
    
def compute_rdp(args, num_steps, orders: Union[List[float], float]
) -> Union[List[float], float]:
    r"""Computes Renyi Differential Privacy (RDP) guarantees of the
    Sampled Gaussian Mechanism (SGM) iterated ``steps`` times.

    Args:
        q: Sampling rate of SGM.
        noise_multiplier: The ratio of the standard deviation of the
            additive Gaussian noise to the L2-sensitivity of the function
            to which it is added. Note that this is same as the standard
            deviation of the additive Gaussian noise when the L2-sensitivity
            of the function is 1.
        steps: The number of iterations of the mechanism.
        orders: An array (or a scalar) of RDP orders.

    Returns:
        The RDP guarantees at all orders; can be ``np.inf``.
    """
    #alpha, rdp = _compute_rdp(args)
    #print(num_steps)
    if isinstance(orders, float):
        rdp = _compute_rdp(args, orders, 1, num_steps)
    else:
        rdp = np.array([_compute_rdp(args, order, 1, num_steps) for order in orders])
    return rdp
    
def compute_rdp_subsample(args, num_steps, delta, orders: Union[List[float], float], sample_rate
) -> Union[List[float], float]:
    if isinstance(orders, float):
        rdp = maf(args, orders, sample_rate, num_steps)
    else:
        rdp = np.array([maf(args, order, sample_rate, num_steps) for order in orders])
        
   # np.array(rens)*num_steps
    return np.array(rdp)
        
    


def get_privacy_spent(
    *, orders: Union[List[float], float], rdp: Union[List[float], float], delta: float
) -> Tuple[float, float]:
    r"""Computes epsilon given a list of Renyi Differential Privacy (RDP) values atS
    multiple RDP orders and target ``delta``.
    The computation of epslion, i.e. conversion from RDP to (eps, delta)-DP,
    is based on the theorem presented in the following work:
    Borja Balle et al. "Hypothesis testing interpretations and Renyi differential privacy."
    International Conference on Artificial Intelligence and Statistics. PMLR, 2020.
    Particullary, Theorem 21 in the arXiv version https://arxiv.org/abs/1905.09982.
    Args:
        orders: An array (or a scalar) of orders (alphas).
        rdp: A list (or a scalar) of RDP guarantees.
        delta: The target delta.
    Returns:
        Pair of epsilon and optimal order alpha.
    Raises:
        ValueError
            If the lengths of ``orders`` and ``rdp`` are not equal.
    """
    #orders = [orders] * len(rdp)
    orders_vec = np.atleast_1d(orders)
    rdp_vec = np.atleast_1d(rdp)
    #print(rdp_vec)
    if len(orders_vec) != len(rdp_vec):
        raise ValueError(
            f"Input lists must have the same length.\n"
            f"\torders_vec = {orders_vec}\n"
            f"\trdp_vec = {rdp_vec}\n"
        )

    eps = (
        rdp_vec
        - ((np.log(delta) + np.log(orders_vec+1)) / (orders_vec))
        + np.log((orders_vec) / (orders_vec+1))
    )
    #print(eps)
    # special case when there is no privacy
    if np.isnan(eps).all():
        return np.inf, np.nan

    idx_opt = np.nanargmin(eps)  # Ignore NaNs
    if idx_opt == 0 or idx_opt == len(eps) - 1:
        extreme = "smallest" if idx_opt == 0 else "largest"
        warnings.warn(
            f"Optimal order is the {extreme} alpha. Please consider expanding the range of alphas to get a tighter privacy bound."
        )
    return eps[idx_opt], orders_vec[idx_opt]
