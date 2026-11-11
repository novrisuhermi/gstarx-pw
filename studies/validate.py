import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time, json, numpy as np
from scipy.stats import chi2
from gstarx_pw import *

def generic_cell(N, seed):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-8.3, -6.7, N); lon = rng.uniform(111.0, 114.4, N)
    return inverse_distance(lat, lon)

def rep_size(W0, T, rng, B=199, rho=0.3):
    N = W0.shape[0]
    p0 = rng.uniform(0.45, 0.65, N); p1 = rng.uniform(0.20, 0.40, N)
    A = np.diag(p0) + p1[:, None] * W0
    Z = simulate(A, T, rng, rho=rho)
    M = annihilator(design_generic(T))
    Zh = M @ Z
    p, Wi = bootstrap_test(Zh, W0, M, rng, B)
    return p <= 0.05, np.mean(Wi > chi2.ppf(0.95, N - 2))

out = {}
for T, R in [(100, 150), (200, 150), (400, 150)]:
    W0 = generic_cell(20, 1000 + T); rng = np.random.default_rng(T)
    t0 = time.time(); r = np.array([rep_size(W0, T, rng) for _ in range(R)])
    out[f"size_N20_T{T}"] = dict(boot=r[:, 0].mean(), eqwise=r[:, 1].mean(), sec=round(time.time() - t0, 1))
    print(T, out[f"size_N20_T{T}"], flush=True)
json.dump(out, open("validate_size.json", "w"), indent=1)
