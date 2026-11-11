"""Quick start: test a pre-specified weight matrix and estimate the neighbourhood on a simulated panel."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import gstarx_pw as g

rng = np.random.default_rng(2026)
N, T = 12, 300
lat, lon = rng.uniform(-8.3, -6.7, N), rng.uniform(111.0, 114.4, N)
W0 = g.inverse_distance(lat, lon)                       # the pre-specified matrix (H0)
W_true = g.sparse_matrix(N, rng)                        # true neighbourhood, unrelated to distance
A = np.diag(rng.uniform(0.45, 0.65, N)) + rng.uniform(0.20, 0.40, N)[:, None] * W_true
X = g.design_generic(T)                                 # intercept, harmonics, moving calendar block
Y = X @ rng.normal(0, 0.5, (X.shape[1], N)) + g.simulate(A, T, rng)
B, Zhat = g.exogenous_stage(Y, X)                       # stage 1: deterministic part by OLS
M = g.annihilator(X)

p, Wi = g.bootstrap_test(Zhat, W0, M, rng, B=199)      # Algorithm 2
print(f"Wald test of the inverse-distance matrix: W_tot = {Wi.sum():.1f}, bootstrap p = {p:.3f}")
p_true, _ = g.bootstrap_test(Zhat, W_true, M, rng, B=199)
print(f"Wald test of the true matrix (control):   bootstrap p = {p_true:.3f}")

kappa, fit = g.select_kappa(Zhat, W0)                   # Algorithm 1 with inner validation of kappa
print(f"selected kappa = {kappa}; MAE of the weights: prior {g.mae_offdiag(W0, W_true):.4f}, "
      f"GSTARX-PW {g.mae_offdiag(fit['W'], W_true):.4f}")
hits = sum(W_true[i, int(np.argmax(fit['W'][i]))] > 0 for i in range(N))
print(f"the strongest estimated link is a true link for {hits} of {N} locations")
