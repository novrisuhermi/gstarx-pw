"""Public data: daily wind speeds at 12 Irish stations (Haslett and Raftery, 1989; R package gstat)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import gstarx_pw as g

w = g.load_irish_wind()
S = w["speeds"].loc["1976-01-01":"1978-12-31"]
Y = np.sqrt(S.to_numpy())                               # square-root transform, as in Haslett and Raftery
X = g.harmonic_design(len(Y), period=365.25, K=3)       # exogenous stage: intercept + 3 annual harmonics
B, Zhat = g.exogenous_stage(Y, X)
W0 = g.inverse_distance(w["lat"], w["lon"])
p, Wi = g.bootstrap_test(Zhat, W0, g.annihilator(X), np.random.default_rng(7), B=199)
print(f"T = {len(Y)} days, N = {Y.shape[1]} stations; inverse-distance W0: W_tot = {Wi.sum():.1f}, bootstrap p = {p:.3f}")
kappa, fit = g.select_kappa(Zhat, W0)
print(f"selected kappa = {kappa}")
for i, c in enumerate(w["codes"]):
    j = int(np.argmax(fit["W"][i]))
    side = "west" if w["lon"][j] < w["lon"][i] else "east"
    print(f"  {w['stations'][i]:14s} <- {w['stations'][j]:14s} (w = {fit['W'][i, j]:.2f}, neighbour lies to the {side})")
