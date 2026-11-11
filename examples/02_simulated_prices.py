"""The full workflow on a SIMULATED weekly price panel for 26 generic cities (City A ... City Z)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
import gstarx_pw as g

d = g.simulate_price_panel(seed=2026)
Y, X, names, east, north = d["y"], d["X"], d["names"], d["east_km"], d["north_km"]
B, Zhat = g.exogenous_stage(Y, X)
M = g.annihilator(X)
print("calendar effects on the log price (mean over cities):", dict(zip(d["columns"][5:], np.round(B[5:].mean(1), 3))))
rng = np.random.default_rng(1)
candidates = {"uniform": g.uniform_matrix(len(names)), "inverse distance": d["W0"],
              "4-nearest neighbours": g.knn_matrix(north, east, 4, kind="euclidean"),
              "cross-correlation": g.cross_correlation_matrix(Zhat), "true (control)": d["W_true"]}
for name, W0 in candidates.items():
    p, Wi = g.bootstrap_test(Zhat, W0, M, rng, B=199)
    print(f"{name:22s} W_tot = {Wi.sum():8.1f}   bootstrap p = {p:.3f}   {'rejected' if p <= 0.05 else 'not rejected'}")
kappa, fit = g.select_kappa(Zhat, d["W0"])
print(f"GSTARX-PW: kappa = {kappa}, MAE prior {g.mae_offdiag(d['W0'], d['W_true']):.4f} -> {g.mae_offdiag(fit['W'], d['W_true']):.4f}")
print("synthetic hubs:", [names[h] for h in d["hubs"]])
