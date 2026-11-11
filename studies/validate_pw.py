import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import time, json, numpy as np
from gstarx_pw import *

def cross_corr_W(Z):
    X, Y = Z[:-1], Z[1:]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    C = (Yc.T @ Xc) / np.sqrt(np.outer((Yc * Yc).sum(0), (Xc * Xc).sum(0)))   # corr(y_i,t , y_j,t-1)
    C = np.abs(C); np.fill_diagonal(C, 0)
    return C / C.sum(1, keepdims=True)

def rep_recovery(W0, T, rng, phi1=(0.20, 0.40)):
    N = W0.shape[0]
    Wt = sparse_matrix(N, rng)
    p0 = rng.uniform(0.45, 0.65, N); p1 = rng.uniform(*phi1, N)
    Z = simulate(np.diag(p0) + p1[:, None] * Wt, T, rng)
    Zh = annihilator(design_generic(T)) @ Z
    k, fit = select_kappa(Zh, W0)
    mask = ~np.eye(N, dtype=bool)
    Avar = var_fit(Zh); Wv = proj_simplex_rows(var_implied(Avar), mask)
    return dict(pw=mae_offdiag(fit["W"], Wt), var=mae_offdiag(Wv, Wt), prior=mae_offdiag(W0, Wt),
                ccor=mae_offdiag(cross_corr_W(Zh), Wt), kappa=k)

rng0 = np.random.default_rng(7)
lat = rng0.uniform(-8.3, -6.7, 20); lon = rng0.uniform(111.0, 114.4, 20); W0 = inverse_distance(lat, lon)
rng = np.random.default_rng(2026)
t0 = time.time(); r = rep_recovery(W0, 200, rng); print("one replication:", round(time.time() - t0, 1), "s", r, flush=True)
res = [r] + [rep_recovery(W0, 200, rng) for _ in range(29)]
summ = {k: float(np.mean([x[k] for x in res])) for k in ["pw", "var", "prior", "ccor"]}
summ["kappa_median"] = float(np.median([x["kappa"] for x in res])); summ["sec"] = round(time.time() - t0, 1)
print(summ); json.dump(summ, open("validate_pw.json", "w"), indent=1)
