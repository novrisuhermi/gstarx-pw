import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Recovery of W as the spatial coefficient shrinks (the plug-in ratio divides by phi_1), and sensitivity to the starting values."""
import json, sys, time, numpy as np
from gstarx_pw import *

rng0 = np.random.default_rng(7)
lat = rng0.uniform(-8.3, -6.7, 20); lon = rng0.uniform(111.0, 114.4, 20); W0 = inverse_distance(lat, lon)
N, T, R = 20, 200, 30
mask = ~np.eye(N, dtype=bool)
Mx = annihilator(design_generic(T))

def cell(c, seed):
    rng = np.random.default_rng(seed); rows = []; t0 = time.time()
    for r in range(R):
        Wt = sparse_matrix(N, rng)
        p0 = rng.uniform(0.45, 0.65, N); p1 = rng.uniform(0.8 * c, 1.2 * c, N)
        Zh = Mx @ simulate(np.diag(p0) + p1[:, None] * Wt, T, rng)
        Av = var_fit(Zh); raw = var_implied(Av)
        k, fit = select_kappa(Zh, W0)
        full = pw_fit(Zh, W0, k, return_starts=True)
        Ws = [s[1] for s in full["starts"]]; Ss = np.array([s[0] for s in full["starts"]])
        spread = np.max([np.abs(Ws[a] - Ws[b]).max(1) for a in range(3) for b in range(a + 1, 3)], axis=0)
        gap = (Ss.max(0) - Ss.min(0)) / Ss.min(0)
        rows.append(dict(prior=mae_offdiag(W0, Wt), var=mae_offdiag(proj_simplex_rows(raw, mask), Wt),
                         raw=mae_offdiag(raw, Wt), signflip=float(np.mean(Av[mask].reshape(N, N - 1).sum(1) <= 0)),
                         pw=mae_offdiag(full["W"], Wt), kappa=k,
                         agree=float(np.mean(spread < 1e-3)), gap_max=float(gap.max()), gap_med=float(np.median(gap)),
                         pw_prior_start=mae_offdiag(Ws[0], Wt), pw_unif_start=mae_offdiag(Ws[1], Wt), pw_var_start=mae_offdiag(Ws[2], Wt),
                         iters=[int(s[4]) for s in full["starts"]]))
    agg = {k_: float(np.mean([x[k_] for x in rows])) for k_ in ["prior", "var", "pw", "signflip", "agree", "gap_med",
                                                                  "pw_prior_start", "pw_unif_start", "pw_var_start"]}
    agg.update(raw_median=float(np.median([x["raw"] for x in rows])), raw_max=float(np.max([x["raw"] for x in rows])),
               kappa_median=float(np.median([x["kappa"] for x in rows])), gap_max=float(np.max([x["gap_max"] for x in rows])),
               iters_median=float(np.median([i for x in rows for i in x["iters"]])), iters_max=int(np.max([i for x in rows for i in x["iters"]])),
               c=c, R=R, sec=round(time.time() - t0, 1))
    print(agg, flush=True); return agg

for arg in sys.argv[1:]:
    c = float(arg)
    json.dump(cell(c, int(1000 * c) + 17), open(f"phi_{arg}.json", "w"), indent=1)
