import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Size and power of the two-stage bootstrap test at N = 38, T = 156 (B = 199) under Gaussian, GARCH,
Student-t and omitted-lag innovations."""
import json, time, numpy as np
from scipy.stats import chi2
from gstarx_pw import *
from coords38 import COORDS
lat = np.array([c[0] for c in COORDS]); lon = np.array([c[1] for c in COORDS])
W0 = inverse_distance(lat, lon); N, T, B = 38, 156, 199
M = annihilator(design_eid_weekly(T)); crit = chi2.ppf(0.95, N - 2)

def run(name, R, seed, truth="prior", innov="gauss", c2=None):
    rng = np.random.default_rng(seed); rej, eq, lb = [], [], []
    t0 = time.time()
    for r in range(R):
        p0 = rng.uniform(-0.15, 0.05, N); p1 = rng.uniform(0.25, 0.45, N)
        Wt = {"prior": W0, "sparse": sparse_matrix(N, rng), "mixture": None}[truth]
        if truth == "mixture":
            Wt = row_normalize(0.6 * W0 + 0.4 * sparse_matrix(N, rng))
        A = np.diag(p0) + p1[:, None] * Wt
        A2 = None if c2 is None else np.diag(rng.choice([-1, 1], N) * rng.uniform(c2 / 2, c2, N))
        Zh = M @ simulate(A, T, rng, rho=0.5, innov=innov, A2=A2)
        p, Wi = bootstrap_test(Zh, W0, M, rng, B)
        rej.append(p <= 0.05); eq.append(np.mean(Wi > crit))
        if c2 is not None or innov != "gauss" or r < 200:
            lb.append(ljung_box_reject(wald_stats(Zh, W0)[3], 12, 2))
    out = dict(design=name, R=R, boot_reject=float(np.mean(rej)), eqwise_share=float(np.mean(eq)),
               lb12_share=float(np.mean(lb)), sec=round(time.time() - t0, 1))
    print(out, flush=True); return out

import sys
designs = {"gauss": ("H0, Gaussian", 1000, 101, {}), "garch": ("H0, GARCH(1,1) common volatility", 1000, 102, {"innov": "garch"}),
           "t5": ("H0, Student-t(5)", 1000, 103, {"innov": "t"}), "lag2a": ("H0, omitted lag-2 dynamics (c=0.40)", 1000, 104, {"c2": 0.40}),
           "lag2b": ("H0, omitted lag-2 dynamics (c=0.55)", 1000, 105, {"c2": 0.55}),
           "sparse": ("H1 sparse, Gaussian", 500, 106, {"truth": "sparse"}), "mixture": ("H1 mixture, Gaussian", 500, 107, {"truth": "mixture"})}
for key in sys.argv[1:]:
    name, R, seed, kw = designs[key]
    json.dump(run(name, R, seed, **kw), open(f"exact_{key}.json", "w"), indent=1)
