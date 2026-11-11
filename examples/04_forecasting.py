"""Rolling-origin forecasts on the simulated simulated price panel, with Diebold-Mariano tests."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import gstarx_pw as g

d = g.simulate_price_panel(seed=2026)
Y, X, W0 = d["y"], d["X"], d["W0"]
window = 156
kappa, _ = g.select_kappa(g.exogenous_stage(Y[:window], X[:window])[1], W0)   # selected once, for speed
errs, _ = g.rolling_origin(Y, X, W0, window=window, H=4, step=2, kappa=kappa)
rmse = {m: np.sqrt((e ** 2).mean(axis=(0, 2))) for m, e in errs.items()}
print("RMSE by horizon (relative to the exogenous stage alone):")
for m in ["var", "fixed", "pw"]:
    print(f"  {m:6s}", np.round(rmse[m] / rmse["exog"], 3))
for other in ["fixed", "var", "exog"]:        # panel test: errors averaged over locations at each origin
    stat, pv = g.dm_test(np.sqrt((errs["pw"][:, 0, :] ** 2).mean(1)), np.sqrt((errs[other][:, 0, :] ** 2).mean(1)))
    print(f"  DM, GSTARX-PW vs {other:5s}: statistic {stat:6.2f}, p = {pv:.3f}")
