"""Forecasting with GSTARX models, rolling-origin evaluation and the Diebold-Mariano test."""
import numpy as np
from scipy import stats
from .core import fixed_fit, var_fit, pw_fit, select_kappa


def forecast_path(A, z_last, H):
    """Recursive forecasts Z_{T+h|T} = A^h Z_T, h = 1..H, of the stochastic component."""
    out, z = np.empty((H, len(z_last))), np.asarray(z_last, float)
    for h in range(H):
        z = A @ z
        out[h] = z
    return out


def dm_test(e1, e2, h=1, power=2):
    """Diebold-Mariano test with the small-sample correction of Harvey, Leybourne and Newbold (1997).
    e1, e2: forecast errors of the two methods. A negative statistic means method 1 is more accurate."""
    d = np.abs(np.asarray(e1)) ** power - np.abs(np.asarray(e2)) ** power
    T, dbar = len(d), d.mean()
    gam = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gam[0] + 2 * sum(gam[1:])) / T
    stat = dbar / np.sqrt(var) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return stat, 2 * stats.t.sf(abs(stat), df=T - 1)


def rolling_origin(Y, X, W0, window, H=4, step=1, kappa=None, methods=("exog", "var", "fixed", "pw")):
    """Rolling-origin evaluation. At every origin the exogenous stage and all models are re-estimated on the
    training window only; forecasts add the deterministic part X b back. Returns a dict of forecast errors,
    each of shape (origins, H, N), and the penalties used by GSTARX-PW (selected at every origin if kappa is None)."""
    T, N = Y.shape
    errs, kaps = {m: [] for m in methods}, []
    for t in range(window, T - H + 1, step):
        Ytr, Xtr = Y[t - window:t], X[t - window:t]
        B, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
        Z = Ytr - Xtr @ B
        As = {}
        if "exog" in methods:
            As["exog"] = np.zeros((N, N))
        if "var" in methods:
            As["var"] = var_fit(Z)
        if "fixed" in methods:
            p0, p1, _ = fixed_fit(Z, W0)
            As["fixed"] = np.diag(p0) + p1[:, None] * W0
        if "pw" in methods:
            k, fit = select_kappa(Z, W0) if kappa is None else (kappa, pw_fit(Z, W0, kappa))
            As["pw"], _ = fit["A"], kaps.append(k)
        for m in methods:
            errs[m].append(Y[t:t + H] - (X[t:t + H] @ B + forecast_path(As[m], Z[-1], H)))
    return {m: np.array(e) for m, e in errs.items()}, np.array(kaps)
