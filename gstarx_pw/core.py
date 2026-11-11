"""Core algorithms of GSTARX-PW.

Simulation of GSTARX(1;1) panels, the exogenous stage, the Wald test of a pre-specified
weight matrix with its two-stage residual bootstrap, and the penalized estimator of W.
"""
import numpy as np

# ----------------------------------------------------------------------------- geometry
def gc_km(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_matrix(u, v, kind="greatcircle"):
    """Pairwise distances: great-circle km for (lat, lon) in degrees, or Euclidean for planar coordinates."""
    if kind == "euclidean":
        return np.hypot(u[:, None] - u[None, :], v[:, None] - v[None, :])
    return gc_km(u, v)


def inverse_distance(lat, lon, kind="greatcircle"):
    """Row-normalized inverse-distance weights (kind: "greatcircle" for lat/lon, "euclidean" for planar)."""
    d = distance_matrix(lat, lon, kind)
    np.fill_diagonal(d, np.inf)
    w = 1.0 / d
    return w / w.sum(1, keepdims=True)


def sparse_matrix(N, rng, k=3):
    W = np.zeros((N, N))
    for i in range(N):
        js = rng.choice(np.delete(np.arange(N), i), k, replace=False)
        W[i, js] = rng.uniform(0.5, 1.5, k)
    return W / W.sum(1, keepdims=True)


def row_normalize(M):
    """Divide every row by its sum."""
    return M / M.sum(1, keepdims=True)


def row_normalise(M):
    return M / M.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- exogenous designs
def design_generic(T, s=52, block=3, start=30):
    """Intercept, two annual harmonics and a calendar block moving one period earlier per year."""
    t = np.arange(1, T + 1)
    cv = np.zeros(T)
    for y in range(T // s + 2):
        b0 = y * s + start - y
        cv[(t - 1 >= b0) & (t - 1 < b0 + block)] = 1.0
    return np.column_stack([np.ones(T), np.sin(2 * np.pi * t / s), np.cos(2 * np.pi * t / s),
                            np.sin(4 * np.pi * t / s), np.cos(4 * np.pi * t / s), cv])


def design_eid_weekly(T=156):
    """Weekly design: intercept, two annual harmonics, Ramadan, Eid-week and post-Eid dummies.
    Weeks start on Monday 4 January 2021; Eid al-Fitr 13 May 2021, 2 May 2022, 21 April 2023, 10 April 2024."""
    eid = np.array(['2021-05-13', '2022-05-02', '2023-04-21', '2024-04-10'], dtype='datetime64[D]')
    weeks = np.datetime64('2021-01-04') + 7 * np.arange(T)
    t = np.arange(1, T + 1)
    ram, eidw, post = np.zeros(T), np.zeros(T), np.zeros(T)
    for e in eid:
        k = int(np.floor((e - np.datetime64('2021-01-04')).astype(int) / 7))
        if 0 <= k < T:
            eidw[k] = 1
        ram[max(k - 4, 0):max(min(k, T), 0)] = 1
        post[min(max(k + 1, 0), T):min(k + 5, T)] = 1
    return np.column_stack([np.ones(T), np.sin(2 * np.pi * t / 52.18), np.cos(2 * np.pi * t / 52.18),
                            np.sin(4 * np.pi * t / 52.18), np.cos(4 * np.pi * t / 52.18), ram, eidw, post])


def annihilator(X):
    return np.eye(X.shape[0]) - X @ np.linalg.solve(X.T @ X, X.T)


# ----------------------------------------------------------------------------- data generation
def simulate(A, T, rng, rho=0.3, burn=200, innov="gauss", A2=None, garch=(0.05, 0.10, 0.85), df=5, sigma=1.0):
    """Simulate Z_t = A Z_{t-1} [+ A2 Z_{t-2}] + a_t with Sigma = sigma^2 [(1-rho) I + rho 11'].

    innov: "gauss", "t" (Student-t with df degrees of freedom, unit variance) or "garch"
    (a GARCH(1,1) volatility common to all locations). A burn-in of `burn` periods is discarded."""
    N = A.shape[0]
    L = np.linalg.cholesky((1 - rho) * np.eye(N) + rho * np.ones((N, N)))
    n = T + burn
    if innov == "t":
        e = rng.standard_t(df, size=(n, N)) / np.sqrt(df / (df - 2))
    else:
        e = rng.standard_normal((n, N))
    if innov == "garch":                      # common conditional variance h_t, unconditional mean 1
        w, a, b = garch
        h = np.ones(n)
        for t in range(1, n):
            h[t] = w + a * np.mean((np.sqrt(h[t - 1]) * e[t - 1]) ** 2) + b * h[t - 1]
        e = e * np.sqrt(h)[:, None]
    a_ = sigma * (e @ L.T)
    Z = np.zeros((n, N))
    for t in range(2, n):
        Z[t] = A @ Z[t - 1] + a_[t]
        if A2 is not None:
            Z[t] += A2 @ Z[t - 2]
    return Z[burn:]


# ----------------------------------------------------------------------------- Wald test and its bootstrap
def wald_stats(Z, W0):
    """Equation-wise Wald statistics of H0: W = W0 on a (batch of) panel(s) Z (..., T, N).
    For linear restrictions W_i = (SSR_restricted - SSR_unrestricted) / sigma_ii (exact OLS identity);
    the restricted model is the fixed-W0 GSTAR regression of y_i on [x_i, X w0_i]."""
    X, Y = Z[..., :-1, :], Z[..., 1:, :]
    n, N = X.shape[-2], X.shape[-1]
    G = np.swapaxes(X, -1, -2) @ X
    Bh = np.linalg.solve(G, np.swapaxes(X, -1, -2) @ Y)          # column i = equation i
    Ru = Y - X @ Bh
    ssr_u = (Ru * Ru).sum(-2)
    V = X @ W0.T                                                  # column i = X w0_i
    sxx, sxv, svv = (X * X).sum(-2), (X * V).sum(-2), (V * V).sum(-2)
    sxy, svy = (X * Y).sum(-2), (V * Y).sum(-2)
    det = sxx * svv - sxv ** 2
    p0, p1 = (svv * sxy - sxv * svy) / det, (sxx * svy - sxv * sxy) / det
    Rr = Y - X * p0[..., None, :] - V * p1[..., None, :]
    ssr_r = (Rr * Rr).sum(-2)
    sig = ssr_u / (n - N)
    return (ssr_r - ssr_u) / sig, p0, p1, Rr


def bootstrap_test(Zhat, W0, M, rng, B=199, return_boot=False):
    """Two-stage residual bootstrap of the aggregate Wald statistic W_tot.

    Zhat : (T, N) residual panel of the exogenous stage; W0 : pre-specified matrix (the null);
    M    : annihilator of the exogenous design (re-estimates the exogenous stage on every bootstrap panel).
    Returns the bootstrap p-value and the equation-wise statistics (and the bootstrap draws of W_tot)."""
    Wi, p0, p1, res = wald_stats(Zhat, W0)
    A0 = np.diag(p0) + p1[:, None] * W0
    T, N = Zhat.shape
    tau = rng.integers(0, res.shape[0], size=(B, T))
    Zb = np.empty((B, T, N))
    Zb[:, 0] = Zhat[rng.integers(0, T, size=B)]
    for t in range(1, T):
        Zb[:, t] = Zb[:, t - 1] @ A0.T + res[tau[:, t]]
    Zb = M @ Zb                                                    # re-estimate the exogenous stage
    Wb = wald_stats(Zb, W0)[0].sum(-1)
    p = (1 + np.sum(Wb >= Wi.sum())) / (B + 1)
    return (p, Wi, Wb) if return_boot else (p, Wi)


# ----------------------------------------------------------------------------- the GSTARX-PW estimator
def proj_simplex_rows(V, mask):
    U = np.where(mask, V, -np.inf)
    S = -np.sort(-U, axis=1)
    fin = np.isfinite(S)
    css = np.cumsum(np.where(fin, S, 0.0), axis=1)
    k = np.arange(1, V.shape[1] + 1)
    cond = fin & (S - (css - 1) / k > 0)
    rho = cond.sum(1)
    theta = (css[np.arange(V.shape[0]), rho - 1] - 1) / rho
    W = np.maximum(U - theta[:, None], 0.0)
    W[~mask] = 0.0
    return W


def fixed_fit(Z, W0):
    X, Y = Z[:-1], Z[1:]
    V = X @ W0.T
    sxx, sxv, svv = (X * X).sum(0), (X * V).sum(0), (V * V).sum(0)
    sxy, svy = (X * Y).sum(0), (V * Y).sum(0)
    det = sxx * svv - sxv ** 2
    p0, p1 = (svv * sxy - sxv * svy) / det, (sxx * svy - sxv * sxy) / det
    R = Y - X * p0 - V * p1
    return p0, p1, (R * R).sum(0) / (X.shape[0] - 2)


def var_fit(Z):
    X, Y = Z[:-1], Z[1:]
    return np.linalg.solve(X.T @ X, X.T @ Y).T                    # A_hat, row i = equation i


def var_implied(A):
    N = A.shape[0]
    off = A * (1 - np.eye(N))
    return off / off.sum(1, keepdims=True)                          # plug-in ratio A_ij / sum_j A_ij, before projection


SETTINGS = dict(kappa_grid=(0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0), val_share=0.2,
                eps=1e-8, max_bcd=200, fista_tol=1e-10, max_fista=1000)


def pw_fit(Z, W0, kappa, s2=None, starts=("prior", "uniform", "var"), cfg=SETTINGS, return_starts=False):
    X, Y = Z[:-1], Z[1:]
    n, N = X.shape
    if s2 is None:
        s2 = fixed_fit(Z, W0)[2]
    lam = kappa * n * s2
    G = X.T @ X
    lmax = np.linalg.eigvalsh(G)[-1]
    mask = ~np.eye(N, dtype=bool)
    init = {"prior": W0, "uniform": mask / (N - 1.0), "var": None}
    out = []
    for st in starts:
        W = proj_simplex_rows(var_implied(var_fit(Z)) if st == "var" else init[st], mask)
        S_old = np.full(N, np.inf)
        for it in range(cfg["max_bcd"]):
            V = X @ W.T                                               # phi step: OLS on [x_i, X w_i]
            sxx, sxv, svv = (X * X).sum(0), (X * V).sum(0), (V * V).sum(0)
            sxy, svy = (X * Y).sum(0), (V * Y).sum(0)
            det = sxx * svv - sxv ** 2
            p0, p1 = (svv * sxy - sxv * svy) / det, (sxx * svy - sxv * sxy) / det
            C = (Y - X * p0).T @ X                                    # row i = c_i'
            Lip = 2 * (p1 ** 2 * lmax + lam) + 1e-12
            Wk, Yk, tk = W.copy(), W.copy(), 1.0                      # w step: FISTA on (10)
            for _ in range(cfg["max_fista"]):
                grad = 2 * (p1[:, None] ** 2 * (Yk @ G) + lam[:, None] * Yk) - 2 * (p1[:, None] * C + lam[:, None] * W0)
                Wn = proj_simplex_rows(Yk - grad / Lip[:, None], mask)
                tn = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
                Yk = Wn + ((tk - 1) / tn) * (Wn - Wk)
                done = np.max(np.abs(Wn - Wk)) < cfg["fista_tol"]
                Wk, tk = Wn, tn
                if done:
                    break
            W = Wk
            R = Y - X * p0 - (X @ W.T) * p1
            S = (R * R).sum(0) + lam * ((W - W0) ** 2).sum(1)
            if np.all(S_old - S <= cfg["eps"] * np.maximum(1.0, S)):
                break
            S_old = S
        out.append((S, W, p0, p1, it + 1))
    Ss = np.array([o[0] for o in out])
    best = Ss.argmin(0)
    W = np.array([out[best[i]][1][i] for i in range(N)])
    p0 = np.array([out[best[i]][2][i] for i in range(N)])
    p1 = np.array([out[best[i]][3][i] for i in range(N)])
    res = dict(W=W, phi0=p0, phi1=p1, A=np.diag(p0) + p1[:, None] * W, S=Ss.min(0))
    if return_starts:
        res["starts"] = out
    return res


def select_kappa(Z, W0, cfg=SETTINGS, return_curve=False):
    """Inner validation: estimate on the training window minus its last n_val periods, forecast those
    periods one step ahead, keep the kappa with the smallest RMSE and refit on the whole window."""
    T = Z.shape[0]
    nval = max(int(round(cfg["val_share"] * T)), 5)
    Zt = Z[:T - nval]
    s2 = fixed_fit(Zt, W0)[2]
    best, curve = (np.inf, None), []
    for k in cfg["kappa_grid"]:
        A = pw_fit(Zt, W0, k, s2=s2, cfg=cfg)["A"]
        err = Z[T - nval:] - Z[T - nval - 1:T - 1] @ A.T
        rmse = np.sqrt(np.mean(err ** 2))
        curve.append((k, rmse))
        if rmse < best[0]:
            best = (rmse, k)
    fit = pw_fit(Z, W0, best[1], cfg=cfg)
    return (best[1], fit, curve) if return_curve else (best[1], fit)


def mae_offdiag(Wh, W):
    N = W.shape[0]
    return np.abs(Wh - W)[~np.eye(N, dtype=bool)].mean()


def ljung_box_reject(R, lags=12, df_adj=2, alpha=0.05):
    from scipy.stats import chi2
    n = R.shape[0]
    Rc = R - R.mean(0)
    den = (Rc * Rc).sum(0)
    Q = np.zeros(R.shape[1])
    for k in range(1, lags + 1):
        r = (Rc[k:] * Rc[:-k]).sum(0) / den
        Q += r * r / (n - k)
    Q *= n * (n + 2)
    return np.mean(chi2.sf(Q, lags - df_adj) < alpha)


# ----------------------------------------------------------------------------- further candidate matrices
def uniform_matrix(N):
    """Uniform weights 1/(N-1) off the diagonal."""
    W = np.ones((N, N)) - np.eye(N)
    return W / (N - 1)


def knn_matrix(lat, lon, k=4, kind="greatcircle"):
    """Binary k-nearest-neighbour matrix, row-normalized."""
    d = distance_matrix(lat, lon, kind)
    np.fill_diagonal(d, np.inf)
    W = np.zeros_like(d)
    for i, js in enumerate(np.argsort(d, axis=1)[:, :k]):
        W[i, js] = 1.0
    return W / W.sum(1, keepdims=True)


def cross_correlation_matrix(Z):
    """Normalized absolute lag-one cross-correlations |corr(Z_i,t, Z_j,t-1)|, zero diagonal."""
    X, Y = Z[:-1], Z[1:]
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    C = np.abs(Yc.T @ Xc) / np.sqrt(np.outer((Yc * Yc).sum(0), (Xc * Xc).sum(0)))
    np.fill_diagonal(C, 0.0)
    return C / C.sum(1, keepdims=True)


def harmonic_design(T, period, K=2, intercept=True, start=1):
    """Intercept and K harmonics of a cycle of length `period`."""
    t = np.arange(start, start + T)
    cols = [np.ones(T)] if intercept else []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
    return np.column_stack(cols)


def exogenous_stage(Y, X):
    """Ordinary least squares of every series on the deterministic design: returns (B_hat, residual panel)."""
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return B, Y - X @ B


# ----------------------------------------------------------------------------- the Wald test in its textbook form
def restriction_matrices(W0):
    """For every equation i, the N-2 linear restrictions R_i a_i = 0 implied by W = W0
    (off-diagonal entries of row i of A proportional to row i of W0)."""
    N = W0.shape[0]
    out = []
    for i in range(N):
        others = np.delete(np.arange(N), i)
        j0 = others[np.argmax(W0[i, others])]
        rows = []
        for j in others:
            if j == j0:
                continue
            r = np.zeros(N)
            r[j], r[j0] = W0[i, j0], -W0[i, j]
            rows.append(r)
        out.append(np.array(rows))
    return out


def wald_stats_explicit(Z, W0):
    """Equation-wise Wald statistics computed with the restriction matrices,
    (R a)' [sigma_ii R (X'X)^{-1} R']^{-1} (R a). Numerically identical to `wald_stats`
    (which uses the SSR identity and is much faster)."""
    X, Y = Z[:-1], Z[1:]
    n, N = X.shape
    V = np.linalg.inv(X.T @ X)
    A = (V @ X.T @ Y).T
    res = Y - X @ A.T
    sig = (res * res).sum(0) / (n - N)
    stats = []
    for i, R in enumerate(restriction_matrices(W0)):
        ra = R @ A[i]
        stats.append(ra @ np.linalg.solve(sig[i] * R @ V @ R.T, ra))
    return np.array(stats)


def kappa_path(Z, W0, kappas, cfg=SETTINGS):
    """GSTARX-PW estimates along a grid of penalties: from the VAR (kappa = 0) to the prior (kappa -> infinity)."""
    s2 = fixed_fit(Z, W0)[2]
    return [pw_fit(Z, W0, k, s2=s2, cfg=cfg) for k in kappas]


def ljung_box(R, lags=12, df_adj=2):
    """Equation-wise Ljung-Box statistics and p-values of a residual panel R (T, N)."""
    from scipy.stats import chi2
    n = R.shape[0]
    Rc = R - R.mean(0)
    den = (Rc * Rc).sum(0)
    Q = np.zeros(R.shape[1])
    for k in range(1, lags + 1):
        r = (Rc[k:] * Rc[:-k]).sum(0) / den
        Q += r * r / (n - k)
    Q *= n * (n + 2)
    return Q, chi2.sf(Q, lags - df_adj)
