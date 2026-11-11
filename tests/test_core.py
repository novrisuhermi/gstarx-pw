import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import gstarx_pw as g

rng = np.random.default_rng(0)
N = 8
lat, lon = rng.uniform(-8.3, -6.7, N), rng.uniform(111, 114.4, N)
W0 = g.inverse_distance(lat, lon)


def panel(W, T=400, seed=1):
    r = np.random.default_rng(seed)
    A = np.diag(r.uniform(0.45, 0.65, N)) + r.uniform(0.2, 0.4, N)[:, None] * W
    return g.simulate(A, T, r), A


def test_identification_from_var_coefficients():
    W = g.sparse_matrix(N, rng)
    A = np.diag(rng.uniform(0.4, 0.6, N)) + rng.uniform(0.2, 0.4, N)[:, None] * W
    assert np.allclose(g.var_implied(A), W, atol=1e-12)


def test_simplex_projection():
    mask = ~np.eye(N, dtype=bool)
    P = g.proj_simplex_rows(rng.normal(size=(N, N)), mask)
    assert np.allclose(P.sum(1), 1) and (P >= 0).all() and np.allclose(np.diag(P), 0)
    assert np.allclose(g.proj_simplex_rows(W0, mask), W0)


def test_wald_identity():
    Z, _ = panel(g.sparse_matrix(N, rng))
    assert np.allclose(g.wald_stats(Z, W0)[0], g.wald_stats_explicit(Z, W0), rtol=1e-8)


def test_penalty_limits():
    Z, _ = panel(g.sparse_matrix(N, rng))
    assert np.abs(g.pw_fit(Z, W0, 1e6)["W"] - W0).max() < 1e-4          # kappa -> infinity: the prior
    ssr = lambda A: ((Z[1:] - Z[:-1] @ A.T) ** 2).sum()
    p0, p1, _ = g.fixed_fit(Z, W0)
    A_fix = np.diag(p0) + p1[:, None] * W0
    assert ssr(g.pw_fit(Z, W0, 0.0)["A"]) <= ssr(A_fix) + 1e-9           # kappa = 0 fits at least as well
    assert ssr(g.var_fit(Z)) <= ssr(g.pw_fit(Z, W0, 0.0)["A"]) + 1e-9    # ... and no better than the VAR


def test_bootstrap_detects_and_holds():
    X = g.design_generic(300)
    M = g.annihilator(X)
    Zt, _ = panel(g.sparse_matrix(N, np.random.default_rng(5)), T=300, seed=3)
    p_alt, _ = g.bootstrap_test(M @ Zt, W0, M, np.random.default_rng(1), B=99)
    Zn, _ = panel(W0, T=300, seed=4)
    p_null, _ = g.bootstrap_test(M @ Zn, W0, M, np.random.default_rng(2), B=99)
    assert 0 < p_alt <= 1 and 0 < p_null <= 1 and p_alt < p_null


def test_forecast_and_dm():
    A = np.diag(np.full(N, 0.5))
    f = g.forecast_path(A, np.ones(N), 3)
    assert f.shape == (3, N) and np.allclose(f[2], 0.125)
    e1, e2 = np.random.default_rng(3).normal(size=(2, 200))
    s, p = g.dm_test(e1, e2)
    assert np.isfinite(s) and 0 <= p <= 1
