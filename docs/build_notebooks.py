"""Builds and executes the tutorial notebooks in notebooks/. Run: python docs/build_notebooks.py"""
import os
import nbformat as nbf
from nbclient import NotebookClient

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NB = os.path.join(ROOT, "notebooks")
SETUP = """import sys, os
sys.path.insert(0, os.path.abspath('..'))   # or: pip install -e .  (from the repository root)
import numpy as np
import matplotlib.pyplot as plt
import gstarx_pw as g
from gstarx_pw.plotting import set_style, plot_links, plot_bootstrap, plot_heatmaps, OKABE_ITO as C
set_style()"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


notebooks = {}

# --------------------------------------------------------------------------------------------------- 01
notebooks["01_quickstart.ipynb"] = [
    md("""# 01 · Quick start

This notebook walks through the two tools of **GSTARX-PW** on a small simulated panel whose true
neighbourhood is *not* the one an analyst would assume:

1. the **Wald test** of a pre-specified weight matrix $W^0$, calibrated by the two-stage residual bootstrap;
2. the **GSTARX-PW estimator** of the row-normalized, non-negative weight matrix.

The model is $Y_t = B'x_t + Z_t$ (exogenous stage) and $Z_t = (\\Phi_0 + \\Phi_1 W) Z_{t-1} + a_t$ (space–time stage)."""),
    code(SETUP),
    md("## 1. Simulate a panel\nTwelve locations at random positions. The analyst assumes inverse-distance weights; the truth links every location to three randomly chosen partners."),
    code("""rng = np.random.default_rng(2026)
N, T = 12, 300
lat, lon = rng.uniform(-8.3, -6.7, N), rng.uniform(111.0, 114.4, N)
W0 = g.inverse_distance(lat, lon)          # the pre-specified matrix (the null hypothesis)
W_true = g.sparse_matrix(N, rng)           # the true neighbourhood
A = np.diag(rng.uniform(0.45, 0.65, N)) + rng.uniform(0.20, 0.40, N)[:, None] * W_true
X = g.design_generic(T)                    # intercept, two harmonics, a moving calendar block
Y = X @ rng.normal(0, 0.5, (X.shape[1], N)) + g.simulate(A, T, rng)
print(Y.shape)"""),
    md("## 2. Exogenous stage\nOrdinary least squares on the deterministic design; the space–time tools work on the residual panel $\\hat Z_t$."),
    code("""B, Zhat = g.exogenous_stage(Y, X)
M = g.annihilator(X)                       # lets the bootstrap re-estimate this stage on every bootstrap panel
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(Y[:, 0], color='#555555', lw=0.8, label='observed $Y_{1,t}$')
ax.plot(X @ B[:, 0], color=C[1], lw=2, label='exogenous part $\\\\hat B\\'x_t$')
ax.legend(); ax.set_xlabel('t'); plt.show()"""),
    md("## 3. Test the pre-specified matrix\nUnder $H_0: W = W^0$ the statistic $\\mathcal W_{\\rm tot}=\\sum_i \\mathcal W_i$ is calibrated by the two-stage bootstrap. As a control we also test the true matrix."),
    code("""p, Wi, Wb = g.bootstrap_test(Zhat, W0, M, rng, B=199, return_boot=True)
p_true, Wi_true, Wb_true = g.bootstrap_test(Zhat, W_true, M, rng, B=199, return_boot=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
plot_bootstrap(ax[0], Wb, Wi.sum(), p, title=f'inverse distance: p = {p:.3f}')
plot_bootstrap(ax[1], Wb_true, Wi_true.sum(), p_true, title=f'true matrix (control): p = {p_true:.3f}')
plt.show()"""),
    md("## 4. Estimate the neighbourhood\nThe penalty $\\kappa$ is chosen by one-step-ahead validation inside the training window ($\\kappa=0$: constrained VAR; $\\kappa\\to\\infty$: the prior)."),
    code("""kappa, fit, curve = g.select_kappa(Zhat, W0, return_curve=True)
print('selected kappa:', kappa)
print('MAE of the weights  prior: %.4f   GSTARX-PW: %.4f' % (g.mae_offdiag(W0, W_true), g.mae_offdiag(fit['W'], W_true)))
fig, ax = plot_heatmaps([W0, W_true, fit['W']], ['prior $W^0$', 'true $W$', 'GSTARX-PW'])
plt.show()"""),
    code("""fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, constrained_layout=True)
for a, (W, t, c) in zip(ax, [(W0, 'prior', C[0]), (W_true, 'truth', C[2]), (fit['W'], 'GSTARX-PW', C[1])]):
    plot_links(a, lat, lon, W, k=1, color=c, labels=[str(i) for i in range(N)]); a.set_title(t)
plt.show()
hits = sum(W_true[i, int(np.argmax(fit['W'][i]))] > 0 for i in range(N))
print(f'the strongest estimated link is a true link for {hits} of {N} locations')"""),
]

# --------------------------------------------------------------------------------------------------- 02
notebooks["02_workflow_simulated_prices.ipynb"] = [
    md("""# 02 · The full workflow on a simulated price panel

This notebook runs the complete workflow on a **simulated** weekly price panel for 26 generic cities, named
City A to City Z and placed at random: an annual price cycle, Ramadan and Eid al-Fitr effects, and a synthetic
'hub' neighbourhood that has nothing to do with distance. No real price data are used."""),
    code(SETUP + "\nimport pandas as pd\nfrom scipy.stats import chi2"),
    code("""d = g.simulate_price_panel(seed=2026)
Y, X, names = d['y'], d['X'], d['names']
east, north = d['east_km'], d['north_km']
weeks = d['weeks'].astype('datetime64[D]')
print(Y.shape, 'weeks x cities;  design columns:', d['columns'])
fig, ax = plt.subplots(figsize=(11, 3.4))
for k in range(4):
    ax.plot(weeks, np.exp(Y[:, k]) / 1000, lw=1.1, label=names[k], color=C[k])
for e in g.EID_AL_FITR:
    ax.axvline(np.datetime64(e), color='#999999', ls=':')
ax.set_ylabel('thousand IDR/kg'); ax.set_title('Simulated weekly prices (dotted: Eid al-Fitr)'); ax.legend(ncol=4, fontsize=8); plt.show()"""),
    md("## 1. Exogenous stage: annual cycle and calendar variation"),
    code("""B, Zhat = g.exogenous_stage(Y, X)
M = g.annihilator(X)
est = pd.DataFrame({'estimated (mean)': B.mean(1), 'true (mean)': d['B_true'].mean(1)}, index=d['columns'])
est.iloc[1:].round(3)"""),
    md("With only five Eid al-Fitr events in five years and strongly co-moving deviations across locations, the calendar coefficients carry noticeable estimation error; the annual cycle is estimated precisely. This uncertainty is exactly what the two-stage bootstrap of the Wald test takes into account."),
    md("## 2. Is a first-order space–time stage adequate? Residual diagnostics"),
    code("""p0, p1, _ = g.fixed_fit(Zhat, d['W0'])
A_fix = np.diag(p0) + p1[:, None] * d['W0']
kappa, fit = g.select_kappa(Zhat, d['W0'])
res = {'exogenous stage only': Zhat[1:], 'GSTARX, inverse distance': Zhat[1:] - Zhat[:-1] @ A_fix.T,
       'GSTARX-PW': Zhat[1:] - Zhat[:-1] @ fit['A'].T}
pd.DataFrame({k: {'locations rejecting white noise (Ljung-Box, 12 lags)': int((g.ljung_box(r)[1] < 0.05).sum()),
                  'residual s.d.': r.std()} for k, r in res.items()}).T.round(4)"""),
    md("## 3. Test the four standard weight matrices (and the truth as a control)"),
    code("""rng = np.random.default_rng(11)
crit = chi2.ppf(0.95, len(names) - 2)
cands = {'uniform': g.uniform_matrix(len(names)), 'inverse distance': d['W0'],
         '4-nearest neighbours': g.knn_matrix(north, east, 4, kind='euclidean'), 'cross-correlation': g.cross_correlation_matrix(Zhat),
         'true W (control)': d['W_true']}
rows = {}
for k, W in cands.items():
    p, Wi = g.bootstrap_test(Zhat, W, M, rng, B=199)
    rows[k] = {'W_tot': Wi.sum(), 'bootstrap p': p, 'equations rejected (chi2, 5%)': int((Wi > crit).sum())}
pd.DataFrame(rows).T.round(3)"""),
    md("## 4. The estimated neighbourhood\nCircles mark the four synthetic hubs."),
    code("""print('selected kappa:', kappa, '| MAE prior %.4f -> GSTARX-PW %.4f' % (g.mae_offdiag(d['W0'], d['W_true']), g.mae_offdiag(fit['W'], d['W_true'])))
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True, constrained_layout=True)
for a, (W, t, c) in zip(ax, [(d['W0'], 'prior', C[0]), (d['W_true'], 'truth', C[2]), (fit['W'], 'GSTARX-PW', C[1])]):
    plot_links(a, north, east, W, k=1, color=c, labels=[n.split()[1] for n in names], xlabel='east (km)', ylabel='north (km)'); a.set_title(t)
    a.scatter(east[d['hubs']], north[d['hubs']], s=110, facecolors='none', edgecolors='black', lw=1.4, zorder=4)
plt.show()
print('synthetic hubs:', [names[h] for h in d['hubs']])"""),
    md("## 5. Rolling-origin forecasts\nThe exogenous stage and all models are re-estimated at every origin (here every fourth week, with $\\kappa$ fixed at the value selected on the first window to keep the notebook fast)."),
    code("""k1, _ = g.select_kappa(g.exogenous_stage(Y[:156], X[:156])[1], d['W0'])
errs, _ = g.rolling_origin(Y, X, d['W0'], window=156, H=4, step=4, kappa=k1)
rmse = pd.DataFrame({m: np.sqrt((e ** 2).mean(axis=(0, 2))) for m, e in errs.items()}, index=[f'h={h}' for h in range(1, 5)])
print((rmse.div(rmse['exog'], axis=0)).round(3).to_string())
for other in ['fixed', 'var', 'exog']:
    s, pv = g.dm_test(np.sqrt((errs['pw'][:, 0] ** 2).mean(1)), np.sqrt((errs[other][:, 0] ** 2).mean(1)))
    print(f'DM test, GSTARX-PW vs {other:5s}: statistic {s:6.2f}, p = {pv:.3f}')"""),
    md("The choice of weight matrix matters little for point forecasts; it matters for **inference and interpretation** — which matrix is adequate, and which locations drive which."),
]

# --------------------------------------------------------------------------------------------------- 03
notebooks["03_public_data_irish_wind.ipynb"] = [
    md("""# 03 · Public data: daily wind speeds at 12 Irish stations

Daily average wind speeds (knots) at 12 meteorological stations in Ireland, 1961–1978 — the classic
space–time data set of Haslett and Raftery (1989, *JRSS C* 38, 1–50), distributed with the R package
**gstat** and downloaded here at run time. Following Haslett and Raftery we model square-root wind speeds."""),
    code(SETUP + "\nimport pandas as pd"),
    code("""w = g.load_irish_wind()
S = w['speeds'].loc['1976-01-01':'1978-12-31']
Y = np.sqrt(S.to_numpy())
print(S.shape, 'days x stations'); pd.DataFrame({'station': w['stations'], 'lat': w['lat'].round(2), 'lon': w['lon'].round(2)}, index=w['codes']).T"""),
    md("## 1. Exogenous stage: the annual cycle"),
    code("""X = g.harmonic_design(len(Y), period=365.25, K=3)
B, Zhat = g.exogenous_stage(Y, X)
M = g.annihilator(X)
fig, ax = plt.subplots(figsize=(11, 3.2))
for k, c in enumerate(['VAL', 'BIR', 'DUB']):
    j = w['codes'].index(c)
    ax.plot(S.index, Y[:, j], lw=0.5, alpha=0.5, color=C[k]); ax.plot(S.index, X @ B[:, j], lw=2, color=C[k], label=w['stations'][j])
ax.legend(ncol=3); ax.set_ylabel('sqrt(knots)'); plt.show()"""),
    md("## 2. Are standard weight matrices adequate?"),
    code("""rng = np.random.default_rng(7)
cands = {'inverse distance': g.inverse_distance(w['lat'], w['lon']), 'uniform': g.uniform_matrix(12),
         '2-nearest neighbours': g.knn_matrix(w['lat'], w['lon'], 2)}
rows = {}
for k, W in cands.items():
    p, Wi = g.bootstrap_test(Zhat, W, M, rng, B=199)
    rows[k] = {'W_tot': Wi.sum(), 'bootstrap p': p}
pd.DataFrame(rows).T.round(3)"""),
    md("## 3. What GSTARX-PW finds: the prevailing westerly flow\nEach arrow points from the neighbour $j$ with the largest estimated weight to the station $i$ it helps predict one day later."),
    code("""W0 = cands['inverse distance']
kappa, fit = g.select_kappa(Zhat, W0)
west = [w['lon'][int(np.argmax(fit['W'][i]))] < w['lon'][i] for i in range(12)]
print('selected kappa:', kappa, '| strongest links coming from the west:', sum(west), 'of 12')
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True, constrained_layout=True)
plot_links(ax[0], w['lat'], w['lon'], W0, k=1, color=C[0], arrows=True, labels=w['codes']); ax[0].set_title('assumed: inverse distance')
plot_links(ax[1], w['lat'], w['lon'], fit['W'], k=1, color=C[1], arrows=True, labels=w['codes']); ax[1].set_title('estimated: GSTARX-PW')
for a in ax: a.set_aspect(1 / np.cos(np.radians(53.4)))
plt.show()
pd.DataFrame({'strongest neighbour': [w['stations'][int(np.argmax(fit['W'][i]))] for i in range(12)],
              'weight': [fit['W'][i].max().round(2) for i in range(12)], 'from the west': west}, index=w['stations'])"""),
    md("Weather systems reach Ireland from the Atlantic, so yesterday's wind at the western coastal stations (Valentia, Belmullet) carries information about today's wind further east. The inverse-distance matrix cannot represent this asymmetry; the estimated matrix shows it directly."),
    md("## 4. Forecasts"),
    code("""errs, _ = g.rolling_origin(Y, X, W0, window=730, H=3, step=7, kappa=kappa)
rmse = pd.DataFrame({m: np.sqrt((e ** 2).mean(axis=(0, 2))) for m, e in errs.items()}, index=['h=1', 'h=2', 'h=3'])
rmse.div(rmse['exog'], axis=0).round(3)"""),
]

# --------------------------------------------------------------------------------------------------- 04
notebooks["04_simulation_studies.ipynb"] = [
    md("""# 04 · Simulation studies

A small size-and-power study, followed by the results of two larger simulation studies read from
`studies/results/` (run `studies/study_exact.py` and `studies/study_phi.py` to reproduce them in full)."""),
    code(SETUP + "\nimport json, glob, pandas as pd"),
    md("## 1. Size and power of the two-stage bootstrap test (N = 10, R = 100 per cell)"),
    code("""r0 = np.random.default_rng(7)
lat, lon = r0.uniform(-8.3, -6.7, 10), r0.uniform(111.0, 114.4, 10)
W0 = g.inverse_distance(lat, lon)
out = {}
for truth in ['prior', 'mixture', 'sparse']:
    for T in [100, 200]:
        rr = np.random.default_rng(T + len(truth)); M = g.annihilator(g.design_generic(T)); rej = 0
        for _ in range(100):
            Wt = {'prior': W0, 'sparse': g.sparse_matrix(10, rr), 'mixture': g.row_normalize(0.6 * W0 + 0.4 * g.sparse_matrix(10, rr))}[truth]
            A = np.diag(rr.uniform(0.45, 0.65, 10)) + rr.uniform(0.2, 0.4, 10)[:, None] * Wt
            rej += g.bootstrap_test(M @ g.simulate(A, T, rr), W0, M, rr, B=199)[0] <= 0.05
        out[(truth, T)] = rej / 100
pd.Series(out).unstack().round(2)"""),
    md("## 2. Size and power of the bootstrap test at N = 38, T = 156"),
    code("""E = {k: json.load(open(f'../studies/results/exact_{k}.json')) for k in ['gauss', 'garch', 't5', 'lag2a', 'lag2b', 'sparse', 'mixture']}
pd.DataFrame({v['design']: {'Ljung-Box rejections': v['lb12_share'], 'equation-wise rejection': v['eqwise_share'],
              'bootstrap rejection': v['boot_reject']} for v in E.values()}).T.round(3)"""),
    md("## 3. Weak spatial coefficients and starting values (N = 20, T = 200)"),
    code("""P = {c: json.load(open(f'../studies/results/phi_{c}.json')) for c in ['0.30', '0.20', '0.10', '0.05']}
pd.DataFrame({f'phi1 ~ {c}': {'prior': v['prior'], 'ratio (median)': v['raw_median'], 'VAR-implied': v['var'],
              'GSTARX-PW': v['pw'], 'median kappa': v['kappa_median'], 'sign flips': v['signflip'],
              'starts agree': v['agree']} for c, v in P.items()}).T.round(4)"""),
]

for name, cells in notebooks.items():
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": NB}}).execute()
    nbf.write(nb, os.path.join(NB, name))
    print("executed", name, flush=True)
