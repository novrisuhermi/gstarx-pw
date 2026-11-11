"""Writes README.md from the template below, taking every quoted number from docs/figures/summary.json.
Run: python docs/build_readme.py"""
import json, os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = json.load(open(os.path.join(ROOT, "docs", "figures", "summary.json")))
p, sp, r = S["sim_pvalues"], S["size_power"], S["rmse_ratio_h1"]
pct = lambda x: f"{100 * x:.0f}%"
VALUES = {"P_UNIF": f"{p['uniform']:.3f}", "P_INVD": f"{p['inverse distance']:.3f}", "P_KNN": f"{p['4-nearest neighbours']:.3f}",
          "P_CCOR": f"{p['cross- correlation']:.3f}", "P_TRUE": f"{p['true W (control)']:.3f}", "KAPPA": f"{S['sim_kappa']:g}",
          "MAE_PRIOR": f"{S['sim_mae_prior']:.4f}", "MAE_PW": f"{S['sim_mae_pw']:.4f}", "HITS": str(S["sim_hits"]), "N_SIM": str(S["sim_N"]),
          "R_VAR": f"{r['var']:.3f}", "R_FIX": f"{r['fixed']:.3f}", "R_PW": f"{r['pw']:.3f}", "WIND_T": str(S["wind_T"]),
          "WIND_P": f"{S['wind_p']:.3f}", "WIND_WEST": str(S["wind_west"]), "S1": pct(sp["prior"][0]), "S2": pct(sp["prior"][1]),
          "S3": pct(sp["prior"][2]), "POW": pct(sp["sparse"][2]), "MIX": pct(sp["mixture"][2])}
README = r'''# GSTARX-PW

**Penalized estimation and Wald testing of the spatial weight matrix in generalized space–time autoregressive models with exogenous variables (GSTARX)**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Space–time autoregressive models describe how a variable at one location responds to its own past and to the
past of its neighbours. Who counts as a neighbour is fixed by the **spatial weight matrix** $W$, and in practice
$W$ is simply *assumed* (inverse distance, uniform, nearest neighbours, cross-correlations) and chosen by comparing
forecast errors. It is never tested.

**GSTARX-PW tests the assumed matrix and, when it is wrong, estimates it.**

![Research framework](docs/figures/framework.png)

---

## Contents

1. [The idea in three steps](#the-idea-in-three-steps)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [A visual tour on simulated data](#a-visual-tour-on-simulated-data)
5. [Public data: the westerly wind over Ireland](#public-data-the-westerly-wind-over-ireland)
6. [How reliable is the test?](#how-reliable-is-the-test)
7. [Notebooks and examples](#notebooks-and-examples)
8. [API at a glance](#api-at-a-glance) · [Default settings](#default-settings) · [Simulation studies](#simulation-studies) · [Data](#data)

---

## The idea in three steps

The GSTARX(1;1) model has an exogenous stage and a space–time stage,

$$Y_t = B' x_t + Z_t, \qquad Z_t = (\Phi_0 + \Phi_1 W)\, Z_{t-1} + a_t ,$$

where $x_t$ holds deterministic regressors (seasonal harmonics, moving holidays such as Eid al-Fitr) and $W$ is
row-normalized with a zero diagonal.

1. **Identification.** $A = \Phi_0 + \Phi_1 W$ is the coefficient matrix of a VAR(1), with $A_{ii} = \phi_{0,i}$ and
   $A_{ij} = \phi_{1,i} w_{ij}$. So $W$ can be recovered from $A$ row by row, and any pre-specified $W^0$ is a set of
   $N-2$ linear restrictions per equation, **which can be tested**.
2. **The test.** A Wald test of those restrictions, with the aggregate statistic $\mathcal{W}_{tot} = \sum_i \mathcal{W}_i$
   calibrated by a residual bootstrap that re-estimates **both** stages on every bootstrap panel.
3. **The estimator.** GSTARX-PW solves, equation by equation, the penalized least-squares problem below, with
   $\lambda_i = \kappa (T-1) s_i^2$ and $w_i$ restricted to the simplex $\Delta$ (non-negative weights summing to one).
   A single scale-free penalty $\kappa$ moves the estimate from the constrained VAR ($\kappa = 0$) to the assumed matrix
   ($\kappa \to \infty$); it is chosen by one-step-ahead validation inside the training window. Computation: block
   coordinate descent, FISTA and an exact projection onto the simplex, from three starting values.

$$\min_{w_i \in \Delta} \; \lVert y_i - \phi_{0,i} x_i - \phi_{1,i} X_{-i} w_i \rVert^2 + \lambda_i \lVert w_i - w_i^0 \rVert^2$$

## Installation

```bash
git clone https://github.com/novrisuhermi/gstarx-pw.git
cd gstarx-pw
pip install -e ".[examples]"      # core: numpy + scipy; examples add matplotlib, pandas, pyreadr, jupyter
python -m pytest -q tests          # optional: 6 unit tests, about two seconds
```

## Quick start

```python
import numpy as np
import gstarx_pw as g

rng = np.random.default_rng(2026)
N, T = 12, 300
lat, lon = rng.uniform(-8.3, -6.7, N), rng.uniform(111.0, 114.4, N)
W0 = g.inverse_distance(lat, lon)                    # the matrix an analyst would assume
W_true = g.sparse_matrix(N, rng)                     # the truth: three partners, unrelated to distance
A = np.diag(rng.uniform(0.45, 0.65, N)) + rng.uniform(0.2, 0.4, N)[:, None] * W_true
X = g.design_generic(T)                              # intercept, harmonics, a moving calendar block
Y = X @ rng.normal(0, 0.5, (X.shape[1], N)) + g.simulate(A, T, rng)

B, Zhat = g.exogenous_stage(Y, X)                    # stage 1: deterministic part
p, Wi = g.bootstrap_test(Zhat, W0, g.annihilator(X), rng, B=199)    # test H0: W = W0
kappa, fit = g.select_kappa(Zhat, W0)                # GSTARX-PW estimate of W
```

```text
Wald test of the inverse-distance matrix: W_tot = 245.6, bootstrap p = 0.005
Wald test of the true matrix (control):   bootstrap p = 0.835
selected kappa = 0.1; MAE of the weights: prior 0.1403, GSTARX-PW 0.0773
the strongest estimated link is a true link for 12 of 12 locations
```

(`python examples/01_quickstart.py` prints exactly this.)

## A visual tour on simulated data

`g.simulate_price_panel()` generates a **synthetic** weekly price panel for 26 generic cities, named City A to
City Z and placed at random: an annual price cycle, Ramadan and Eid al-Fitr effects, and a neighbourhood in which
every city follows two of four *hub* cities, a structure that has nothing to do with distance. No real price data
are used anywhere in this repository.

### 1 · The data and the exogenous stage

![Simulated panel](docs/figures/simulated_panel.png)

Left: simulated prices in City A to City D (dotted lines: Eid al-Fitr). Right: the exogenous stage, two annual
harmonics plus Ramadan, Eid-week and post-Eid dummies fitted by least squares, removes the deterministic profile;
the space–time stage models what is left.

### 2 · Testing the assumed weight matrices

![Testing the candidates](docs/figures/test_candidates.png)

The four standard matrices are all rejected (bootstrap p = {{P_UNIF}}, {{P_INVD}}, {{P_KNN}} and {{P_CCOR}} for
uniform, inverse-distance, four-nearest-neighbour and cross-correlation weights), while the true matrix, tested as a
control, is not (p = {{P_TRUE}}). Centre and right: the bootstrap distribution of $\mathcal{W}_{tot}$ under $H_0$ and the
observed value (orange line).

### 3 · Estimating the neighbourhood

![Heat maps](docs/figures/heatmaps.png)

![Strongest links](docs/figures/links_map.png)

GSTARX-PW (here with $\kappa$ = {{KAPPA}}, selected by inner validation) recovers the hub columns that the
inverse-distance matrix cannot represent: the mean absolute error of the weights falls from {{MAE_PRIOR}} to
{{MAE_PW}}, and the strongest estimated link of {{HITS}} of the {{N_SIM}} cities points to one of their true hubs (circled).

### 4 · What the penalty does

![Penalty path](docs/figures/kappa_path.png)

Left: the validation error used to choose $\kappa$. Right: as $\kappa$ grows, the estimate moves from the data
(constrained VAR) to the prior. In this design the prior is wrong, so the distance to the truth is smallest at small
$\kappa$; when the prior is nearly right, validation selects a large $\kappa$ and the estimate stays close to it.

### 5 · Forecasting

![Forecasts](docs/figures/forecasts.png)

Rolling-origin forecasts, with every stage re-estimated at every origin. At one week ahead the RMSE relative to the
exogenous stage alone is {{R_VAR}} for the VAR(1), {{R_FIX}} for the GSTARX with inverse-distance weights and {{R_PW}}
for GSTARX-PW. The weight matrix matters little for point forecasts; it matters for **inference and interpretation**,
that is, for which matrix is adequate and which locations drive which.

## Public data: the westerly wind over Ireland

Daily wind speeds at 12 Irish meteorological stations (Haslett and Raftery, 1989), distributed with the R package
**gstat** and downloaded at run time by `g.load_irish_wind()`. We model square-root speeds for 1976–1978
({{WIND_T}} days) with an annual cycle in the exogenous stage.

![Irish wind series](docs/figures/irish_wind_series.png)

![Irish wind map](docs/figures/irish_wind_map.png)

The inverse-distance matrix is rejected (bootstrap p = {{WIND_P}}). The estimated matrix tells a physical story that
distance cannot: for **{{WIND_WEST}} of the 12 stations the strongest neighbour lies to the west**, mostly the Atlantic
stations Valentia and Belmullet. Weather systems reach Ireland from the west, so yesterday's wind on the west coast
predicts today's wind further east. The only exception is Valentia itself, the westernmost station.

## How reliable is the test?

### Size and power

![Size and power](docs/figures/size_power.png)

Two-stage bootstrap test, N = 10, 200 panels per point. Under $H_0$ the rejection rate is {{S1}}, {{S2}} and {{S3}} at
T = 100, 200 and 400 (nominal 5%); against a neighbourhood unrelated to distance the power rises to {{POW}}. A
matrix that departs only partly from the prior is much harder to detect ({{MIX}} at T = 400), so a non-rejection should
be read together with the estimate $\hat W$.

### Larger panels and misspecified innovations

![Calibration](docs/figures/calibration.png)

1000 simulated panels with N = 38 locations and T = 156 weeks, a weekly design with the actual Eid al-Fitr dates and
spatial coefficients between 0.25 and 0.45. The size stays close to 5% under GARCH and Student-t innovations and
is only moderately inflated by omitted second-order dynamics; power against moderate departures is limited at this
sample size.

### Weak spatial coefficients

![Weak coefficients](docs/figures/weak_coefficients.png)

Recovering $W$ from $A$ requires dividing by the spatial coefficient. As $\phi_1$ shrinks, the plug-in ratio breaks
down and the estimated $\phi_1$ changes sign in up to 38% of the equations; GSTARX-PW instead shrinks towards the prior,
and its error never exceeds the prior's.

## Notebooks and examples

| Notebook | What it shows |
|---|---|
| [`01_quickstart.ipynb`](notebooks/01_quickstart.ipynb) | Simulate, test, estimate and visualize in a few minutes |
| [`02_workflow_simulated_prices.ipynb`](notebooks/02_workflow_simulated_prices.ipynb) | The complete workflow on the simulated price panel: exogenous stage, residual diagnostics, tests of four standard matrices, estimated neighbourhood, rolling-origin forecasts and Diebold–Mariano tests |
| [`03_public_data_irish_wind.ipynb`](notebooks/03_public_data_irish_wind.ipynb) | Public data: Irish wind speeds and the westerly flow |
| [`04_simulation_studies.ipynb`](notebooks/04_simulation_studies.ipynb) | A small size-and-power study and the results of the larger simulation studies |

The same material as plain scripts: `examples/01_quickstart.py`, `02_simulated_prices.py`, `03_irish_wind.py`
and `04_forecasting.py`. The figures of this README are produced by `python docs/make_figures.py`.

## API at a glance

| Function | Purpose |
|---|---|
| `distance_matrix`, `inverse_distance`, `uniform_matrix`, `knn_matrix`, `cross_correlation_matrix` | distances (great-circle or Euclidean) and the standard candidate matrices |
| `exogenous_stage(Y, X)`, `annihilator(X)` | stage 1: least squares on the deterministic design |
| `harmonic_design`, `weekly_calendar_design`, `design_eid_weekly`, `design_generic` | deterministic designs (harmonics, Ramadan/Eid dummies) |
| `wald_stats(Z, W0)`, `wald_stats_explicit(Z, W0)` | equation-wise Wald statistics (fast SSR form / form with restriction matrices) |
| `bootstrap_test(Zhat, W0, M, rng, B=199)` | two-stage residual bootstrap p-value |
| `pw_fit(Z, W0, kappa)`, `select_kappa(Z, W0)`, `kappa_path(Z, W0, kappas)` | the GSTARX-PW estimator and the choice of $\kappa$ |
| `fixed_fit`, `var_fit`, `var_implied` | fixed-weight GSTAR, unrestricted VAR(1), plug-in ratio estimate of $W$ |
| `simulate(A, T, rng, ...)` | simulation with Gaussian, Student-t or GARCH innovations and an optional omitted lag |
| `forecast_path`, `rolling_origin`, `dm_test` | forecasting, rolling-origin evaluation, Diebold–Mariano test |
| `ljung_box` | residual diagnostics |
| `simulate_price_panel`, `example_coordinates`, `load_irish_wind` | example data (simulated / public) |

## Default settings

| Element | Setting |
|---|---|
| Starting values | prior $W^0$, uniform weights, row-normalized VAR estimate projected onto the simplex; per equation the lowest objective is kept |
| Block coordinate descent | stop when the relative decrease of the objective in a sweep is below $10^{-8}$, or after 200 sweeps |
| FISTA ($w$ step) | step $1/L$ with $L = 2 \lbrace \phi_{1,i}^2 \lambda_{\max}(X'X) + \lambda_i \rbrace$; stop when no weight changes by more than $10^{-10}$, or after 1000 iterations |
| Penalty | $\kappa \in \lbrace 0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 \rbrace$, one-step-ahead validation on the last 20% of the training window |
| Bootstrap | $B = 199$; rows of the residual matrix resampled; exogenous stage re-estimated on every bootstrap panel |
| Simulation | burn-in of 200 periods |

All settings live in `gstarx_pw.SETTINGS` and can be overridden through the `cfg` argument.

## Repository structure

```text
gstarx-pw/
├── gstarx_pw/            the package: core.py, forecasting.py, datasets.py, plotting.py
├── examples/             four runnable scripts
├── notebooks/            four executed notebooks
├── studies/              larger simulation studies, with their results (studies/results/*.json)
├── tests/                unit tests (pytest)
└── docs/                 make_figures.py, build_notebooks.py, build_readme.py, framework.tex and the README figures
```

## Simulation studies

```bash
python studies/study_exact.py gauss garch t5 lag2a lag2b sparse mixture   # size and power at N = 38, T = 156 (~8 min)
python studies/study_phi.py 0.30 0.20 0.10 0.05                           # weak spatial coefficients and starting values (~6 min)
python studies/validate.py && python studies/validate_pw.py                # size of the test and recovery of W in standard designs
```

Random seeds are fixed in every script; the outputs are in `studies/results/`.

## Data

Every example uses **simulated data** or the **public Irish wind data**; no other data are included in this repository.

## References

- Haslett, J., Raftery, A. E. (1989). Space-time modelling with long-memory dependence: assessing Ireland's wind power resource. *Journal of the Royal Statistical Society, Series C* 38(1), 1–50. https://doi.org/10.2307/2347679
- Pebesma, E. J. (2004). Multivariable geostatistics in S: the gstat package. *Computers & Geosciences* 30(7), 683–691. https://doi.org/10.1016/j.cageo.2004.03.012
- Borovkova, S., Lopuhaä, H. P., Ruchjana, B. N. (2008). Consistency and asymptotic normality of least squares estimators in generalized STAR models. *Statistica Neerlandica* 62(4), 482–508. https://doi.org/10.1111/j.1467-9574.2008.00391.x
- Beck, A., Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences* 2(1), 183–202. https://doi.org/10.1137/080716542
- Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. *Proceedings of ICML 2008*, 272–279. https://doi.org/10.1145/1390156.1390191

## License

MIT, see [LICENSE](LICENSE). Contact: Novri Suhermi, Department of Statistics, Institut Teknologi Sepuluh Nopember,
Surabaya, Indonesia (novri.suhermi@its.ac.id).
'''
for k, v in VALUES.items():
    README = README.replace("{{" + k + "}}", v)
assert "{{" not in README
open(os.path.join(ROOT, "README.md"), "w").write(README)
print("README.md written", len(README.split()), "words")
