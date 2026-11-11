"""Generates every figure of the README (simulated and public data only). Run: python docs/make_figures.py"""
import sys, os, json, glob, subprocess
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gstarx_pw as g
from gstarx_pw.plotting import set_style, plot_links, plot_bootstrap, OKABE_ITO as C

set_style()
OUT = os.path.join(ROOT, "docs", "figures"); os.makedirs(OUT, exist_ok=True)
summary = {}
save = lambda fig, name: (fig.savefig(os.path.join(OUT, name)), plt.close(fig))

# ---------------------------------------------------------------- 1. simulated price panel (generic cities)
d = g.simulate_price_panel(seed=2026)
Y, X, names, lat, lon = d["y"], d["X"], d["names"], d["north_km"], d["east_km"]
B, Zhat = g.exogenous_stage(Y, X); M = g.annihilator(X)
weeks = d["weeks"].astype("datetime64[D]")
eid = [np.datetime64(e) for e in g.EID_AL_FITR]
show = [0, 1, 2, 3]                      # City A to City D
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
for k, i in enumerate(show):
    ax[0].plot(weeks, np.exp(d["log_price"][:, i]) / 1000, lw=1.2, label=names[i], color=C[k])
for e in eid:
    ax[0].axvline(e, color="#999999", ls=":", lw=1); ax[1].axvline(e, color="#999999", ls=":", lw=1)
ax[0].set_ylabel("price (thousand IDR/kg)"); ax[0].set_title("Simulated weekly prices (dotted: Eid al-Fitr)"); ax[0].legend(ncol=2, fontsize=8)
i = show[0]
ax[1].plot(weeks, Y[:, i], color="#555555", lw=0.9, label="weekly log price")
ax[1].plot(weeks, X @ B[:, i], color=C[1], lw=1.8, label="exogenous stage: annual cycle + Ramadan/Eid")
ax[1].set_title(f"{names[i]}: what the exogenous stage removes"); ax[1].legend(fontsize=8)
save(fig, "simulated_panel.png")

# ---------------------------------------------------------------- 2. testing the candidate matrices
rng = np.random.default_rng(11)
cands = {"uniform": g.uniform_matrix(len(names)), "inverse\ndistance": d["W0"], "4-nearest\nneighbours": g.knn_matrix(lat, lon, 4, kind="euclidean"),
         "cross-\ncorrelation": g.cross_correlation_matrix(Zhat), "true W\n(control)": d["W_true"]}
pv, boot = {}, {}
for k_, W0_ in cands.items():
    p_, Wi_, Wb_ = g.bootstrap_test(Zhat, W0_, M, rng, B=199, return_boot=True); pv[k_] = p_; boot[k_] = (Wi_.sum(), Wb_, p_)
summary["sim_pvalues"] = {k_.replace("\n", " "): v for k_, v in pv.items()}
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
ax[0].bar(range(len(pv)), list(pv.values()), color=[C[1]] * 4 + [C[2]])
ax[0].axhline(0.05, color="black", ls="--", lw=1); ax[0].set_xticks(range(len(pv))); ax[0].set_xticklabels(list(pv), fontsize=8)
ax[0].set_ylabel("bootstrap p-value"); ax[0].set_title("Wald test of each candidate $W^0$")
for a, k_ in zip(ax[1:], ["inverse\ndistance", "true W\n(control)"]):
    Wobs, Wb_, p_ = boot[k_]
    plot_bootstrap(a, Wb_, Wobs, p_, color=C[0], title=k_.replace("\n", " ") + f": p = {p_:.3f}")
ax[1].annotate("observed", xy=(boot["inverse\ndistance"][0], 5), xytext=(-60, 40), textcoords="offset points",
               arrowprops=dict(arrowstyle="->"), fontsize=9)
save(fig, "test_candidates.png")

# ---------------------------------------------------------------- 3. estimation: heat maps and maps
kap, fit, curve = g.select_kappa(Zhat, d["W0"], return_curve=True)
summary.update(sim_N=len(names), sim_kappa=kap, sim_mae_prior=g.mae_offdiag(d["W0"], d["W_true"]), sim_mae_pw=g.mae_offdiag(fit["W"], d["W_true"]),
               sim_hits=int(sum(d["W_true"][i, int(np.argmax(fit["W"][i]))] > 0 for i in range(len(names)))))
order = np.argsort(lon)
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
vmax = max(d["W_true"].max(), fit["W"].max())
for a, (M_, t) in zip(ax, [(d["W0"], "prior $W^0$ (inverse distance)"), (d["W_true"], "true $W$ (synthetic hubs)"),
                           (fit["W"], f"GSTARX-PW estimate ($\\kappa$ = {kap:g})")]):
    im = a.imshow(M_[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=vmax); a.set_title(t); a.grid(False)
    a.set_xlabel("neighbour j (west to east)"); a.set_ylabel("location i (west to east)")
fig.colorbar(im, ax=ax, shrink=0.85, label="weight")
save(fig, "heatmaps.png")
fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), sharey=True, constrained_layout=True)
for a, (M_, t, c) in zip(ax, [(d["W0"], "prior: strongest link of each location", C[0]),
                              (d["W_true"], "truth: strongest link of each location", C[2]),
                              (fit["W"], "GSTARX-PW: strongest link of each location", C[1])]):
    plot_links(a, lat, lon, M_, k=1, color=c, xlabel="east (km)", ylabel="north (km)"); a.set_title(t, fontsize=10)
    a.scatter(lon[d["hubs"]], lat[d["hubs"]], s=110, facecolors="none", edgecolors="black", lw=1.4, zorder=4)
save(fig, "links_map.png")

# ---------------------------------------------------------------- 4. the penalty path
ks = [0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1000]
path = g.kappa_path(Zhat, d["W0"], ks)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True)
kk = [max(k_, 0.003) for k_, _ in curve]
ax[0].semilogx(kk, [r for _, r in curve], "o-", color=C[0]); ax[0].axvline(max(kap, 0.003), color=C[1], ls="--")
ax[0].set_xlabel(r"$\kappa$ (0 drawn at the left edge)"); ax[0].set_ylabel("validation RMSE"); ax[0].set_title("Choosing the penalty by inner validation")
ax[1].semilogx([max(k_, 0.003) for k_ in ks], [g.mae_offdiag(f["W"], d["W_true"]) for f in path], "o-", color=C[2], label="distance to the truth")
ax[1].semilogx([max(k_, 0.003) for k_ in ks], [g.mae_offdiag(f["W"], d["W0"]) for f in path], "s-", color=C[4], label="distance to the prior")
ax[1].axvline(max(kap, 0.003), color=C[1], ls="--"); ax[1].set_xlabel(r"$\kappa$"); ax[1].set_ylabel(r"mean absolute difference of $\hat W$")
ax[1].set_title(r"From the VAR ($\kappa$=0) to the prior ($\kappa\to\infty$)"); ax[1].legend(fontsize=8)
save(fig, "kappa_path.png")

# ---------------------------------------------------------------- 5. size and power (small Monte Carlo)
Ts, R = [100, 200, 400], 200
r0 = np.random.default_rng(7); latm, lonm = r0.uniform(-8.3, -6.7, 10), r0.uniform(111.0, 114.4, 10); W0m = g.inverse_distance(latm, lonm)
rates = {}
for truth in ["prior", "mixture", "sparse"]:
    rates[truth] = []
    for T in Ts:
        rr = np.random.default_rng(1000 + T + len(truth)); Mm = g.annihilator(g.design_generic(T)); rej = 0
        for _ in range(R):
            Wt = {"prior": W0m, "sparse": g.sparse_matrix(10, rr), "mixture": g.row_normalize(0.6 * W0m + 0.4 * g.sparse_matrix(10, rr))}[truth]
            A = np.diag(rr.uniform(0.45, 0.65, 10)) + rr.uniform(0.2, 0.4, 10)[:, None] * Wt
            rej += g.bootstrap_test(Mm @ g.simulate(A, T, rr), W0m, Mm, rr, B=199)[0] <= 0.05
        rates[truth].append(rej / R)
summary["size_power"] = rates
fig, ax = plt.subplots(figsize=(5.6, 3.6), constrained_layout=True)
for k_, (truth, lab) in enumerate([("prior", "H0 true (size)"), ("mixture", "partial departure"), ("sparse", "unrelated to distance")]):
    ax.plot(Ts, rates[truth], "o-", color=C[k_], label=lab)
ax.axhline(0.05, color="black", ls=":", lw=1); ax.set_xscale("log"); ax.set_xticks(Ts); ax.set_xticklabels(Ts); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("T"); ax.set_ylabel("rejection rate at 5%"); ax.set_ylim(0, 1.02); ax.set_title(f"Two-stage bootstrap test, N = 10, R = {R}"); ax.legend(fontsize=8)
save(fig, "size_power.png")

# ---------------------------------------------------------------- 6. results of the simulation studies in studies/
res = os.path.join(ROOT, "studies", "results")
P = {float(c): json.load(open(os.path.join(res, f"phi_{c}.json"))) for c in ["0.30", "0.20", "0.10", "0.05"]}
cs = sorted(P)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True)
for k_, (key, lab) in enumerate([("prior", "prior $W^0$"), ("var", "VAR-implied (projected)"), ("pw", "GSTARX-PW")]):
    ax[0].plot(cs, [P[c][key] for c in cs], "o-", color=C[k_], label=lab)
ax[0].set_xlabel(r"typical spatial coefficient $\phi_1$"); ax[0].set_ylabel(r"MAE of $\hat W$"); ax[0].set_title("Weak spatial coefficients (N=20, T=200)"); ax[0].legend(fontsize=8)
ax[1].plot(cs, [P[c]["raw_median"] for c in cs], "o-", color=C[3], label="plug-in ratio (median MAE)")
ax[1].set_xlabel(r"typical spatial coefficient $\phi_1$"); ax[1].set_ylabel("median MAE", color=C[3]); ax[1].set_title("The plug-in ratio breaks down")
a2 = ax[1].twinx(); a2.bar(cs, [100 * P[c]["signflip"] for c in cs], width=0.02, alpha=0.4, color=C[4]); a2.set_ylabel(r"sign flips of $\hat\phi_1$ (%)", color=C[4]); a2.grid(False)
save(fig, "weak_coefficients.png")
E = {k_: json.load(open(os.path.join(res, f"exact_{k_}.json"))) for k_ in ["gauss", "garch", "t5", "lag2a", "lag2b", "sparse", "mixture"]}
labs = ["Gaussian", "GARCH", "Student-t", "omitted lag 2\n(as in data)", "omitted lag 2\n(much stronger)", "sparse\nalternative", "mixture\nalternative"]
fig, ax = plt.subplots(figsize=(10, 3.6), constrained_layout=True)
vals = [100 * E[k_]["boot_reject"] for k_ in E]
ax.bar(range(7), vals, color=[C[0]] * 5 + [C[2]] * 2)
for x, v in enumerate(vals): ax.text(x, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9)
ax.axhline(5, color="black", ls=":", lw=1); ax.set_xticks(range(7)); ax.set_xticklabels(labs, fontsize=8.5)
ax.set_ylabel("bootstrap rejection rate (%)"); ax.set_ylim(0, 40)
ax.set_title("Bootstrap test at N = 38, T = 156: rejection rates (1000 panels under $H_0$, 500 under alternatives)")
save(fig, "calibration.png")

# ---------------------------------------------------------------- 7. forecasting
errs, _ = g.rolling_origin(Y, X, d["W0"], window=156, H=4, step=2, kappa=kap)
rm = {m: np.sqrt((e ** 2).mean(axis=(0, 2))) for m, e in errs.items()}
summary["rmse_ratio_h1"] = {m: float(rm[m][0] / rm["exog"][0]) for m in rm}
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
for k_, (m, lab) in enumerate([("var", "VAR(1)"), ("fixed", "GSTARX, inverse distance"), ("pw", "GSTARX-PW")]):
    ax[0].plot(range(1, 5), rm[m] / rm["exog"], "o-", color=C[k_], label=lab)
ax[0].axhline(1, color="black", ls=":", lw=1); ax[0].set_xticks(range(1, 5)); ax[0].set_xlabel("horizon h (weeks)")
ax[0].set_ylabel("RMSE / RMSE(exogenous stage only)"); ax[0].set_title(f"Rolling-origin accuracy ({errs['pw'].shape[0]} origins)"); ax[0].legend(fontsize=8)
i = show[0]; tt = weeks[156:156 + 2 * errs["pw"].shape[0]:2]
ax[1].plot(tt, np.exp(Y[156:156 + 2 * len(tt):2, i]) / 1000, "o-", ms=3, color="#444444", lw=1, label="actual")
ax[1].plot(tt, np.exp(Y[156:156 + 2 * len(tt):2, i] - errs["pw"][:, 0, i]) / 1000, "s-", ms=3, color=C[1], lw=1.2, label="GSTARX-PW, one week ahead")
ax[1].set_ylabel("price (thousand IDR/kg)")
ax[1].set_title(f"{names[i]}: one-step-ahead forecasts"); ax[1].legend(fontsize=8)
save(fig, "forecasts.png")

# ---------------------------------------------------------------- 8. public data: Irish wind
w = g.load_irish_wind()
S = w["speeds"].loc["1976-01-01":"1978-12-31"]; Yw = np.sqrt(S.to_numpy()); Xw = g.harmonic_design(len(Yw), period=365.25, K=3)
Bw, Zw = g.exogenous_stage(Yw, Xw); W0w = g.inverse_distance(w["lat"], w["lon"])
pw_, Wiw, Wbw = g.bootstrap_test(Zw, W0w, g.annihilator(Xw), np.random.default_rng(7), B=199, return_boot=True)
kw, fw = g.select_kappa(Zw, W0w)
west = sum(w["lon"][int(np.argmax(fw["W"][i]))] < w["lon"][i] for i in range(12))
summary.update(wind_T=len(Yw), wind_p=pw_, wind_kappa=kw, wind_west=int(west), wind_Wtot=float(Wiw.sum()))
fig, ax = plt.subplots(figsize=(12, 3.4), constrained_layout=True)
for k_, c in enumerate(["VAL", "BIR", "DUB"]):
    j = w["codes"].index(c); ax.plot(S.index, Yw[:, j], lw=0.6, alpha=0.55, color=C[k_])
    ax.plot(S.index, Xw @ Bw[:, j], lw=2.2, color=C[k_], label=f"{w['stations'][j]} (annual cycle)")
ax.set_ylabel(r"$\sqrt{\mathrm{wind\ speed}}$ (knots$^{1/2}$)"); ax.set_title("Daily wind speeds, 1976-1978 (public data, Haslett and Raftery 1989)"); ax.legend(fontsize=8, ncol=3)
save(fig, "irish_wind_series.png")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.0), sharey=True, constrained_layout=True)
plot_links(ax[0], w["lat"], w["lon"], W0w, k=1, color=C[0], arrows=True, labels=w["codes"])
ax[0].set_title("Assumed: inverse distance (strongest link)", fontsize=10)
plot_links(ax[1], w["lat"], w["lon"], fw["W"], k=1, color=C[1], arrows=True, labels=w["codes"])
ax[1].set_title(f"Estimated by GSTARX-PW: {west} of 12 strongest links come from the west", fontsize=10)
for a in ax: a.set_aspect(1 / np.cos(np.radians(53.4)))
fig.suptitle(f"Irish wind: the inverse-distance matrix is rejected (bootstrap p = {pw_:.3f}); arrows point from neighbour to location", fontsize=11)
save(fig, "irish_wind_map.png")

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
