"""Plotting helpers (matplotlib) with a consistent, colour-blind-safe style."""
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def set_style():
    """Grey panels, white grid lines and the colour-blind-safe Okabe-Ito palette."""
    plt.rcParams.update({"axes.facecolor": "#EAEAF2", "axes.edgecolor": "white", "axes.grid": True,
                         "grid.color": "white", "grid.linewidth": 1.0, "axes.axisbelow": True,
                         "axes.prop_cycle": cycler(color=OKABE_ITO), "font.size": 10,
                         "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
                         "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})


def plot_heatmaps(mats, titles, order=None, vmax=None, cmap="viridis", figsize=None):
    """Side-by-side heat maps of weight matrices (rows: location i, columns: neighbour j)."""
    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=figsize or (4.1 * k, 3.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    vmax = vmax or max(float(np.max(m)) for m in mats)
    for ax, M, t in zip(axes, mats, titles):
        M = M if order is None else M[np.ix_(order, order)]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(t)
        ax.set_xlabel("neighbour j")
        ax.set_ylabel("location i")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.85, label="weight $w_{ij}$")
    return fig, axes


def plot_links(ax, lat, lon, W, k=1, color="#0072B2", arrows=False, labels=None, lw=1.6,
               xlabel="longitude", ylabel="latitude"):
    """Draw the k strongest links of every location (line opacity proportional to the weight).
    With arrows=True each arrow points from the neighbour j to the location i that it influences."""
    N = len(lat)
    wmax = W.max()
    for i in range(N):
        for j in np.argsort(W[i])[::-1][:k]:
            if W[i, j] <= 0:
                continue
            a = 0.25 + 0.75 * W[i, j] / wmax
            if arrows:
                ax.annotate("", xy=(lon[i], lat[i]), xytext=(lon[j], lat[j]),
                            arrowprops=dict(arrowstyle="-|>", color=color, alpha=a, lw=lw, shrinkA=5, shrinkB=5))
            else:
                ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color=color, alpha=a, lw=lw)
    ax.scatter(lon, lat, s=22, color="#222222", zorder=3)
    if labels is not None:
        for x, y, s in zip(lon, lat, labels):
            ax.annotate(s, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


def plot_bootstrap(ax, Wb, W_obs, p, color="#0072B2", title=None):
    """Bootstrap distribution of the aggregate statistic W_tot under H0 with the observed value."""
    ax.hist(Wb, bins=30, color=color, alpha=0.75)
    ax.axvline(W_obs, color="#D55E00", lw=2.2)
    ax.set_xlabel(r"$\mathcal{W}_{\rm tot}$ under $H_0$ (bootstrap)")
    ax.set_ylabel("count")
    ax.set_title(title or f"bootstrap p = {p:.3f}")
    return ax
