"""Example data: a simulated weekly price panel for generic cities and the public Irish wind data."""
import os
import re
import tempfile
import urllib.request
import numpy as np
from .core import inverse_distance, simulate, row_normalize

# dates of Eid al-Fitr (public calendar)
EID_AL_FITR = ["2021-05-13", "2022-05-02", "2023-04-21", "2024-04-10", "2025-03-31"]

GSTAT_WIND_URL = "https://raw.githubusercontent.com/cran/gstat/master/data/wind.rda"


def example_coordinates(N=26, seed=7, width_km=300.0, height_km=150.0):
    """N generic cities named City A, City B, ... at random planar positions (km east, km north)."""
    rng = np.random.default_rng(seed)
    names = [f"City {chr(65 + i)}" if i < 26 else f"City {chr(65 + i // 26 - 1)}{chr(65 + i % 26)}" for i in range(N)]
    return names, rng.uniform(0, width_km, N), rng.uniform(0, height_km, N)


def weekly_calendar_design(T, start="2021-01-04", eid_dates=EID_AL_FITR, harmonics=2, period=52.18, pre=4, post=4):
    """Weekly exogenous design: intercept, annual harmonics and Ramadan (pre), Eid-week and post-Eid dummies.
    Returns the design matrix, the week start dates and the column names."""
    weeks = np.datetime64(start) + 7 * np.arange(T)
    t = np.arange(1, T + 1)
    cols, names = [np.ones(T)], ["intercept"]
    for k in range(1, harmonics + 1):
        cols += [np.sin(2 * np.pi * k * t / period), np.cos(2 * np.pi * k * t / period)]
        names += [f"sin{k}", f"cos{k}"]
    ram, eid, aft = np.zeros(T), np.zeros(T), np.zeros(T)
    for e in np.array(eid_dates, dtype="datetime64[D]"):
        k = int((e - np.datetime64(start)).astype(int) // 7)
        if 0 <= k < T:
            eid[k] = 1
        ram[max(k - pre, 0):max(min(k, T), 0)] = 1
        aft[min(max(k + 1, 0), T):max(min(k + 1 + post, T), 0)] = 1
    cols += [ram, eid, aft]
    names += ["ramadan", "eid_week", "post_eid"]
    return np.column_stack(cols), weeks, names


def simulate_price_panel(seed=2026, N=26, T=261, rho=0.3, sigma=0.05):
    """A SIMULATED weekly price panel for N generic cities (City A, City B, ...); no real data.

    The weekly log price of every city is a seasonal and calendar profile (exogenous stage: two annual
    harmonics and Ramadan, Eid-week and post-Eid effects) plus a GSTAR(1;1) deviation whose true
    neighbourhood is a synthetic 'hub' structure: every city puts weight 0.5 on each of two of four
    randomly chosen hub cities, so the true weights are unrelated to the inverse-distance prior.
    Returns a dict; the modelled series is d["y"] (weekly log prices), positions are d["east_km"], d["north_km"]."""
    rng = np.random.default_rng(seed)
    names, east, north = example_coordinates(N)
    W0 = inverse_distance(north, east, kind="euclidean")
    hubs = rng.choice(N, 4, replace=False)
    Wt = np.zeros((N, N))
    for i in range(N):
        for h in rng.choice([h for h in hubs if h != i], 2, replace=False):
            Wt[i, h] += 0.5
    Wt = row_normalize(Wt)
    phi0, phi1 = rng.uniform(0.00, 0.20, N), rng.uniform(0.50, 0.70, N)
    A = np.diag(phi0) + phi1[:, None] * Wt
    Z = simulate(A, T, rng, rho=rho, sigma=sigma)
    X, weeks, cols = weekly_calendar_design(T)
    B = np.zeros((X.shape[1], N))
    B[0] = np.log(40000) + rng.normal(0, 0.08, N)          # price level
    B[1] = 0.12 + rng.normal(0, 0.02, N)                    # annual cycle
    B[2] = 0.10 + rng.normal(0, 0.02, N)
    B[3:5] = rng.normal(0, 0.03, (2, N))
    B[5] = 0.05 + rng.normal(0, 0.01, N)                    # Ramadan weeks
    B[6] = 0.10 + rng.normal(0, 0.02, N)                    # Eid week
    B[7] = -0.04 + rng.normal(0, 0.015, N)                  # four weeks after Eid
    y = X @ B + Z
    return dict(weeks=weeks, names=names, east_km=east, north_km=north, X=X, columns=cols, y=y, log_price=y,
                returns=np.diff(y, axis=0), W0=W0, W_true=Wt, A_true=A, phi0=phi0, phi1=phi1, hubs=hubs, B_true=B)


def _dms(s):
    m = re.match(r"(\d+)d(\d+)'(?:([\d.]+)\")?([NSEW])", s)
    deg, mins, secs, hemi = float(m.group(1)), float(m.group(2)), float(m.group(3) or 0), m.group(4)
    v = deg + mins / 60 + secs / 3600
    return -v if hemi in "SW" else v


def load_irish_wind(cache=None, url=GSTAT_WIND_URL):
    """Daily average wind speeds (knots) at 12 Irish meteorological stations, 1961-1978
    (Haslett and Raftery, 1989), as distributed with the R package gstat. Downloaded at run time.
    Returns a dict with a pandas DataFrame of speeds (dates x station codes) and the station coordinates."""
    import pandas as pd
    import pyreadr
    path = cache or os.path.join(tempfile.gettempdir(), "gstat_wind.rda")
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    r = pyreadr.read_r(path)
    wind, loc = r["wind"], r["wind.loc"]
    codes = [c for c in wind.columns if c not in ("year", "month", "day")]
    dates = pd.to_datetime(dict(year=1900 + wind["year"], month=wind["month"], day=wind["day"]))
    speeds = wind[codes].set_index(dates)
    loc = loc.assign(Code=loc["Code"].astype(str)).set_index("Code").loc[codes]
    return dict(speeds=speeds, codes=codes, stations=[str(s) for s in loc["Station"]],
                lat=np.array([_dms(str(s)) for s in loc["Latitude"]]),
                lon=np.array([_dms(str(s)) for s in loc["Longitude"]]))
