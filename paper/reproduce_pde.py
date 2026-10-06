"""Regenerate every PDE number reported in the manuscript in one run.

Usage (from the repository root):

    python paper/reproduce_pde.py            # about 25 to 40 minutes on a laptop
    python paper/reproduce_pde.py --quick    # skips the slow beta* sensitivity sweep

All random draws use fixed seeds, so the output is identical on every run.
Results are printed next to the section of the paper they appear in and are
written to ``paper/results/pde_results.json``; ``paper/make_figures.py`` draws
Figures 8, 9 and 10 from that file.

The one analysis that needs patient data (the correlation between severity and
simulated crisis count across the 90 patients, Section 5.5) runs only if the
restricted Baylor spreadsheet is present in ``data/``. No patient data is
stored in this repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_model import (BASE, BETA_MAX, BETA_MIN, critical_amplitude, deterministic_field, grid,  # noqa: E402
                       scalar_model, simulate, source)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
STRATA = {"high": 0.0167, "moderate": 0.0158, "mild": 0.0139}
COHORT_BETAS = [0.0121, 0.0139, 0.0158, 0.0167, 0.0176]

# Seeds (fixed; listed here so every number can be traced)
SEED_STRATA = 3000        # Section 5.5, 200 realisations per stratum
SEED_PDE_BASE = 11        # Section 5.7 / Fig 9a, PDE, 200 realisations
SEED_PEAK = 12            # scalar model of the peak
SEED_AVG = 13             # scalar model of the average
SEED_EQUAL = 21           # equal total source test
SEED_FIG8B = 700          # Fig 8b, seed 700 + i, 120 realisations
SEED_CORR = 5000          # per patient, seed 5000 + i, 20 realisations
SEED_ELAS = 1             # elasticities, 40 realisations, base beta = 0.0167
SEED_REC_REF = 0          # recovery reference table, seed 0 + j, 100 realisations
SEED_REC_TRUE = 0         # draws of the 60 true beta values
SEED_REC_PAT = 1000       # one timeline per synthetic patient, seed 1000 + i


def log(msg):
    print(msg, flush=True)


def strata():
    res = {}
    for name, b in STRATA.items():
        c, f = simulate(b, 200, SEED_STRATA)
        res[name] = {"beta": b, "mean": float(c.mean()), "sd": float(c.std()), "first_day": float(np.nanmean(f))}
    c, _ = simulate(0.0121, 200, SEED_STRATA)
    res["lowest"] = {"beta": 0.0121, "fraction_with_crisis": float((c > 0).mean()), "mean": float(c.mean()),
                     "max_count": int(c.max())}
    return res


def baselines():
    p = BASE
    x, _ = grid(p)
    sbar = float(source(p, x).mean())
    out = {"S_bar": sbar, "beta": COHORT_BETAS, "pde": [], "peak": [], "average": []}
    for b in COHORT_BETAS:
        out["pde"].append(float(simulate(b, 200, SEED_PDE_BASE)[0].mean()))
        out["peak"].append(float(scalar_model(b, 200, SEED_PEAK, p.sigma_eta).mean()))
        out["average"].append(float(scalar_model(b * sbar, 200, SEED_AVG, p.sigma_eta * sbar).mean()))
    return out


def equal_source():
    p = BASE
    x, _ = grid(p)
    total0 = source(p, x).sum()
    out = {"factor": [0.5, 0.75, 1.0, 1.5, 2.0], "mean": []}
    for f in out["factor"]:
        q = p.with_(sigma=p.sigma * f)
        b = 0.0158 * total0 / source(q, x).sum()
        out["mean"].append(float(simulate(b, 200, SEED_EQUAL, q)[0].mean()))
    return out


def fig8b():
    betas = np.linspace(BETA_MIN, BETA_MAX, 5)
    mean, sd = [], []
    for i, b in enumerate(betas):
        c, _ = simulate(float(b), 120, SEED_FIG8B + i)
        mean.append(float(c.mean()))
        sd.append(float(c.std()))
    return {"beta": betas.tolist(), "mean": mean, "sd": sd}


def elasticities():
    b0, K = 0.0167, 40
    n0 = float(simulate(b0, K, SEED_ELAS)[0].mean())
    names = ["r_star", "delta", "sigma", "alpha", "x0", "kappa", "D", "sigma_eta"]
    out = {"base_beta": b0, "n0": n0, "S": {}, "by_perturbation": {}}
    for k in names:
        vals = []
        for pct in (-0.2, -0.1, 0.1, 0.2):
            q = BASE.with_(**{k: getattr(BASE, k) * (1 + pct)})
            n = float(simulate(b0, K, SEED_ELAS, q)[0].mean())
            vals.append(((n - n0) / n0) / pct)
        out["S"][k] = float(np.mean(vals))
        out["by_perturbation"][k] = vals
    return out


def beta_star_sensitivity():
    b0 = critical_amplitude(BASE)
    out = {"beta_star": b0, "shift": {}}
    for k in ["r_star", "delta", "alpha", "sigma", "D", "x0"]:
        row = {}
        for pct in (-0.2, -0.1, 0.1, 0.2):
            row[f"{int(pct * 100):+d}%"] = critical_amplitude(BASE.with_(**{k: getattr(BASE, k) * (1 + pct)}))
        out["shift"][k] = row
    return out


def recovery():
    grid_b = np.round(np.arange(0.0115, 0.0185 + 1e-9, 0.00025), 5)
    ref = []
    for j, b in enumerate(grid_b):
        c, f = simulate(float(b), 100, SEED_REC_REF + j)
        f = np.where(np.isnan(f), BASE.T, f)
        ref.append((f.mean(), f.std() + 1e-9, c.mean(), c.std() + 1e-9))
    ref = np.array(ref)
    true_b = np.random.default_rng(SEED_REC_TRUE).uniform(BETA_MIN, BETA_MAX, 60)
    est = []
    for i, b in enumerate(true_b):
        c, f = simulate(float(b), 1, SEED_REC_PAT + i)
        t1 = BASE.T if np.isnan(f[0]) else f[0]
        d = ((t1 - ref[:, 0]) / ref[:, 1]) ** 2 + ((c[0] - ref[:, 2]) / ref[:, 3]) ** 2
        est.append(grid_b[np.argmin(d)])
    est = np.array(est)
    bstar = 0.01273
    return {"true": true_b.tolist(), "estimated": est.tolist(),
            "r": float(np.corrcoef(true_b, est)[0, 1]),
            "mae": float(np.abs(est - true_b).mean()),
            "median_rel_err_pct": float(np.median(np.abs(est - true_b) / true_b) * 100),
            "bias": float((est - true_b).mean()),
            "regime_correct": int(((est > bstar) == (true_b > bstar)).sum())}


def convergence():
    T = BASE.T
    beta = 0.0158
    ref_t = deterministic_field(beta, BASE.dt / 128, T)
    dts = [0.5, 0.25, 0.125, 0.0625, 0.03125]
    et = [float(np.abs(deterministic_field(beta, d, T) - ref_t).max()) for d in dts]
    ot = [float(np.log2(et[i] / et[i + 1])) for i in range(len(et) - 1)]
    # spatial: N = 65 ... 1040 against N = 2080, compared at the coarse nodes
    Ns = [65, 130, 260, 520, 1040]
    p_ref = BASE.with_(N=2080)
    x_ref = np.linspace(0, 1, 2080)
    ref_x = deterministic_field(beta, BASE.dt, T, p_ref)
    ex = []
    for n in Ns:
        xn = np.linspace(0, 1, n)
        ex.append(float(np.abs(deterministic_field(beta, BASE.dt, T, BASE.with_(N=n)) - np.interp(xn, x_ref, ref_x)).max()))
    ox = [float(np.log2(ex[i] / ex[i + 1])) for i in range(len(ex) - 1)]
    # mass conservation under pure diffusion and the amplification factor
    q = BASE.with_(alpha=0.0, delta=0.0)
    x, ab = grid(q)
    from scipy.linalg import solve_banded
    r = source(q, x).copy()
    m0 = r.sum()
    for _ in range(int(q.T / q.dt)):
        r = solve_banded((1, 1), ab, r)
    drift = float(abs(r.sum() - m0) / m0)
    lam = q.D * q.dt * q.N**2
    theta = np.linspace(0, np.pi, 2001)
    gmax = float(np.max(np.abs(1.0 / (1.0 + 4 * lam * np.sin(theta / 2) ** 2))))
    return {"temporal_errors": et, "temporal_orders": ot, "spatial_errors": ex, "spatial_orders": ox,
            "mass_drift": drift, "max_amplification": gmax}


def correlations():
    try:
        sys.path.insert(0, HERE)
        from cohort_data import load_tdd_term_counts
        n = load_tdd_term_counts()
    except FileNotFoundError as e:
        log(f"  skipped (restricted data not present: {e})")
        return None
    sev = (n - n.min()) / (n.max() - n.min())
    beta = BETA_MIN + (BETA_MAX - BETA_MIN) * sev
    cnt, first = [], []
    for i, b in enumerate(beta):
        c, f = simulate(float(b), 20, SEED_CORR + i)
        cnt.append(c.mean())
        first.append(np.nanmean(f) if np.isfinite(f).any() else BASE.T)
    return {"r_count": float(np.corrcoef(sev, cnt)[0, 1]), "r_first": float(np.corrcoef(sev, first)[0, 1]),
            "count_range": [float(min(cnt)), float(max(cnt))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the beta* sensitivity sweep")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    log(f"r_rest = {BASE.r_rest}  (paper Table 2)")
    res = {"parameters": BASE.__dict__}
    steps = [("strata", strata), ("baselines", baselines), ("equal_source", equal_source), ("fig8b", fig8b),
             ("elasticities", elasticities), ("recovery", recovery), ("convergence", convergence),
             ("correlations", correlations)]
    if not args.quick:
        steps.insert(0, ("beta_star", beta_star_sensitivity))
    for name, fn in steps:
        t0 = time.time()
        log(f"[{name}] running ...")
        res[name] = fn()
        log(f"[{name}] done in {time.time() - t0:.0f} s")
        fname = "pde_results_quick.json" if args.quick else "pde_results.json"
        with open(os.path.join(OUT, fname), "w") as fh:
            json.dump(res, fh, indent=1)
    report(res)


def report(res):
    log("\n================ Numbers as reported in the manuscript ================")
    if "beta_star" in res:
        b = res["beta_star"]
        log(f"Sec 4.5  beta* = {b['beta_star']:.5f}")
        rs = b["shift"]["r_star"]
        log(f"Sec 5.8  beta* with r* +20% = {rs['+20%']:.4f}, r* -20% = {rs['-20%']:.4f}")
        x0 = b["shift"]["x0"]
        log(f"Sec 5.8  x0 +-10% shifts beta* by at most "
            f"{max(abs(x0['-10%'] / b['beta_star'] - 1), abs(x0['+10%'] / b['beta_star'] - 1)) * 100:.1f} percent")
    s = res["strata"]
    log(f"Sec 5.5  high {s['high']['mean']:.2f} +- {s['high']['sd']:.2f}, first ~day {s['high']['first_day']:.0f}; "
        f"moderate {s['moderate']['mean']:.2f}, first ~day {s['moderate']['first_day']:.0f}; "
        f"mild {s['mild']['mean']:.2f} +- {s['mild']['sd']:.2f}, first ~day {s['mild']['first_day']:.0f}")
    lo = s["lowest"]
    log(f"Sec 5.6  beta=0.0121: {lo['fraction_with_crisis'] * 100:.0f}% of realisations with a crisis, "
        f"mean {lo['mean']:.2f}, max per realisation {lo['max_count']}")
    bl = res["baselines"]
    log(f"Sec 5.7  scalar peak {min(bl['peak']):.1f} to {max(bl['peak']):.1f}; scalar average "
        f"{min(bl['average']):.1f} to {max(bl['average']):.1f}; PDE {min(bl['pde']):.1f} to {max(bl['pde']):.1f}")
    log("Sec 5.7  equal total source: " + ", ".join(f"{v:.1f}" for v in res["equal_source"]["mean"]))
    r = res["recovery"]
    log(f"Sec 5.7  recovery r = {r['r']:.3f}, MAE = {r['mae']:.1e}, median rel. error = "
        f"{r['median_rel_err_pct']:.1f}%, regime correct {r['regime_correct']}/60")
    e = res["elasticities"]["S"]
    log("Sec 5.8  elasticities: " + ", ".join(f"{k} {v:+.2f}" for k, v in e.items()))
    c = res["convergence"]
    log(f"Sec 5.8  temporal orders {min(c['temporal_orders']):.2f} to {max(c['temporal_orders']):.2f}; "
        f"spatial orders {', '.join(f'{o:.2f}' for o in c['spatial_orders'])}; "
        f"mass drift {c['mass_drift']:.1e}; max |g| = {c['max_amplification']:.6f}")
    if res.get("correlations"):
        cc = res["correlations"]
        log(f"Sec 5.5  correlations: count r = {cc['r_count']:.2f}, first crisis r = {cc['r_first']:.2f}")


if __name__ == "__main__":
    main()
