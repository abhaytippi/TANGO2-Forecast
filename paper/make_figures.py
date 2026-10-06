"""Draw Figure 9 (f_ode.pdf) and Figure 10 (f_tor.pdf) from paper/results/pde_results.json.

Run after ``python paper/reproduce_pde.py``. Output goes to paper/results/.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
BSTAR = 0.01273
RED, ORA, GRN, PUR, BLU, GRY = "#C0392B", "#E08A2E", "#2E8B57", "#7D5BA6", "#2C6FAD", "#808080"

plt.rcParams.update({
    "font.family": "DejaVu Serif", "mathtext.fontset": "dejavuserif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left",
})

LABEL = {"r_star": r"$r^*$ (threshold)", "delta": r"$\delta$ (clearance)", "sigma": r"$\sigma$ (source width)",
         "alpha": r"$\alpha$ (growth)", "x0": r"$x_0$ (locus)", "kappa": r"$\kappa$ (reset)",
         "D": r"$D$ (diffusion)", "sigma_eta": r"$\sigma_\eta$ (noise)"}
LINE = {"r_star": RED, "delta": PUR, "alpha": BLU, "sigma": ORA, "D": GRN, "x0": GRY}


def fig_tor(res):
    S = res["elasticities"]["S"]
    order = sorted(S, key=lambda k: -abs(S[k]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    vals = [S[k] for k in order]
    cols = [RED if abs(v) >= 1 else ORA if abs(v) >= 0.5 else GRN for v in vals]
    y = np.arange(len(order))[::-1]
    a.barh(y, vals, color=cols, edgecolor="black", linewidth=0.6, height=0.8)
    a.set_yticks(y, [LABEL[k] for k in order])
    a.axvline(0, color="black", lw=0.8)
    for xv in (-1, 1):
        a.axvline(xv, color="gray", ls=":", lw=1.5)
    a.set_xticks(range(-5, 3))
    a.set_xlabel(r"Elasticity $S_\theta$ of crisis count")
    a.set_title("(a) Parameter sensitivity")

    bs = res["beta_star"]
    b0 = bs["beta_star"]
    b.axhspan(0.0121, 0.0176, color="#E8DCC0", alpha=0.7, label=r"cohort range of $\beta$", lw=0)
    b.axhline(b0, color="black", ls=":", lw=1.2)
    pct = [-20, -10, 0, 10, 20]
    for k in ["r_star", "delta", "alpha", "sigma", "D", "x0"]:
        row = bs["shift"][k]
        yy = [row["-20%"], row["-10%"], b0, row["+10%"], row["+20%"]]
        b.plot(pct, yy, "-o", color=LINE[k], lw=2, ms=6, label=LABEL[k])
    b.set_xticks(pct)
    b.set_xlabel("Parameter perturbation (%)")
    b.set_ylabel(r"Critical amplitude $\beta^*$")
    b.set_title(r"(b) $\beta^*$ relative to the cohort range")
    b.grid(alpha=0.3)
    b.legend(ncol=2, fontsize=8.5, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "f_tor.pdf"))
    plt.close(fig)


def fig_ode(res):
    bl, rec = res["baselines"], res["recovery"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.9))
    a.plot(bl["beta"], bl["peak"], "-s", color=ORA, lw=2, ms=7, label="scalar model at peak")
    a.plot(bl["beta"], bl["pde"], "-o", color=PUR, lw=2, ms=8, label="reaction diffusion model")
    a.plot(bl["beta"], bl["average"], "-^", color=BLU, lw=2, ms=7, label="scalar model on average")
    a.axvline(BSTAR, color=RED, ls=":", lw=1.5)
    a.set_yscale("symlog", linthresh=1)
    a.set_yticks([0, 1, 2, 5, 10, 20, 40], ["0", "1", "2", "5", "10", "20", "40"])
    a.set_ylim(0, 160)
    a.set_xlabel(r"Source amplitude $\beta$")
    a.set_ylabel("Mean crises in 900 days")
    a.set_title("(a) Diffusion versus scalar baselines")
    a.legend(fontsize=9, loc="upper left", frameon=False)

    t, e = np.array(rec["true"]), np.array(rec["estimated"])
    lim = [0.0118, 0.0179]
    b.plot(lim, lim, "--", color="gray", lw=1.2, label="perfect recovery")
    b.scatter(t, e, color=GRN, edgecolor="#1f6b40", s=30, zorder=3, label="synthetic patients")
    b.axvline(BSTAR, color=RED, ls=":", lw=1.5)
    b.axhline(BSTAR, color=RED, ls=":", lw=1.5)
    b.text(0.97, 0.06, f"r = {rec['r']:.3f}", transform=b.transAxes, ha="right")
    b.set_xlabel(r"True $\beta$")
    b.set_ylabel(r"Recovered $\beta$")
    b.set_title("(b) Parameter recovery")
    b.legend(fontsize=9, loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "f_ode.pdf"))
    plt.close(fig)


def fig_bif(res):
    import sys
    sys.path.insert(0, HERE)
    from pde_model import BASE, steady_peak

    b0 = res["beta_star"]["beta_star"]
    betas = np.linspace(0.008, 0.020, 49)
    peaks = [steady_peak(float(b)) for b in betas]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    a.axvspan(0.008, b0, color=GRN, alpha=0.16, lw=0)
    a.axvspan(b0, 0.020, color=RED, alpha=0.11, lw=0)
    a.plot(betas, peaks, color=PUR, lw=2.5)
    a.axhline(BASE.r_star, color="#B8860B", ls="--", lw=1.5)
    a.axvline(b0, color=RED, ls=":", lw=2)
    a.plot([b0], [BASE.r_star], "o", color=RED, ms=9, mec="white", zorder=5)
    a.annotate(rf"$\beta^* = {b0:.5f}$", xy=(b0, BASE.r_star), xytext=(0.0152, 0.615),
               arrowprops=dict(arrowstyle="->", color="black", lw=1.2), fontsize=11)
    a.text(0.0084, 0.83, "no crisis", color="#1f6b40", fontweight="bold", fontsize=11)
    a.text(0.0153, 0.83, "recurrent crises", color="#8e2a20", fontweight="bold", fontsize=11)
    a.set_xlim(0.0075, 0.0205)
    a.set_ylim(0.55, 0.92)
    a.set_xlabel(r"$\beta$")
    a.set_ylabel(r"steady state $\max_x r$")
    a.set_title(r"(a) Critical amplitude $\beta^*$")

    f8 = res["fig8b"]
    b.errorbar(f8["beta"], f8["mean"], yerr=f8["sd"], fmt="-o", color=BLU, lw=2, ms=7, mec="#1b4a75",
               capsize=4)
    b.axvline(b0, color=RED, ls=":", lw=2)
    b.text(b0 + 0.00005, max(f8["mean"]) * 0.93, r"$\beta^*$", color=RED, fontsize=12)
    b.set_xlabel(r"$\beta$")
    b.set_ylabel("crises in 900 days")
    b.set_title(r"(b) Crisis burden versus $\beta$")
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "f_bif.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    with open(os.path.join(RES, "pde_results.json")) as fh:
        res = json.load(fh)
    fig_tor(res)
    fig_ode(res)
    fig_bif(res)
    print("wrote f_tor.pdf, f_ode.pdf and f_bif.pdf to", RES)
