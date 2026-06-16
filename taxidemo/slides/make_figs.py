"""Generate the figures used in pug-2026.qmd into ./fig/.

Run with the taxidemo venv:  python make_figs.py
Figures:
  fig/bo-illustration.png        1-D GP + Expected-Improvement on the taxi sim
  fig/calibration-convergence.png  discrepancy vs evaluation, warmed calibrate-local store
  fig/bo-convergence.png         best profit vs evaluation, warmed bo-local store
The emukit-taxi.png screenshot is a static asset (not generated here).
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).parent / "fig"
FIG.mkdir(exist_ok=True)

# Deck palette (custom.scss)
NAVY = "#1a3a5c"
BLUE = "#1a6fb5"
BAND = "#1a6fb5"
ACCENT = "#d1495b"   # warm accent for the acquisition / next-pick
GRID = "#c9d6e3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.edgecolor": NAVY,
    "axes.labelcolor": NAVY,
    "text.color": NAVY,
    "xtick.color": NAVY,
    "ytick.color": NAVY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 160,
})


def fig_bo_illustration() -> None:
    """1-D Bayesian optimization on the taxi sim: vary fleet size at fixed
    pricing, fit a GP, show where Expected Improvement samples next."""
    from taxidemo.runner import evaluate

    FIXED = {"base_fare": 5.0, "cost_per_tile": 5.0, "max_multiplier": 2.0}

    def profit(n, seed, reps=5):
        r = evaluate({**FIXED, "taxi_count": int(n)}, seed=seed, n_repeats=reps)
        return r["profit"]

    # True (noisy) black-box curve over the fleet-size range.
    grid = np.arange(2, 101, 2, dtype=float)
    true = np.array([profit(n, seed=7) for n in grid])

    # A handful of "observed" designs the optimizer has tried so far.
    x_obs = np.array([8.0, 30.0, 60.0, 95.0])
    y_obs = np.array([profit(n, seed=42) for n in x_obs])

    # Fit a BoTorch GP (the same surrogate polarisopt's `gp` plugin uses).
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition.analytic import ExpectedImprovement

    torch.set_default_dtype(torch.double)
    lo, hi = 1.0, 100.0
    nrm = lambda a: (np.asarray(a, dtype=float) - lo) / (hi - lo)
    Xn = torch.tensor(nrm(x_obs)).reshape(-1, 1)
    Y = torch.tensor(y_obs).reshape(-1, 1)
    gp = SingleTaskGP(Xn, Y, outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    gp.eval()

    Xg = torch.tensor(nrm(grid)).reshape(-1, 1)
    post = gp.posterior(Xg)
    mean = post.mean.detach().squeeze().numpy()
    std = post.variance.sqrt().detach().squeeze().numpy()
    ei = ExpectedImprovement(gp, best_f=Y.max(), maximize=True)
    eivals = ei(Xg.unsqueeze(1)).detach().numpy().ravel()
    x_next = grid[int(np.argmax(eivals))]

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(8.2, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )
    ax.plot(grid, true / 1000, color=GRID, lw=2, ls="--", label="true profit (hidden)")
    ax.fill_between(grid, (mean - 1.645 * std) / 1000, (mean + 1.645 * std) / 1000,
                    color=BAND, alpha=0.18, label="GP 90% interval")
    ax.plot(grid, mean / 1000, color=BLUE, lw=2.4, label="GP mean")
    ax.scatter(x_obs, y_obs / 1000, color=NAVY, s=55, zorder=5, label="evaluations")
    ax.set_ylabel("profit  (×1000)")
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=11)
    ax.set_title("Bayesian optimization of fleet size  (GP surrogate + Expected Improvement)",
                 color=NAVY, fontsize=14, pad=8)

    axe.fill_between(grid, 0, eivals, color=ACCENT, alpha=0.25)
    axe.plot(grid, eivals, color=ACCENT, lw=2)
    axe.axvline(x_next, color=ACCENT, ls=":", lw=2)
    axe.annotate(f"sample next:\n{int(x_next)} taxis", xy=(x_next, max(eivals)),
                 xytext=(x_next + 4, max(eivals) * 0.9), color=ACCENT, fontsize=11,
                 va="top")
    axe.set_ylabel("EI")
    axe.set_xlabel("fleet size  (taxi_count)")
    axe.set_yticks([])
    fig.savefig(FIG / "bo-illustration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bo-illustration.png  (EI next-pick = {int(x_next)} taxis)")


def _conv_frame(workspace, study):
    import pandas as pd
    from polarisopt.samples.store import SampleStore
    from polarisopt.utils.paths import workspace_layout
    layout = workspace_layout(os.path.expanduser(workspace))
    store = SampleStore.open(layout["db"], study)
    raw = store.to_dataframe()
    df = pd.DataFrame({
        "iteration": raw["iteration"],
        "status": raw["status"],
        "val": [m[0] if m else None for m in raw["metric"]],
        "id": raw["id"],
    })
    df = df[df["status"] == "finished"].sort_values("id").reset_index(drop=True)
    return df


def fig_calibration_convergence() -> None:
    df = _conv_frame("~/taxidemo-runs/calibrate-local", "taxi-calibrate")
    evals = np.arange(1, len(df) + 1)
    n_warm = int((df["iteration"] == 0).sum())
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(evals, df["val"].cummin(), drawstyle="steps-post", color=NAVY, lw=2.4, label="best so far")
    sc = ax.scatter(evals, df["val"], c=df["iteration"], cmap="viridis", s=42, zorder=3)
    ax.axvline(n_warm + 0.5, ls="--", color=GRID, lw=1.5)
    ax.text(n_warm / 2, ax.get_ylim()[1], "LHS warm-up", ha="center", va="top", color=NAVY, fontsize=10)
    ax.axhline(0.02, ls=":", color=ACCENT, lw=2, label="epsilon stop (0.02)")
    ax.set_yscale("log")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("discrepancy vs field data")
    ax.set_title("Calibration: discrepancy drops below epsilon, loop stops early",
                 color=NAVY, fontsize=14, pad=8)
    ax.legend(loc="upper right", frameon=False)
    cb = fig.colorbar(sc, ax=ax, label="BO iteration")
    cb.outline.set_edgecolor(GRID)
    fig.savefig(FIG / "calibration-convergence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote calibration-convergence.png  ({len(df)} evals, best={df['val'].min():.4f})")


def fig_bo_convergence() -> None:
    layout_dir = os.path.expanduser("~/taxidemo-runs/bo-local")
    if not Path(layout_dir, "polarisopt.db").exists():
        print("skip bo-convergence.png (bo-local not run yet)")
        return
    df = _conv_frame("~/taxidemo-runs/bo-local", "taxi-bo")
    evals = np.arange(1, len(df) + 1)
    n_warm = int((df["iteration"] == 0).sum())
    best = df["val"].cummax()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(evals, best / 1000, drawstyle="steps-post", color=NAVY, lw=2.4, label="best profit so far")
    sc = ax.scatter(evals, df["val"] / 1000, c=df["iteration"], cmap="viridis", s=42, zorder=3)
    ax.axvline(n_warm + 0.5, ls="--", color=GRID, lw=1.5)
    ax.text(n_warm / 2, ax.get_ylim()[0], "LHS warm-up", ha="center", va="bottom", color=NAVY, fontsize=10)
    ax.set_xlabel("evaluation")
    ax.set_ylabel("profit  (×1000)")
    ax.set_title("Bayesian optimization: profit climbs, then plateaus and stops",
                 color=NAVY, fontsize=14, pad=8)
    ax.legend(loc="lower right", frameon=False)
    cb = fig.colorbar(sc, ax=ax, label="BO iteration")
    cb.outline.set_edgecolor(GRID)
    fig.savefig(FIG / "bo-convergence.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote bo-convergence.png  ({len(df)} evals, max profit={df['val'].max():.0f})")


if __name__ == "__main__":
    fig_bo_illustration()
    fig_calibration_convergence()
    fig_bo_convergence()
