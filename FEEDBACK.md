# polarisopt feedback

Issues found by the taxidemo agent while exercising polarisopt through the taxi
demo. Filed here (not fixed in `src/polarisopt/**`, which a separate agent
maintains) for the polarisopt maintainer. Newest first.

## 2026-06-16: BO studies are not reproducible under a fixed `seed:`

**Component:** sequential phase / acquisition generator (`gp` + `qei`), BoTorch
acquisition optimization.
**Severity:** medium. Per-run results are valid, but a study cannot be
reproduced, and run-to-run quality varies. A fresh run can fail to converge.

### What I observed

Two runs of the *same* study (`taxidemo/studies/calibrate-local.yaml`,
`seed: 42`, identical config) produced very different trajectories and outcomes:

| run | evals | iterations | best discrepancy | stop reason | taxi_count |
|-----|-------|-----------|------------------|-------------|-----------|
| A (`~/taxidemo-runs/calibrate-local`, from notebook 05) | 56 | 10 | **0.0159** | epsilon (< 0.02) | 41 |
| B (fresh `/tmp` workspace, same seed 42) | 64 | 12 | **0.1623** | max_iter (epsilon never hit) | 37 |

Run B never converged: the best discrepancy plateaued an order of magnitude
above the epsilon threshold and the loop ran out at `max_iter`. Run A converged
cleanly to 0.0159 and stopped early on epsilon.

### Why

The 16-point LHS warm-up is byte-identical across runs, so the design layer is
seeded correctly off `seed: 42`. The divergence is entirely in the BO phase,
which means the acquisition step's randomness is not pinned by the study seed.
BoTorch's `optimize_acqf` uses random restart initialization
(`gen_batch_initial_conditions`) and the MC acquisition uses a sampler; both
draw from torch's global RNG. With that RNG unseeded, every run takes a
different optimization path.

### Suggested fix

Thread the study `seed` into the acquisition generator. At minimum:

- `torch.manual_seed(...)` before fitting the GP and calling `optimize_acqf`
  each iteration. Derive a per-iteration value (e.g. `seed + iteration`) so
  successive batches stay distinct but reproducible.
- Pass an explicit `seed=` to the MC sampler used by `qei` / `qehvi`
  (e.g. `SobolQMCNormalSampler(..., seed=...)`).

Reproducibility makes demos, debugging, and regression tests possible, and
removes the "sometimes it does not converge" failure mode.

### Reproduce

```bash
# Run A is whatever already sits in ~/taxidemo-runs/calibrate-local.
# Run B: copy the study, point workspace at a throwaway dir, keep seed: 42.
cp taxidemo/studies/calibrate-local.yaml /tmp/study.yaml
#   in /tmp/study.yaml set:  workspace: /tmp/recheck
polarisopt run /tmp/study.yaml
# Compare best metric + eval count against run A; they differ run to run.
```

Environment: macOS, Python 3.13.11, torch 2.12.0, botorch 0.18.1,
gpytorch 1.15.2, polarisopt 0.10.0.
