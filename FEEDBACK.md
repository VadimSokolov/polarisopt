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

Four runs of the *same* study (`taxidemo/studies/calibrate-local.yaml`,
`seed: 42`, identical config) each produced a different trajectory and outcome:

| run | evals | iterations | best discrepancy | stop reason |
|-----|-------|-----------|------------------|-------------|
| 1 (notebook 05, `~/taxidemo-runs/calibrate-local`) | 56 | 10 | 0.0159 | epsilon (< 0.02) |
| 2 (fresh workspace, same seed) | 64 | 12 | 0.1623 | max_iter (epsilon never hit) |
| 3 (fresh workspace, same seed) | 44 | 7  | 0.0067 | epsilon |
| 4 (fresh workspace, same seed) | 60 | 11 | 0.0029 | epsilon |

Same seed, four different eval counts and best discrepancies spanning two orders
of magnitude (0.0029 to 0.1623); run 2 never converged at all.

A maximize-profit BO study (`bo-local.yaml`, same machinery) *does* reproduce its
reported result (32 evals, profit 16,834) run to run, but only because its LHS
warm-up's first sample is already the global optimum: the reported max and the
plateau-stop timing are fixed by the deterministic warm-up, and the still
non-deterministic BO phase never beats it. The BO phase is non-deterministic in
both cases; it only surfaces in the outcome when the answer depends on the BO
phase, as in calibration where the warm-up is far from the target.

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
