# Taxi demo

[`taxidemo/`](https://github.com/anl-polaris/polaris-hpc/tree/master/taxidemo)
is a self-contained demonstration of polarisopt on a toy problem: a
pure-Python, seedable port of the
[Emukit Playground](https://github.com/amzn/emukit-playground) taxi simulator.
Every evaluation takes well under a second, but the moving parts are the same
as a POLARIS calibration — a stochastic black-box simulator behind a
`Simulator` plugin, a YAML-declared study, the SampleStore, and the same CLI
on a laptop and on a Slurm cluster.

It is also a worked example of **plugin packaging**: `pip install -e
./taxidemo` registers `simulator: {type: taxi}` and
`metric: {type: output_match}` through the `polarisopt.simulators` /
`polarisopt.metrics` entry-point groups — no polarisopt changes, no import
boilerplate (see [Plugin authoring](plugins.md)).

| Workflow | Where | Interface |
|---|---|---|
| Explore the simulator | local | notebook `01_taxi_simulator.ipynb` |
| LHS screening | local | notebook `02_lhs_local.ipynb` (Python API) |
| Morris screening | local | notebook `04_morris_screening.ipynb` |
| LHS + Bayesian optimization | LCRC Crossover | CLI, analysis in `03_bo_crossover.ipynb` |
| Calibration (parameter recovery) | local | notebook `05_calibration.ipynb` |

Quick start:

```bash
pip install -e '.[bo]'            # from the repo root
pip install -e './taxidemo[notebooks]'

polarisopt run taxidemo/studies/lhs-local.yaml
polarisopt status taxidemo/studies/lhs-local.yaml
```

From the demo's reference runs (seed 42): the playground's default settings
earn ≈ 6,000 profit; a 64-sample LHS screen finds 15,645; the BO study finds
**16,834 in 32 evaluations** and stops itself on a plateau criterion. The full
walkthrough, including the Crossover CLI session and what to expect from each
notebook, is in the
[taxidemo README](https://github.com/anl-polaris/polaris-hpc/blob/master/taxidemo/README.md).
