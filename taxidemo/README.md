# taxidemo — polarisopt workflows on a toy taxi simulator

A self-contained demonstration of `polarisopt` using a pure-Python port of the
[Emukit Playground](https://github.com/amzn/emukit-playground) taxi simulator
as the black-box model. Everything that takes hours per evaluation with a real
POLARIS model runs in under a second here, but the workflow is identical: a
stochastic simulator, a YAML-declared study, and the same CLI / Python API /
SampleStore on both your laptop and the cluster.

Three workflows, in increasing order of machinery:

| # | Workflow | Where | Interface |
|---|----------|-------|-----------|
| 1 | Explore the simulator | local | Jupyter — [`notebooks/01_taxi_simulator.ipynb`](notebooks/01_taxi_simulator.ipynb) |
| 2 | Latin-hypercube screening | local | Jupyter — [`notebooks/02_lhs_local.ipynb`](notebooks/02_lhs_local.ipynb) |
| 2b | Morris sensitivity screening | local | Jupyter — [`notebooks/04_morris_screening.ipynb`](notebooks/04_morris_screening.ipynb) |
| 3 | LHS warm-up + Bayesian optimization | LCRC Crossover (Slurm) | **CLI**, then [`notebooks/03_bo_crossover.ipynb`](notebooks/03_bo_crossover.ipynb) for analysis |
| 4 | Calibration — recover parameters from observed data | local | Jupyter — [`notebooks/05_calibration.ipynb`](notebooks/05_calibration.ipynb) |

Workflow 2 is shown through the Python API in a notebook (interactive plots,
SampleStore right in the session); workflow 3 is shown through the CLI because
that is the natural interface on a cluster login node — submit, detach,
monitor, retry.

## The simulator

A square grid of one-way roads. Taxis drive one tile per step; journey
requests appear on sidewalks, are dispatched to the nearest available taxi,
and pay a base fare plus a distance charge with surge pricing on completion.
The fleet costs `0.1 × taxi_count` per step to operate, and customers refuse
to book when prices are set too high. Outputs per episode: `profit`, `missed`
customers, mean `pick_up_time`, `journeys_completed`, `profit_per_journey`.

The port is faithful to the original JavaScript
([`js/simulators/taxi.js`](https://github.com/amzn/emukit-playground/blob/master/js/simulators/taxi.js))
including its quirks — see the docstring in
[`src/taxidemo/simulator.py`](src/taxidemo/simulator.py). Unlike the original
it is seedable, so every sample is reproducible.

```python
from taxidemo import run_taxi_simulation

run_taxi_simulation(seed=0, taxi_count=40, base_fare=6.5)
# {'profit': ..., 'missed': ..., 'pick_up_time': ..., ...}
```

## Install

```bash
pip install -e '.[bo]'                      # polarisopt + BoTorch stack, from the repo root
pip install -e ./taxidemo                   # this demo
pip install -e './taxidemo[notebooks]'      # + jupyter/matplotlib/pandas, for the notebooks
```

Installing `taxidemo` registers the `taxi` simulator type with polarisopt via
an entry point (`[project.entry-points."polarisopt.simulators"]` in
[`pyproject.toml`](pyproject.toml)) — no imports or sitecustomize needed,
`simulator: {type: taxi}` just works in any study YAML. The plugin runs each
sample as a slave subprocess (`python -m taxidemo.runner`), mirroring the
master/slave pattern of real POLARIS runs, and averages `n_repeats` seeded
episodes per design to tame the simulator noise.

## Workflow 2 — LHS screening, local machine

Study definition: [`studies/lhs-local.yaml`](studies/lhs-local.yaml) — 64 LHS
designs over `taxi_count`, `base_fare`, `cost_per_tile`, `max_multiplier`
(3 seeded repeats each), local runner, multi-key identity metric so the
analysis gets `profit`, `missed`, `pick_up_time`, `journeys_completed`
straight from the SampleStore.

Open [`notebooks/02_lhs_local.ipynb`](notebooks/02_lhs_local.ipynb) and run it
top to bottom — it launches the study through the Python API (`run_study`),
then pulls the results into a DataFrame and plots the marginal response of
profit to each knob. The full study takes about half a minute.

The identical study also runs from the CLI, which is what the cluster workflow
builds on:

```bash
polarisopt run    studies/lhs-local.yaml
polarisopt status studies/lhs-local.yaml
```

Both paths share the workspace (`~/taxidemo-runs/lhs-local`), so you can run
from the CLI and analyze in the notebook — finished samples are never
re-evaluated.

### Variant: Morris screening (workflow 2b)

[`studies/morris-local.yaml`](studies/morris-local.yaml) swaps the LHS design
for a Morris elementary-effects design — 8 trajectories × 5 points = 40 local
evaluations — and [`notebooks/04_morris_screening.ipynb`](notebooks/04_morris_screening.ipynb)
ranks the knobs by μ\*/σ with SALib. Knobs that land near the origin of the
Morris plot can be pinned in `simulator.options` (the `taxi` plugin accepts
any simulator parameter there), shrinking the space the BO phase has to
search. Same YAML schema, one `design` block changed.

## Workflow 3 — LHS + sequential BO on Crossover (CLI)

Study definition: [`studies/bo-crossover.yaml`](studies/bo-crossover.yaml) —
maximize `profit` (`minimize: false`) with a GP surrogate and qLogEI
acquisition: 16 LHS warm-up samples, then batches of 4, stopping on a profit
plateau or after 10 iterations. Each sample is a single-core Slurm job on the
TPS partition; the master process stays on the login node fitting the GP and
submitting batches.

```bash
# sanity checks first — schema/plugin validation, then a staged dry run
polarisopt validate studies/bo-crossover.yaml
polarisopt plan     studies/bo-crossover.yaml    # renders the sbatch script, submits nothing

# the run itself (blocks while the loop runs — use tmux/screen, or nohup)
polarisopt run studies/bo-crossover.yaml
```

Monitor and recover from a second shell:

```bash
polarisopt status studies/bo-crossover.yaml          # per-phase sample counts
squeue -u $USER                                      # the underlying jobs
polarisopt logs studies/bo-crossover.yaml 7          # stdout/stderr of sample 7
polarisopt retry-failed studies/bo-crossover.yaml --run   # flip FAILED → PENDING and resume
polarisopt resume studies/bo-crossover.yaml          # pick up after any interruption
```

When the run finishes, analyze it with
[`notebooks/03_bo_crossover.ipynb`](notebooks/03_bo_crossover.ipynb):
convergence of best-so-far profit, where the acquisition function concentrated
its samples, and the best design found.

### What to expect

From the runs used to build this demo (seed 42 throughout):

| study | evaluations | best profit |
|---|---|---|
| defaults (playground settings) | — | ≈ 6,000 |
| LHS screen (workflow 2) | 64 | 15,645 |
| LHS + BO (workflow 3) | 32 | **16,834** |

The 32-evaluation sequential study beat the twice-as-large uniform screen. In
this particular run the winning design actually came from the warm-up (the
notebook's convergence plot shows it), and the acquisition batches then failed
to improve on it — so the plateau criterion cut the study off after 4
iterations instead of burning the full budget. Both halves of that are the
point: warm-up + acquisition explores efficiently, and the stop criteria keep
you from paying for evaluations that aren't helping. With a per-evaluation
cost of hours instead of seconds, that is real money.

## Workflow 4 — calibration by parameter recovery

The workflows above *optimize* an objective; calibration is the inverse
problem polarisopt was built for — match **observed data**. Where a real
POLARIS calibration matches link counts through the `link_moe` metric, taxidemo
matches three taxi-system
observables (journeys completed, mean pick-up time, missed customers) through
its own [`output_match`](src/taxidemo/metrics.py) metric — a custom `Metric`
plugin registered via the `polarisopt.metrics` entry point, demonstrating the
second plugin family after the simulator itself.

[`studies/calibrate-local.yaml`](studies/calibrate-local.yaml) runs the same
GP + qEI loop as workflow 3 but *minimizes* the mean squared relative error
against a targets file, with an `epsilon` stop once the data is matched to
within noise. [`notebooks/05_calibration.ipynb`](notebooks/05_calibration.ipynb)
stages the full parameter-recovery test: simulate synthetic field data from
hidden "true" parameters (seeds the calibration never sees), calibrate against
that data alone, then compare — first in data space (does the calibrated model
reproduce the observations at held-out seeds?), then in parameter space (how
close to the truth, and what the 3-observables-vs-4-knobs identifiability gap
means for real network calibration).

## Layout

```
taxidemo/
├── src/taxidemo/
│   ├── simulator.py      # the taxi simulator (stdlib-only, seedable)
│   ├── runner.py         # slave entry point: python -m taxidemo.runner in.json out.json
│   ├── plugin.py         # polarisopt Simulator plugin (type: taxi)
│   └── metrics.py        # polarisopt Metric plugin (type: output_match)
├── studies/
│   ├── lhs-local.yaml       # workflow 2
│   ├── morris-local.yaml    # workflow 2b
│   ├── bo-crossover.yaml    # workflow 3
│   └── calibrate-local.yaml # workflow 4
├── notebooks/            # workflows 1–4, in order
└── tests/                # simulator unit tests + plugin round-trip tests
```

## Tests

```bash
python -m pytest taxidemo/tests
```
