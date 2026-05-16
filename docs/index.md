# polarisopt

Modular design-of-experiments and Bayesian optimization for [POLARIS](https://polaris.taps.anl.gov/).

> **Status: 0.1.0 development release.** Public API may still shift in 0.x.

## What it does

`polarisopt` orchestrates POLARIS calibration and exploration studies. Two algorithm families:

1. **Static design of experiments** — Latin Hypercube, Morris, Sobol, manual designs. One-shot sample generation for screening and sensitivity analysis.
2. **Sequential design of experiments** — warm-up + surrogate-driven Bayesian optimization. Plug-in surrogates (GP via BoTorch), acquisition functions (qLogEI, qLogEHVI), stopping criteria. Single- and multi-objective.

Studies are configured in YAML and executed via the `polarisopt` CLI or programmatically. Sample state is persisted to SQLite so studies can be resumed after interruptions.

## At a glance

```yaml
# study.yaml
name: dfw-calibration
workspace: /lcrc/.../experiments/dfw-001

simulator:
  type: polaris
  options:
    binary: /lcrc/.../polaris_exe/Integrated_Model.sif
    model_source: /lcrc/.../DFW_2050_20251028
    scenario_file: scenario_abm.json
    output_db_filename: DFW-Result.h5
    num_threads: "16"

runner:
  type: slurm
  options:
    default_resources:
      partition: bdwall
      account: POLARIS
      time: "02:00:00"
      nodes: 1
      cpus_per_task: 16
      mem: 64G

parameters:
  source: ./params.yaml

metric:
  type: link_moe
  options:
    target: /lcrc/.../baseline/DFW-Result.h5

phases:
  - name: lhs-screen
    type: static
    design: { type: lhs, options: { n: 16 } }

  - name: bo
    type: sequential
    warm_up: { type: lhs, options: { n: 8 } }
    generator:
      type: acquisition
      options:
        surrogate: { type: gp, options: { nu: 2.5 } }
        acquisition: { type: qei, options: { mc_samples: 256 } }
    batch_size: 4
    stop:
      type: any
      criteria:
        - { type: max_iter, options: { n: 12 } }
        - { type: epsilon, options: { epsilon: 0.01 } }
```

```bash
polarisopt run study.yaml
polarisopt status study.yaml
polarisopt resume study.yaml
```

## Why it exists

Calibrating large POLARIS models requires:

- a **flexible parameter space** mapped into POLARIS JSON configuration files,
- **scalable evaluation** on HPC (Slurm) with file transfer between user storage and cluster scratch,
- a **persistent record of every evaluation** that survives crashes and supports restart,
- **modern Bayesian optimization** (batch, multi-objective, well-tested) when sequential refinement is needed.

`polarisopt` provides all of that behind ABCs so users can swap in custom surrogates, acquisition functions, simulators, runners, or stopping criteria from YAML — no core changes needed.

## Master/slave architecture

```
┌──────────────────────────────────────────────────────────┐
│ Master process (polarisopt run study.yaml)               │
│                                                          │
│   SampleStore (SQLite, single source of truth)           │
│   StudyRunner.run()                                      │
│     StaticDesignStudy / SequentialDesignStudy            │
│       Surrogate.fit() / Acquisition.optimize()           │
│       Runner.submit(JobSpec)                             │
│       Runner.status(job)                                 │
│       Simulator.collect_output() → Metric.compute()      │
└──────────────────────────────────────────────────────────┘
                       │  sbatch + squeue + scancel
                       ▼
┌──────────────────────────────────────────────────────────┐
│ Slaves (compute nodes)                                   │
│   POLARIS binary running on per-sample workspace         │
└──────────────────────────────────────────────────────────┘
```

The master never imports POLARIS. Slaves run the binary against a per-sample workspace prepared by the master's `Simulator.prepare()`. State transitions go through the SampleStore so the master can be killed and resumed.

## Next steps

- [Getting started](getting-started.md)
- [Architecture](architecture.md)
- [Study YAML reference](yaml-reference.md)
- [Plugin authoring](plugins.md)
