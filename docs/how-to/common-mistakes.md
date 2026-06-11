# Common mistakes — and how to catch them in <1s

The most expensive class of polarisopt mistake isn't a runtime crash —
it's a typo in `options:` that passes YAML validation, then crashes
after the model copy. On DFW-scale models that's 30s of staging burned
per try.

The cure since v0.8:

```bash
polarisopt validate study.yaml
```

`validate` introspects every plugin's `__init__` signature and errors
on `options:` keys that aren't accepted. Run it before every submit.

## The typo class

These are the typos that bit the calibration agent during Phase 1, all
caught by `polarisopt validate` in v0.8+:

| You wrote | Real arg | Plugin |
|---|---|---|
| `distance: l1` | `aggregation: l1` | `ChoiceShareMetric` |
| `sim_key: demand_db` | `source_key: demand_db` | `ChoiceShareMetric` |
| `count_col: "n"` | `count_col: "count"` (default; rename your target) | `ChoiceShareMetric` |
| `keys_to_extract: value` | `keys: value` | `IdentityMetric` |
| `population_scal_factor: 0.05` | `population_scale_factor: 0.05` | `polaris_convergence` runner_options |
| `num_threds: "16"` | `num_threads: "16"` | `polaris` / `polaris_convergence` |

`polarisopt validate` flags every row above as:

```text
metric 'choice_share': option(s) ['distance', 'sim_key'] not in __init__
signature. Accepted: ['aggregation', 'count_col', 'source_key', ...]
```

The runner_options whitelist on `PolarisConvergenceSimulator` catches
typos that aren't in the `__init__` signature itself (because they're
forwarded as CLI flags to your runner script). `polarisopt plan` runs
that check.

## Validate workflow

For day-to-day editing, two commands cover the ground:

```bash
# 1. Schema + signature check. <1s. Run on every edit.
polarisopt validate study.yaml

# 2. Stage sample 0 + render sbatch script. ~10s. Run before submitting.
polarisopt plan study.yaml
```

`plan` is `validate` plus:

- actually stages sample 0's workspace (catches transfer config errors,
  module-load failures, runner-script path issues),
- renders the sbatch script to disk so you can inspect it,
- soft-validates `polaris_convergence` `runner_options` against the
  `KNOWN_RUNNER_OPTIONS` whitelist (catches polarislib `ConvergenceConfig`
  typos that `validate`'s signature check can't see).

The staged folder is left intact for inspection. Delete when done.

## What `validate` can't catch

- **Semantic mistakes** — `aggregation: l1` is accepted; whether L1 is
  the right distance for your problem isn't a typo. Read the metric
  docstring.
- **Branch-specific polarislib knobs** — if your polarislib branch
  added a new `ConvergenceConfig` field that polarisopt hasn't
  whitelisted, you'll get a `plan` warning. Either add it to
  `KNOWN_RUNNER_OPTIONS` (in your polarislib fork's polarisopt clone)
  or ignore the warning. It's a soft check, not a hard one.
- **POLARIS binary crashes mid-iteration** — those show up in
  `polarisopt logs <id> --binary` (see
  [debug-failed-samples](debug-failed-samples.md)).

## See also

- [Debug failed samples](debug-failed-samples.md) — what to do when
  `validate` passed but the run still failed.
- [Migrate from EQSQL](migrate-from-eqsql.md) — for porting workflows
  that already exist.
