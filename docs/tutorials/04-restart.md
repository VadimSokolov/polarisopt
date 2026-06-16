# 04 · Restart and resume

polarisopt is built to survive interruptions. The `SampleStore` is the
single source of truth, and sequential phases checkpoint the iteration
counter + RNG state to `phase_state` rows. Crash, OOM-kill, Ctrl-C, or
schedule the master into the future — when you re-run, it picks up.

## 1. Start the demo

Reuse the Branin demo from [Tutorial 01](01-branin-demo.md), but bump
the budget so we have time to interrupt:

```yaml
# restart.yaml
name: restart-demo
workspace: /tmp/restart-demo
seed: 42

simulator: { type: mock, options: { function: branin } }
runner:    { type: local, options: {} }

parameters:
  inline:
    - { name: x1, file: dummy.json, min: -5.0, max: 10.0 }
    - { name: x2, file: dummy.json, min:  0.0, max: 15.0 }

metric:
  type: identity
  options: { keys: value }

phases:
  - name: bo
    type: sequential
    warm_up: { type: lhs, options: { n: 8 } }
    generator:
      type: acquisition
      options:
        surrogate:  { type: gp,  options: {} }
        acquisition: { type: qei, options: { mc_samples: 128 } }
    batch_size: 2
    stop:
      type: max_iter
      options:
        n: 10            # 8 + 10*2 = 28 evaluations total
```

```bash
polarisopt run restart.yaml
```

Wait for the warm-up to finish and the BO loop to log a few iterations.
Then hit **Ctrl-C**.

## 2. Check what survived

```bash
polarisopt status restart.yaml
```

You should see something like:

```
bo: {'finished': 14}    # 8 warm-up + 3 iterations × q=2
```

Open the store to confirm `phase_state` was checkpointed:

```python
from polarisopt.samples.store import SampleStore
from polarisopt.utils.paths import workspace_layout
from polarisopt.config import load_study_config

cfg = load_study_config("restart.yaml")
store = SampleStore.open(workspace_layout(cfg.workspace)["db"], cfg.name)
print(store.load_phase_state("bo"))
# {'iteration': 3, 'rng_state': b'...', 'surrogate_state': None, 'updated_at': ...}
```

## 3. Resume

```bash
polarisopt resume restart.yaml
```

The master:

1. Reads the latest `phase_state` row.
2. Restores the RNG state into the runtime generator.
3. Picks up the iteration counter where it left off.
4. Refits the surrogate from the SampleStore.
5. Continues the loop until `max_iter=10`.

Result: 8 warm-up + 10 BO iterations × 2 = 28 finished samples, with
identical statistical behavior as if the interruption never happened.

## 4. Why this works

The master only ever holds three things that *change* during a phase:

| State | Where it lives |
|---|---|
| Sample inputs / outputs / status | `samples` table |
| Current iteration | `phase_state.iteration` |
| RNG | `phase_state.rng_state` (pickle of `np.random.BitGenerator.state`) |
| Surrogate | **Refit on demand** — not serialized |

Refitting the surrogate from scratch on resume is fast (seconds) and
deterministic, so we avoid pickling torch state across versions.

## 5. Edge cases

**Mid-iteration crash, before any sample finished.** Restart re-runs the
exact same batch — same RNG state means same proposals.

**Mid-iteration crash, after some samples finished.** Restart will see
PENDING samples in the store, evaluate them first, *then* continue the
loop. No sample is evaluated twice.

**Mid-evaluation crash.** Slurm jobs may still be running on compute
nodes. Resume queries `squeue`/`sacct` for each `runner_task_id`:

- Still running → wait for completion.
- Terminal-success → collect output + metric.
- Terminal-failure → mark FAILED.
- Forgotten by Slurm → after `orphan_threshold` UNKNOWN polls, mark
  FAILED with `"job orphaned"`.

**Configuration changed between crash and resume.** Sequence numbers,
RNG, and store survive a YAML edit. Don't change `parameters` or
`metric` — those tie the SampleStore rows to a specific search space.
Do feel free to bump `stop.criteria` knobs (`max_iter` etc.) to extend
or shorten the loop.

## See also

- [Concept: Restart correctness](../concepts/restart-correctness.md)
- [How-to: Debug failed samples](../how-to/debug-failed-samples.md)
