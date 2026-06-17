# How-to guides

Task-oriented recipes. Each guide assumes you've worked through at least
the [Branin demo tutorial](../tutorials/01-branin-demo.md) and answers
one focused "how do I X?" question.

- [Add a new design](add-design.md) — subclass `Design`, register, use from YAML
- [Swap surrogate](swap-surrogate.md) — switch GP for a custom model
- [Run on Slurm](run-on-slurm.md) — sbatch resources, partitions, accounts
- [Run on PBS](run-on-pbs.md) — qsub resources for clusters running PBS Pro (Improv, Bebop)
- [Use Globus for file transfer](use-globus.md) — the `anl` Transfer backend
- [Migrate from EQSQL](migrate-from-eqsql.md) — drop-in compat shim + new API
- [Debug failed samples](debug-failed-samples.md) — logs, store queries, common failure modes
- [Common mistakes](common-mistakes.md) — option typos and the `validate` / `plan` workflow that catches them
- [Use from a notebook](use-from-notebook.md) — the programmatic API mirror of the CLI; `SampleStore` analysis helpers; read-while-running pattern
