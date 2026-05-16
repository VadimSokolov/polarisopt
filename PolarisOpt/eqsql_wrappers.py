"""EQSQL backend for PolarisOpt.

Submits a POLARIS sample to the polarislib EQSQL Postgres task queue
instead of sbatch'ing it directly. Mirrors the surface of
``slurm_wrappers.run_sim_slurm`` so the dispatch sites in
``eval_sim.run_task`` and ``manager.Manager.run_task`` can pick a backend
from settings.json.

Settings (in settings.json):

    "eqsql": {
        "useeqsql": true,                   # turn this backend on
        "worker_id": "xover.vsokolov.*",    # regex pinning your workers
        "exp_id": "polaris-opt",            # tag for grouping tasks
        "task_type": 1,                     # worker --task-type must match
        "priority": 1,
        "name": "Austin",                   # default: falls back to slurm.name
        "ncpus": "16",                      # default: falls back to slurm.ncpus
        "db_url": null                      # optional explicit Postgres URL;
                                            # defaults to polarislib DbConfig
    }

This is fire-and-forget. ``run_sim_eqsql`` writes a shell script next to
the sample folder, submits it as a ``bash-script`` task, sets
``sample.status='running'`` and returns the ``sample``. Completion is
detected the same way as the slurm path — via the ``finished`` sentinel
written by POLARIS into the task output directory.
"""

import os
import shlex


def _settings(manager):
    return manager.dictionary.get("eqsql", {})


def _get_engine(manager):
    """Return a SQLAlchemy engine for the EQSQL Postgres DB, cached on manager."""
    if getattr(manager, "_eqsql_engine", None) is not None:
        return manager._eqsql_engine

    cfg = _settings(manager)
    db_url = cfg.get("db_url")
    if db_url:
        from sqlalchemy import create_engine

        engine = create_engine(db_url, isolation_level="AUTOCOMMIT", pool_pre_ping=True)
    else:
        from polaris.utils.db_config import DbConfig

        engine = DbConfig.eqsql_db().create_engine()

    manager._eqsql_engine = engine
    return engine


def _build_script(sample, manager):
    """Write a self-contained bash script for this sample and return its path.

    Mirrors the command structure that ``run_sim_slurm`` builds inside its
    slurm template ($SCRIPT): cd into the sample folder, stage the binary
    under ``./bin/``, then either run convergence.py or the binary directly.
    """
    cfg = _settings(manager)
    slurm_cfg = manager.dictionary.get("slurm", {})
    jobname = cfg.get("name", slurm_cfg.get("name", "polaris-opt"))
    num_threads = str(cfg.get("ncpus", slurm_cfg.get("ncpus", "1")))

    polarisbin_dir = os.path.dirname(manager.polaris_executable)
    polarisbin = f"./bin/{os.path.basename(manager.polaris_executable)}"
    scenariopath = os.path.join(sample.folder, manager.polaris_scenario_file)

    lines = [
        "#!/bin/bash",
        "set -e",
        f"cd {shlex.quote(sample.folder)}",
        f"cp -r {shlex.quote(polarisbin_dir)} bin",
    ]
    if manager.convergence:
        with open(os.path.join(manager.working_dir, manager.convergence_path), "r") as fh:
            pyscript = fh.read()
        pyscript = pyscript.replace("$POLARISBIN", "'" + polarisbin + "'")
        pyscript = pyscript.replace("$PRJDIR", "'" + sample.folder + "'")
        pyscript = pyscript.replace("$DBNAME", "'" + jobname + "'")
        pyscript = pyscript.replace("$NCPUS", num_threads)
        pyscript = pyscript.replace("$NRUNS", str(manager.num_abm_runs))
        pyscript = pyscript.replace("$RESTART", str(sample.start_iteration_from))
        convfn = f"{sample.folder}/{jobname}-{sample.index}.py"
        with open(convfn, "w") as fh:
            fh.write(pyscript)
        lines.append(f"python {shlex.quote(convfn)}")
    else:
        lines.append(" ".join([polarisbin, shlex.quote(scenariopath), num_threads]))

    shfn = f"{sample.folder}/{jobname}-{sample.index}.sh"
    with open(shfn, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.chmod(shfn, 0o755)
    return shfn


def run_sim_eqsql(sample, manager):
    """Submit a sample to EQSQL and return immediately (fire-and-forget).

    Returns the same shape as ``run_sim_slurm``:
      - ``sample`` (with ``status='running'`` and ``eqsql_task_id`` set) on success
      - ``False`` on submission failure
    """
    from polaris.hpc.eqsql.eq import insert_task

    cfg = _settings(manager)
    shfn = _build_script(sample, manager)

    definition = {
        "task-type": "bash-script",
        "command": f"bash {shlex.quote(shfn)}",
        "copy-local": False,
    }

    engine = _get_engine(manager)
    exp_id = f"{cfg.get('exp_id', 'polaris-opt')}-{sample.index}"

    print(f"Submitting EQSQL task for sample {sample.index} (exp_id={exp_id})")
    with engine.connect() as conn:
        result = insert_task(
            conn,
            definition=definition,
            input={"folder": sample.folder, "index": sample.index},
            exp_id=exp_id,
            worker_id=cfg.get("worker_id"),
            task_type=int(cfg.get("task_type", 1)),
            priority=int(cfg.get("priority", 1)),
        )
    if not result.succeeded:
        print(f"EQSQL insert_task failed: {result.reason}")
        return False

    task = result.value
    sample.eqsql_task_id = task.task_id
    sample.status = "running"
    print(f"EQSQL task {task.task_id} queued for sample {sample.index}")
    return sample
