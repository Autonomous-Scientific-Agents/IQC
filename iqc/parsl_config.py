"""Parsl configurations for IQC dispatch.

Two factories are exposed:

- ``make_aurora_config(...)``: production config for ALCF Aurora — one Parsl
  worker per Intel GPU tile (12/node), PALS-aware launcher, retries on failure.
- ``make_local_config(...)``: laptop/compute-node config using LocalProvider
  with a configurable worker count — for smoke tests without a PBS allocation.

Both return a ``parsl.config.Config`` ready to pass to ``parsl.load(cfg)``.
"""
from __future__ import annotations

import os
from typing import Optional

from parsl.addresses import address_by_interface
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import MpiExecLauncher
from parsl.providers import LocalProvider, PBSProProvider


def _aurora_address() -> str:
    """Force Parsl interchange to bind to the Aurora HSN0 IP.

    Default `address_by_query()` returns the public IP via api.ipify.org, which
    fails on login nodes (proxy 503) and yields a non-routable address on
    compute nodes. The interchange then advertises a long list of IPs to the
    managers (10 interfaces on a typical compute node) and at 100+ node scale
    most managers fail to find a viable ZMQ route, hit `expire_bad_managers`,
    and their tasks get re-queued in a heartbeat-cascade that never converges.
    Pinning to hsn0 avoids the probe list entirely.
    """
    return address_by_interface("hsn0")


# 6 GPUs x 2 tiles = 12 tile identifiers, one per worker.
AURORA_TILE_NAMES = [f"{g}.{t}" for g in range(6) for t in range(2)]

# ALCF-recommended core-pinning string for the Parsl worker pool: skips the
# kernel-reserved cores 49-52 and the matching SMT siblings 153-156. Copy
# verbatim from the Aurora Parsl docs.
AURORA_CPU_AFFINITY = (
    "list:"
    "1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:"
    "33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:"
    "69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204"
)


def _default_worker_init(
    venv_activate: Optional[str] = None,
    execute_dir: Optional[str] = None,
) -> str:
    """Worker-init shell snippet used inside every PBSPro job on Aurora.

    ``export TMPDIR=/tmp`` is the documented Aurora workaround for the AF_UNIX
    "path too long" Parsl bug that started appearing in Oct 2025.

    The four trailing exports (``ZE_FLAT_DEVICE_HIERARCHY``,
    ``ZES_ENABLE_SYSMAN``, ``SYCL_CACHE_PERSISTENT``, ``OMP_NUM_THREADS``) are
    the per-process env required by ExaChem (and harmless to other Aurora
    SYCL/oneAPI workloads). Setting them in ``worker_init`` makes them
    inherited by every task subprocess (e.g. ``mpiexec exachem``).
    """
    parts = ["export TMPDIR=/tmp", "module load frameworks"]
    if venv_activate:
        parts.append(f"source {venv_activate}")
    if execute_dir:
        parts.append(f"cd {execute_dir}")
    parts.extend(
        [
            "export ZE_FLAT_DEVICE_HIERARCHY=FLAT",
            "export ZES_ENABLE_SYSMAN=1",
            "export SYCL_CACHE_PERSISTENT=1",
            "export OMP_NUM_THREADS=1",
        ]
    )
    return "; ".join(parts)


def make_aurora_config(
    *,
    venv_activate: str,
    nodes_per_block: int = 1,
    max_blocks: int = 1,
    queue: str = "debug",
    walltime: str = "0:30:00",
    account: str = "IQC",
    filesystems: str = "home:flare",
    execute_dir: Optional[str] = None,
    extra_worker_init: str = "",
    retries: int = 2,
    run_dir: Optional[str] = None,
    heartbeat_threshold: int = 300,
    heartbeat_period: int = 30,
    one_worker_per_node: bool = False,
) -> Config:
    """Build a Parsl Config tuned for ALCF Aurora.

    Each Parsl worker is pinned to one Intel Data Center GPU Max 1550 tile (12
    workers per node); ``MpiExecLauncher`` with ``--ppn 1`` runs one Parsl
    process manager per node, which in turn forks the worker pool.

    Parameters
    ----------
    venv_activate
        Absolute path to the venv's ``bin/activate`` (sourced inside every PBS
        block before tasks run).
    nodes_per_block
        Aurora nodes requested per PBS submission. The smoke test should use 1.
    max_blocks
        Upper bound on concurrent PBS jobs Parsl will queue.
    queue
        Aurora queue name (``debug``, ``debug-scaling``, ``prod``,
        ``EarlyAppAccess``).
    walltime
        PBS walltime, format ``H:MM:SS``.
    account
        Charging account.
    filesystems
        PBS ``-l filesystems=`` value.
    execute_dir
        Worker working directory. Defaults to ``os.getcwd()``.
    extra_worker_init
        Extra shell appended to the default ``worker_init`` (semicolon-prefixed).
    retries
        Re-queue count for failed tasks. The whole point of this prototype: a
        single rank's GPU abort no longer kills the job.
    run_dir
        Optional ``runinfo`` directory override.
    heartbeat_threshold
        Seconds without a worker heartbeat before Parsl declares the worker
        lost and reschedules its tasks. Default 300s (vs. Parsl's 120s) to
        absorb stragglers at the 100+ node scale where MACE model loads and
        Lustre fan-in can starve heartbeats during cold-start.
    heartbeat_period
        Seconds between heartbeats from each worker manager.
    one_worker_per_node
        When True, run a single Parsl worker per node and let that worker's
        task own the whole node (all 12 tiles, all cores). Used for codes
        like ExaChem that fan out internally with their own ``mpiexec``
        (e.g. 13 ranks/node for CC) — Parsl must NOT also place 12 tasks per
        node or oversubscription kills everything. Default False preserves
        the per-tile worker placement for tile-parallel codes.
    """

    if execute_dir is None:
        execute_dir = os.getcwd()
    worker_init = _default_worker_init(venv_activate, execute_dir)
    if extra_worker_init:
        worker_init = f"{worker_init}; {extra_worker_init}"

    # For node-exclusive tasks (e.g. ExaChem's internal mpiexec), expose 1
    # worker/node with no per-tile pinning so the task can spawn its own ranks.
    if one_worker_per_node:
        accelerators_arg = None
        max_workers_arg = 1
        cpu_affinity_arg = "none"
    else:
        accelerators_arg = AURORA_TILE_NAMES
        max_workers_arg = len(AURORA_TILE_NAMES)
        cpu_affinity_arg = AURORA_CPU_AFFINITY

    htex_kwargs = dict(
        label="aurora_htex",
        address=_aurora_address(),
        max_workers_per_node=max_workers_arg,
        cpu_affinity=cpu_affinity_arg,
        prefetch_capacity=0,
        heartbeat_period=heartbeat_period,
        heartbeat_threshold=heartbeat_threshold,
    )
    if accelerators_arg is not None:
        htex_kwargs["available_accelerators"] = accelerators_arg

    htex_kwargs["provider"] = PBSProProvider(
        account=account,
        queue=queue,
        worker_init=worker_init,
        walltime=walltime,
        scheduler_options=f"#PBS -l filesystems={filesystems}",
        launcher=MpiExecLauncher(
            bind_cmd="--cpu-bind",
            overrides="--ppn 1",
        ),
        select_options="",
        nodes_per_block=nodes_per_block,
        min_blocks=0,
        max_blocks=max_blocks,
        cpus_per_node=208,
    )

    cfg_kwargs = {
        "executors": [HighThroughputExecutor(**htex_kwargs)],
        "retries": retries,
    }
    if run_dir is not None:
        cfg_kwargs["run_dir"] = run_dir
    return Config(**cfg_kwargs)


def make_aurora_single_alloc_config(
    *,
    nodes_per_block: int,
    venv_activate: Optional[str] = None,
    execute_dir: Optional[str] = None,
    extra_worker_init: str = "",
    retries: int = 2,
    run_dir: Optional[str] = None,
    heartbeat_threshold: int = 300,
    heartbeat_period: int = 30,
    one_worker_per_node: bool = False,
) -> Config:
    """Build a Parsl Config that uses the *current* Aurora PBS allocation.

    Unlike ``make_aurora_config`` (which uses PBSProProvider to submit a NEW
    PBS job for workers), this config uses ``LocalProvider`` paired with
    ``MpiExecLauncher --ppn 1`` to spawn one ``process_worker_pool.py`` on
    every node of the allocation we already own.

    This avoids the driver/worker queue race in the two-job architecture: when
    the script (driver) runs, the workers run too — same PBS job, same wall
    clock. When the wall clock ends, both stop together. The only cost is the
    driver lives on the head node alongside one manager + 12 workers, but the
    driver is a lightweight Python event loop and ZMQ broker so that's fine.

    Caller is responsible for:
      1. Being inside a PBS allocation.
      2. Passing the *actual* number of nodes from ``$PBS_NODEFILE``. Parsl's
         launcher hardcodes ``mpiexec -n (tasks_per_node * nodes_per_block)``;
         if this is 1, only ONE manager spawns and you waste N-1 nodes.

    Parameters
    ----------
    nodes_per_block
        How many nodes to spread workers across — must match the PBS
        allocation. Pass ``$(wc -l < $PBS_NODEFILE)``.
    venv_activate
        Optional path to a venv's ``bin/activate`` sourced before tasks run.
    execute_dir
        Optional worker working directory.
    extra_worker_init
        Extra shell appended to the default ``worker_init`` (semicolon-prefixed).
    retries
        Per-task retry count.
    run_dir
        Optional ``runinfo`` directory override.
    heartbeat_threshold
        Seconds without heartbeat before a manager is declared lost.
    heartbeat_period
        Seconds between heartbeats from each worker manager.
    one_worker_per_node
        Same semantics as in ``make_aurora_config``: True means one worker per
        node owning all 12 tiles, for codes (e.g. ExaChem) that drive their
        own intra-node mpiexec.
    """

    worker_init = _default_worker_init(venv_activate, execute_dir)
    if extra_worker_init:
        worker_init = f"{worker_init}; {extra_worker_init}"

    if one_worker_per_node:
        accelerators_arg = None
        max_workers_arg = 1
        cpu_affinity_arg = "none"
    else:
        accelerators_arg = AURORA_TILE_NAMES
        max_workers_arg = len(AURORA_TILE_NAMES)
        cpu_affinity_arg = AURORA_CPU_AFFINITY

    htex_kwargs = dict(
        label="aurora_htex_single",
        address=_aurora_address(),
        max_workers_per_node=max_workers_arg,
        cpu_affinity=cpu_affinity_arg,
        prefetch_capacity=0,
        heartbeat_period=heartbeat_period,
        heartbeat_threshold=heartbeat_threshold,
        provider=LocalProvider(
            init_blocks=1,
            min_blocks=1,
            max_blocks=1,
            nodes_per_block=nodes_per_block,
            worker_init=worker_init,
            launcher=MpiExecLauncher(
                bind_cmd="--cpu-bind",
                overrides="--ppn 1",
            ),
        ),
    )
    if accelerators_arg is not None:
        htex_kwargs["available_accelerators"] = accelerators_arg

    cfg_kwargs = {
        "executors": [HighThroughputExecutor(**htex_kwargs)],
        "retries": retries,
    }
    if run_dir is not None:
        cfg_kwargs["run_dir"] = run_dir
    return Config(**cfg_kwargs)


def make_local_config(
    *,
    max_workers: int = 4,
    available_accelerators: Optional[list] = None,
    retries: int = 1,
    run_dir: Optional[str] = None,
    label: str = "local_htex",
) -> Config:
    """Build a Parsl Config that runs workers locally (no PBS).

    Use for smoke tests on a single compute node (or a laptop) without
    submitting through PBS. Each worker is a child process of the driver.

    Parameters
    ----------
    max_workers
        Number of concurrent worker processes.
    available_accelerators
        Optional accelerator IDs (e.g. Aurora tile names like ``"0.0"``). When
        set, Parsl exports ``ZE_AFFINITY_MASK`` (or ``CUDA_VISIBLE_DEVICES``,
        etc.) per worker, mirroring the production tile-pin behavior.
    retries
        Per-task retry count.
    run_dir
        Optional ``runinfo`` directory override.
    label
        Executor label.
    """

    htex_kwargs = {
        "label": label,
        "max_workers_per_node": max_workers,
        "provider": LocalProvider(init_blocks=1, min_blocks=1, max_blocks=1),
    }
    if available_accelerators is not None:
        htex_kwargs["available_accelerators"] = available_accelerators
    cfg_kwargs = {
        "executors": [HighThroughputExecutor(**htex_kwargs)],
        "retries": retries,
    }
    if run_dir is not None:
        cfg_kwargs["run_dir"] = run_dir
    return Config(**cfg_kwargs)
