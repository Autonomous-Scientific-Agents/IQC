#!/bin/bash -l
#---------------- PBS options ----------------#
#PBS -N iqc_parsl_smoke
#PBS -l select=1:system=aurora
#PBS -l walltime=00:30:00
#PBS -q debug
#PBS -A IQC
#PBS -l filesystems=home:flare
#---------------------------------------------#
#
# Single-node Aurora smoke test for the Parsl dispatcher.
#
# This script runs the Parsl driver as a normal Python process; the Parsl
# HighThroughputExecutor itself uses MpiExecLauncher under the hood to start
# one worker pool on this node (12 workers, one per Intel GPU tile).
#
# Venv requirement:
#   Uses /lus/flare/projects/IQC/keceli/IQC/.venv — a self-contained venv
#   created with `uv` against spack Python 3.12.12, carrying its own
#   torch 2.12.0+xpu and MACE stack. Do NOT `module load frameworks` here:
#   the venv has no system-site-packages and brings everything it needs.
#   The older venv_iqc_aurora is broken on compute nodes (stale torch/HDF5
#   deps that don't match the current frameworks module).
#
# What to verify after the job completes:
#   - runinfo/<run_id>/parsl.log shows 12 workers spawning.
#   - Each worker's manager log shows ZE_AFFINITY_MASK matching a tile id (0.0 .. 5.1).
#   - iqc_<task>_results_<run_id>.jsonl exists with one line per input row.
#   - The corresponding .parquet file exists and is readable.
#
# The retry-survival test (the load-bearing one for this prototype) lives in a
# separate run with IQC_TEST_FAIL_ROW set; see the plan.

set -euo pipefail

cd "$PBS_O_WORKDIR"

# IQC venv (self-contained: spack Python 3.12.12 + torch 2.12.0+xpu + MACE).
# No `module load frameworks` — the venv is built without system-site-packages
# and carries its own XPU stack.
VENV=/lus/flare/projects/IQC/keceli/IQC/.venv
source "$VENV/bin/activate"

# Aurora Parsl AF_UNIX workaround (Oct 2025+)
export TMPDIR=/tmp

# Smoke-test input: 4 small molecules concatenated into one xyz.
SMOKE_XYZ="$PBS_O_WORKDIR/xyz/smoke_4mol.xyz"
if [[ ! -f "$SMOKE_XYZ" ]]; then
    cat \
        "$PBS_O_WORKDIR/xyz/water.xyz" \
        "$PBS_O_WORKDIR/xyz/methane.xyz" \
        "$PBS_O_WORKDIR/xyz/ethylene.xyz" \
        "$PBS_O_WORKDIR/xyz/benzene.xyz" \
        > "$SMOKE_XYZ"
fi

LOG="parsl_smoke_${PBS_JOBID:-local}.log"
echo "Started: $(date '+%F %T')" | tee "$LOG"
echo "Node: $(hostname)" | tee -a "$LOG"

# Use --parsl-local because we already own one Aurora node from PBS — no need
# to ask Parsl to submit a second PBS job. We do enable --parsl-tile-pin so the
# workers get pinned to the 12 Intel GPU tiles, exactly mirroring the
# multi-node production config.
iqc-parsl \
    --parsl-local \
    --parsl-workers 12 \
    --parsl-tile-pin \
    --parsl-retries 2 \
    -t opt \
    --calculator mace \
    -x "$SMOKE_XYZ" \
    --scratch "$PBS_O_WORKDIR/parsl_smoke_scratch_${PBS_JOBID:-local}" \
    -l INFO 2>&1 | tee -a "$LOG"

echo "Finished: $(date '+%F %T')" | tee -a "$LOG"
