# Sweep bookkeeping: IQC-side implementation

Implements the IQC portions of the *"Redesign sweep bookkeeping"* proposal
(`/lus/flare/projects/HiFiThermKin/sweep/docs/bookkeeping_redesign.md`): the
per-job incremental append (**A**), the DuckDB query layer (**D**), and the
atomic chunk-claiming primitives. Only the dispatcher + a new query module live
here; the sweep-side rewiring (build_chunk, retiring the text ledgers) lives in
the sweep repo and consumes this module.

## What changed

### A. Per-job incremental append (`iqc/ensemble_launcher_dispatch.py`)
The head process now writes each result to the combined per-job JSONL **the
moment its future resolves**, in completion order (`concurrent.futures.
as_completed`), from a single writer, flushing per row. Previously every result
was buffered into `raw_results` and written only at the epilogue — the write a
walltime kill lost. The combined file is now durable row-by-row, so a kill loses
at most the rows still in flight. File count for the sweep drops from ~10⁵
(per-mol partials) to ~10² (per-job).

Per-mol partials are **off by default** and gated behind `--el-keep-partials`
(see the deviation note below).

### Identity: stored `unique_name_base` (`iqc/main.py`, `iqc/datatools.py`, `iqc/cli.py`)
`_process_one_row` emits a `unique_name_base` column carrying the **stable input
identity**, so downstream done/remaining is an exact column join instead of
every tool regex-stripping `unique_name` (the "identity by string parsing" root
cause). The value is chosen per input mode:
- tabular inputs: the UID read from a `--uid-column` (defaulting to a
  `unique_name` column when present), so the id survives rechunking rather than
  being the volatile `{stem}_row{i}` filename base;
- a multi-frame single XYZ file: `{base}_frame{i}` so distinct configurations
  don't collapse to one identity;
- a directory of files / single structure / SMILES: the filename/derived base.

### D. Query layer + claiming (`iqc/sweep_bookkeeping.py`, `iqc-sweep-book`)
Read-only DuckDB over the per-job parquet/JSONL glob:
- `done_uids` / `remaining` / `summary` — one definition of *done*: a
  **successful** row (finite energy, no `error`/`*_error`, not `nonphysical`,
  `opt_converged` not false), matching `status_query` so there is no cross-tool
  drift. A non-null energy alone is not "done" — IQC keeps a final energy on an
  exhausted optimization / later-stage failure, and those must stay eligible for
  retry. JSONL results left by a walltime-killed job (before the epilogue
  parquet conversion) are read with truncated-last-line tolerance, and an empty
  result glob (fresh sweep) yields empty done / all-remaining rather than an
  error. Many concurrent readers are safe by construction.
- `claim_chunk` / `complete_chunk` / `fail_chunk` — whole-chunk claiming via
  atomic `os.rename` (todo → claimed/`<user>` → done, or back to todo on
  failure, auto re-eligible). Exactly one user wins a contested chunk.

DuckDB is optional: `pip install 'iqc[bookkeeping]'`.

## Deviations from the proposal (and why)

1. **`unique_name_base` is a real stored column, not `unique_name_base` derived
   in SQL.** The proposal's SQL selected `unique_name_base` but no such column
   existed — results only had suffixed `unique_name`, so the "no regex" goal
   was unmet. We store the base at write time (single producer) and keep a
   documented regex **fallback** only for legacy files that predate the column.

2. **Completion-order collection (`as_completed`), not `futures.items()`.** The
   proposal snippet iterated submission order, which blocks on row 0 and would
   still lose everything after the first slow row on a kill. EL's `submit`
   returns a `concurrent.futures.Future`, so `as_completed` gives true
   incremental durability.

3. **Per-mol partials retained behind a flag, not deleted.** The proposal calls
   the epilogue write "the sole reason per-mol partials exist," but the code
   documented a second reason: at high inner-rank counts a worker's
   `ClusterClient.teardown()` can hang past walltime, so the row's future never
   resolves and the head never sees it (6/829 rows at `el-nodes-per-mol`≈8,
   job 8670952). Incremental head-write does not cover that case. Partials are
   therefore **off by default** (achieving the inode goal) but re-enablable via
   `--el-keep-partials` for teardown-hang-prone configs.

4. **All outcome classes preserved.** The simplified proposal snippet collapsed
   the success / skipped-existing / bad-input / failure / never-collected
   distinctions; the implementation keeps each (and still writes failure rows
   for rows the cluster never returned, so a broken run exits nonzero rather
   than silently dropping inputs).

## Out of scope here (sweep repo follow-ups)
- Switching `build_chunk` to the atomic claim primitives.
- Retiring `done.txt` / `claimed.txt` / `skip_index.jsonl` + the harvest once
  the query layer is validated against both legacy per-run and new per-job
  parquets (both are readable by `iqc-sweep-book` today via `union_by_name`).
