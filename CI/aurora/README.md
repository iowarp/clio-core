# CI/aurora: clio-fs validation at scale on Aurora, and IO500 against DAOS

One command runs the `clio_fs_suite` (the tests `CI/ares` runs) on a PBS
allocation of Aurora compute nodes (32 by default), then IO500 on clio-fs
and on DAOS with the same work per rank, and writes a report.

It is manual (no cron, no self-hosted runner).

## Run it

```bash
# 1. Build a Release tree with the FUSE adapter. On the login nodes cmake may
#    pick a pkg-config that cannot see the system libfuse3:
cmake -S . -B build/aurora -DCMAKE_BUILD_TYPE=Release \
      -DPKG_CONFIG_EXECUTABLE=/usr/bin/pkg-config ...
cmake --build build/aurora -j 24
ls build/aurora/bin/clio_cte_fuse      # must exist

# 2. Check the harness on 2 nodes (debug queue, a few minutes):
CI/aurora/run.sh --build build/aurora --suite smoke --nodes 2 --queue debug \
    --time 00:30:00 --wait

# 3. The 32-node run: one job per suite, one after another.
CI/aurora/run.sh --build build/aurora --suite all --nodes 32 --wait

# 4. Compare a later run with that one:
CI/aurora/run.sh --build build/aurora --suite all --nodes 32 --wait \
    --baseline ~/clio_aurora_runs/<previous run dir>
```

| option | default | meaning |
|---|---|---|
| `--suite` | `all` | comma list of `smoke`, `cache`, `posix`, `faults`, `stress`, `safe`, `perf`, `io500`; `all` is every one but `smoke` and `cache` |
| `--nodes` | 32 | nodes per job |
| `--queue`, `--time`, `--account` | `debug-scaling`, `01:00:00`, `IOWarp` | PBS settings |
| `--out` | `~/clio_aurora_runs/<date>-<sha>-<nodes>n` | results directory |
| `--wait` | off | block until the jobs end, then print the report |
| `--resume` | off | with `--out <earlier dir>`: run only the tests that have no verdict yet |
| `--baseline` | none | earlier run dir; enables regression detection |
| `--threshold` | 0.20 | a metric worse by more than this fraction is a regression |
| `--report DIR` | | only rebuild `DIR/report.md` and `DIR/io500/io500.md` |

Without `--wait` the command returns once every job is submitted; run
`CI/aurora/run.sh --report <out>` once they have ended.

Exit status 0 means every test passed, nothing regressed and both IO500
runs produced a score. Read `<out>/report.md` first, then
`<out>/io500/io500.md`; per-leg detail is in `<out>/<leg>/summary.md`, daemon
logs in `<out>/<leg>/logs/`, the job's own output in `<out>/pbs-<suite>.out`.

## Why one job per suite

`debug-scaling` caps a job at one hour and a user at one running and one
queued job. Each suite is therefore its own job, submitted as soon as the
queue accepts it (the command stays in the foreground until the last one is
in, so run it under `nohup` or `tmux` for more than two suites), and each leg gets a time budget (walltime
minus 7 minutes for teardown and the report). When the budget runs out the
tests that were not started are recorded `NOTRUN` (which the report counts
as not passed) and the job still tears down and reports. Continue with the
same command plus `--out <same dir> --resume`.

## What runs

| suite | leg | profile | groups |
|---|---|---|---|
| `smoke` | `smoke` | persistent | six POSIX and cross-node tests (harness check) |
| `cache` | `cache` | persistent | `cache` alone (a few minutes; the policy-cache tests) |
| `posix` | `posix_dist` | persistent | `posix`, `dist`, `dirs`, `apps`, `cache` |
| `faults` | `node_faults` | persistent | `fault`, `repl` (killed daemons, lost nodes, aimed replication/cache faults) |
| `stress` | `stress_tiers` | tiered | `stress`, `cache` (tier overflow with the organizer migrating) |
| `safe` | `safe_bdev` | safe | `safe` (dead / replaced / added drives of a safe_bdev array) |
| `perf` | `perf` | tiered | `tier`, `perf` |
| `io500` | | | `benchmarks/io500/pbs_io500_aurora.sh`, `MODE=cliofs` then `MODE=daos` |

`CI/ares/README.md` describes the test groups. Every leg composes the
branch's default chain: persistent core + volatile core under the write-back
policy cache, the shared-memory cache (metadata only) on top, then stream
and filesystem (`clio_fs_suite/cluster.py`).

## What is different from Ares

- **No local drives.** Every tier, log and mountpoint is under
  `/tmp/$USER/clio_fs_suite` (tmpfs: node-local, counted against the node's
  memory). The tiers are separate CTE targets with different scores, so
  tiering, flushing and restart recovery are exercised, but the `perf` and
  `tier_*` numbers measure the software stack over RAM, not devices. Compare
  them with other Aurora runs only. There is no `ares_tiers` profile.
- **No ctest leg.** Run the unit tests in the regular CI.
- **The `mdtest` group is not run**: it launches with Open MPI's `mpirun`
  options. IO500 covers mdtest here.
- **Python.** The compute image's `/usr/bin/python3` is 3.6 and the suite
  needs 3.7+. The runner passes the `python3` on your `PATH` at submission
  (it must be on a filesystem the compute nodes mount, e.g. a module under
  `/opt/aurora`) to the job and to every node's agent; `CLIO_SUITE_PYTHON`
  overrides it.
- **Runtime libraries.** The daemons get the compiler's runtime library
  directory (from the build's `CMAKE_CXX_COMPILER`) and the job's
  `LD_LIBRARY_PATH` after the build's `bin/`.
- `patchelf` on `PATH` at submission lets the suite point its binary
  snapshot at itself. Without it the daemons load libraries from the build
  directory, so do not rebuild during a run.

## IO500 against DAOS

The `io500` suite compares clio-fs with DAOS through dfuse, both as a POSIX
mount on the same DAOS pool, with the same node count, 16 ranks per node and
the same caps on work per rank (`IO500_EASY_MB` 768, `IO500_HARD_SEG` 400,
`IO500_MD_N` 125), both with `SYNC_FS_ONLY=1`:

- `cliofs`: clio-fs with `COMPOSE=split`: a volatile DRAM core (25% of each
  node's share of the data, plus 512 MB for metadata) and a persistent core
  whose only tier is a native DAOS bdev, under the write-back cache and the
  shared-memory cache; peers talk over Thallium on the Slingshot fabric.
  The build needs `-DCLIO_CORE_ENABLE_THALLIUM=ON -DCLIO_ENABLE_DAOS=ON`.
  The full list of settings is `tuned` in `run.sh`.
- `daos`: IO500 on a dfuse mount of the same pool (`--thread-count=32
  --eq-count=16`, otherwise dfuse defaults).

`--repeat N` (default 1) submits N runs of each, alternating
(`cliofs-1`, `daos-1`, `cliofs-2`, ...), so both systems see the same spread
of nodes and of load on the shared pool. One run is not a result: use at
least 3.

```bash
CI/aurora/run.sh --build build/aurora --suite io500 --nodes 32 --repeat 3 --wait
```

`<out>/io500/io500.md` (`benchmarks/io500/compare_io500.py`) has, per phase,
each system's mean with its lowest and highest run and the coefficient of
variation, and the ratio of the means. A phase whose runs differ by more
than 10% is marked `~`: these runs do not settle its ratio. Above the table
every run is listed with what would make it unfit to compare: a non-zero
exit, a missing score, different caps, or error lines (failed fsync, ENOSPC,
I/O errors) in the ior / mdtest outputs. The command exits non-zero if any
run is unfit.

`IO500_CLIOFS_VARS` / `IO500_DAOS_VARS` append `qsub -v` settings to one
side. The work per rank is capped, so the scores are not valid IO500
submissions.

What the comparison does not equalise: clio-fs acknowledges a write once it
is in the DRAM cache and makes it durable on `fsync`, or at the next periodic
flush (`flush_period_ms`, 10 s); dfuse writes through to DAOS. ior calls
`fsync` before it stops its clock, so the ior phases include the flush.
mdtest ends a phase with `sync`, which the clio-fs mount does not turn into
a flush (the adapter has no syncfs handler), so the small files of
mdtest-hard-write may still be in DRAM when its clock stops.

## Files

| file | role |
|---|---|
| `run.sh` | submission, legs, report |
| `../ares/compare.py` | failures and regressions vs a baseline, `report.md` |
| `../../benchmarks/io500/pbs_io500_aurora.sh` | the IO500 job (clio-fs, DAOS or Flare) |
| `../../benchmarks/io500/compare_io500.py` | the IO500 table |
