---
name: clio-bench
description: Use when building, running, scaling up, or debugging the gpu_vector benchmarks (kmeans, grayscott, gmx, lbann, lammps_md) on Aurora or any multi-node cluster -- submitting PBS jobs, choosing node counts, reading rank logs, or deciding what a failure signature (REFUSED rc=11, ROUND CAP, "generation never reached", Gone, exit=134/139/124) means. Encodes the mandatory small-scale ladder, the shared config generator, fail-fast launching, binary provenance, and the failure signatures learned during the 2026-09 evaluation. Triggers on "run the benchmark", "submit to 64 nodes", "E1/E5", "pbs_e5", "run_colocated", "rank log", "REFUSED", "ROUND CAP", "livelock", "debug queue".
---

# Running the gpu_vector benchmarks without burning the allocation

Every rule here was paid for in node-hours. Read `RELIABILITY.md` in
`context-transfer-engine/adapter/gpu_vector/benchmark/` for the defects
behind them.

## The procedure (no step is optional)

1. **Build with provenance.** `build_newcoro_aurora.sh <name> <name>` with the
   knobs (`TAG`, `EXTRA_CXX`, `IGC_FC`, `AOT=1`). Every build appends a line to
   `build-spike/MANIFEST.tsv` (time, binary, commit, +dirty, knobs). Never
   identify a binary by its suffix alone; read the manifest.
2. **After ANY header or struct change, rebuild every benchmark binary.** The
   benchmarks compile runtime headers into themselves. An old binary against a
   new library crashes inside `main()` and looks like a runtime bug. The
   launchers refuse a binary older than the runtime libraries
   (`bench_check_binary`); do not override that with `BENCH_ALLOW_STALE=1`
   unless you know why the library is newer.
3. **One runtime config.** Launchers source `bench_config.sh` and call
   `bench_clio_yaml <out> <hostfile> [port]`; tiers come from `BENCH_TIERS`.
   Do not write yaml by hand in a job script: the E4 script once lacked one
   knob (`neighborhood_size`) and failed at 64 nodes while E1 and E5 had it.
4. **Climb the ladder before any multi-node submit.** `bench_ladder.sh <wl>
   <workdir>` runs 1 rank resident, 2 resident, 2 out of core, 4 out of core
   on ONE node with co-located runtimes, gating on clean exits, checksum
   agreement, and evictions > 0 for out-of-core rungs. A configuration that
   has not passed the ladder is not submitted anywhere.
   After any runtime change, also run `bench_colocated_tests.sh <workdir>`:
   the reproducers of the known defects as pass/fail cases (smoke, range
   split, targets race with and without the fix, peer death, probe out of
   core), ~10 minutes on one node.
   After any CTE or runtime change, run the CPU-only CTE stress ladder,
   `context-transfer-engine/benchmark/stress_ladder.sh <workdir>` (~4 min,
   no GPU): `clio_cte_vector_stress` drives the CTE the way the vector does
   (generational pages, halo and random reads across co-located runtimes,
   PodMulti batches, checkpoints, the gate barrier) and verifies every
   byte. It found the torn-read defects (8, 11) the GPU benchmarks turned
   into "wrong checksum, no error". Build it with
   `context-transfer-engine/benchmark/build_stress_aurora.sh`; it is
   subject to the same staleness check as the GPU binaries.
5. **Iterate on a held dev node**, not on the queue: `qsub -v
   DEVNODE_DIR=<flare dir> pbs_devnode_aurora.sh` (1 node, debug queue, 1 h)
   executes scripts dropped into `<dir>/queue/`, output in `<dir>/out/`.
   Markers and outputs must live on flare; `/tmp` is node-local and a script
   polling a login-node `/tmp` file waits forever.
6. **Change one thing per run.** Two changes between runs cost a full job to
   untangle, twice.
7. **Scale in the queue that fits**: debug (<=2 nodes, 1 h), debug-scaling
   (<=256 nodes, 1 h, one running job per user), capacity (<=16 nodes),
   small/prod (>=256). A 256-node request waited hours behind a 228-node job;
   four 64-node jobs went through.

## Co-located runtimes (the smallest distributed scale)

`run_colocated.sh <n> <workdir> <exe> [args]` runs n runtimes on one host:
rank r gets `CLIO_PORT = 9460 + 8r`, its own GPU tile, and the hostfile lists
`host:port` per rank (the runtime resolves its own entry by port). Fault
injection: `BENCH_RANK_DELAY_<r>` (late start), `BENCH_RANK_KILL_<r>`
(SIGKILL after N s), `BENCH_RANK_ENV_<r>="VAR=v ..."` (per-rank env, e.g.
`CLIO_CTE_REGISTER_DELAY_MS`). Fail-fast is on: the first non-zero rank stops
the rest (`BENCH_NO_FAILFAST=1` to observe a hang instead).

## Failure signatures

| In the rank log | Meaning | Look at |
|---|---|---|
| `writeback ... REFUSED by the runtime (rc=11)` | PutBlob rc 10+1: the OWNER node's CTE had no registered storage target when the write arrived. Startup race (fixed: ExtendBlob waits for Create) or a permanent registration failure. | Owner's log for `Failed to register target` (permanent) vs `waiting for Create` (race). |
| `REFUSED ... (rc=4294966296)` | kRun2RunNetworkTimeoutRC (-1000): the origin gave up on a replica. Was the task-progress probe's false Gone (fixed: generations, two strikes). | `[TaskProgress]` lines on the origin; `task_progress_interval_ms`. |
| `GetBlob '...': generation N never reached` then `DEVICE FATAL 7` | A generational get waited the whole bound (now 120 s, `CLIO_GEN_WAIT_MS`) for a page the writer never published, or published late. | The writer rank: did it publish? Skew > bound? |
| `ROUND CAP HIT ... N block(s) still suspended` | Livelock: faults not being satisfied, almost always because a PEER died. Find the rank with `exit=134/139` first; the ROUND CAP ranks are victims. | The one rank with a different exit code. `CLIO_GV_WALL_CAP_MS` bounds this in time. |
| `Segmentation fault from GPU at 0x0 ... NotPresent` right after a FATAL | Consequence of the FATAL above it, not a separate bug. | The line above. |
| `Segmentation fault from GPU at 0x14..` (a host address) | GPU read of a host buffer whose pinning went stale. Intermittent, one rank. Open. | Which rank; peers then livelock (see ROUND CAP). |
| `exit=139` ~7 s after start, in `main()` | Stale binary: built before a runtime header changed. | `MANIFEST.tsv` vs library mtimes; rebuild. |
| `exit=124` with the timed region complete | The rank finished its work and hung on exit: a peer already exited and took its node's containers (embedded runtime per rank). Benchmarks now barrier before exit. | `GetOrCreateTag` waits; `RouteLocal returned 4`. |
| `connected to runtime (server_pid=...)` in a rank that should be its own runtime | It attached to a STALE runtime left by a previous run on the same node. | `pgrep -f <benchmark>`; kill by process group. |
| `Creating pool '...' with N containers` on nodes 0 and 32 only | Broadcast split by `neighborhood_size` (fixed: one query per container). | `neighborhood_size` in the yaml; runtime commit. |

## Reading a cell

`E5CELL <wl> hbm_gb=<b> nodes=<n> ms=<slowest rank> ... rc=<worst>`: the
`ms` is the timed region on the slowest rank; `rc` is the worst rank's exit
(124 cap, 134 abort, 139 segfault, 143 killed by fail-fast). A cell with
`ms>0` and `rc!=0` measured something and then failed on exit.
