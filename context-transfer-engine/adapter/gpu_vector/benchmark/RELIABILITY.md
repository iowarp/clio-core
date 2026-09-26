# Reliability work, post-submission (started 2026-09-26)

Goal: make multi-node clio-core runs stable, and make failures cheap and
legible. Rules for this work: reproduce every defect at the smallest scale
that shows it, confirm the mechanism from code and logs before changing
anything, fix in clio-core proper (not in job scripts), then keep the
reproducer as a test. Nothing is submitted at scale that has not passed the
same configuration at small scale.

## Defects seen during the evaluation (16-64 nodes), with status

| # | Symptom | Mechanism | Status |
|---|---|---|---|
| 1 | Writeback "REFUSED (rc=4294966296)" on a healthy peer; the one rank without a FATAL is the backlogged target | The probe is keyed by the origin task's heap address (net_key). A probe delayed behind a backlogged target answers after that task completed and its address was reused by a newer task, whose replica is then failed with kRun2RunNetworkTimeoutRC (-1000). Secondary: recv_map_ entry erased before the response is transmitted; cross-origin key collisions. | Fixed in code and built: per-origin generation carried by the probe, stale answers dropped, two Gone answers required, recv_map_ erased only after the response is transmitted. Verified non-regressing: 2 co-located ranks, out of core (1.5k faults/evicts per rank), probe at 1 s, zero Gone (run 31). A backlogged-target reproducer is still to be written. |
| 2 | "GetBlob: generation N never reached" then FATAL 7 on a healthy neighbour | Fixed 10 s generational-get bound; out-of-core steps take ~40 s and ranks drift | Fixed in code: CLIO_GEN_WAIT_MS, default 120 s (7d67cafd). To do: yaml setting, not env. |
| 3 | At 64 nodes pool creation reached nodes 0 and 32 only | ResolveRangeQuery (ipc_manager.cc) split a range wider than neighborhood_size into at most neighborhood_size multi-container sub-ranges; SendIn sent each only to the node owning its first container; RecvInHandleOne marks received tasks routed so the receiver never fans out; the origin counted one replica per sub-range and reported success. 64 nodes / 32 = exactly nodes 0 and 32. No test ever had N > neighborhood_size. | Fixed and verified on the dev node: 4 co-located ranks with neighborhood_size=2 all register targets and agree on the checksum (run 24). Fix: one single-container Range per container. |
| 4 | Ranks hang in GetOrCreateTag after the final checkpoint until the cap | Each rank embeds its node's runtime; a rank that exits takes the node's containers; peers whose tag lives there wait forever (SWIM off, so nothing marks the node dead) | Worked around: exit barrier in kmeans / grayscott (kmdone / gsdone). Runtime fix open. |
| 5 | Writeback "REFUSED (rc=11)" during seeding on a subset of nodes | PutBlob rc = 10 + 1: the CTE container is registered before its Create runs (pool_manager.cc:866 then :882); Create awaits one broadcast bdev create per target (slower with N); a peer that finished earlier sends puts into that window and finds target_list_ empty (core_runtime.cc:7680). kmeans gated only on a 500 ms sleep. | Fixed in code: targets_ready_ flag set at the end of Create; ExtendBlob waits on it (bounded, 120 s, CLIO_CTE_TARGETS_READY_WAIT_MS) before refusing; a failed RegisterTarget logs kError. A 2-rank delayed-start reproducer does not open the window (Create needs one round trip to a live peer); a 4-rank staggered start (rank 1 waits on ranks 2,3) is the next attempt. |
| 6 | One rank dies (GPU NotPresent fault, OUT_OF_RESOURCES); the other 63 livelock at ROUND CAP for the full cap | No failure propagation between ranks; 2,000,000-round cap is minutes long | Open. Fail-fast in the launcher + a wall-clock cap. |
| 7 | Same config passes at 4 nodes, fails at 64 | Configs edited by hand per script; three scripts generate three yamls; binaries with ad-hoc suffixes and no provenance | Open. One config generator, a build manifest, a mandatory scale ladder. |

## Plan (priority order)

1. Fail fast in the launcher and a wall-clock round cap (defect 6).
2. Smallest-scale reproducers, then fixes, for defects 1, 3, 4, 5.
3. Generation wait as a yaml setting (defect 2).
4. One config generator used by every PBS script (defect 7).
5. Build manifest per benchmark binary (defect 7).
6. Scale ladder script: 1 -> 4 -> 16 nodes with the resident and out-of-core
   gates, prints the 64-node submit line only when every rung passes.
7. Each reproducer becomes a distributed test under gpu_vector/test.
8. A `clio-bench` runbook skill: procedure + failure-signature table.

## Enabler: several runtimes on one node

The debug queue allows 2 nodes and one running job per user, and the repo's
own distributed tests need docker (one container = one hostname). To iterate
in minutes, the runtime gets an optional per-entry port in the hostfile
(`host:port`); a process picks the entry whose port equals its own
`networking.port` / `CLIO_PORT`. Identity, shared-memory segment names
(already suffixed with the port), the client cache key (`addr:port`) and the
per-user IPC sweep (pid-aware) all already tolerate this; only the hostfile
parse, the identity check and the three peer dial sites change. Entries
without a port behave exactly as before. This is a test enabler, not the
production layout: one runtime per node remains the design.

## Other findings

- `runtime.task_progress_interval_ms` defaults to 5000 (on); the header said
  "default 0 = disabled". Comment and default yaml corrected.
- The generational-get wait passed a millisecond nap to `yield()`, which takes
  microseconds: the wait was a hot spin. Fixed (x1000).
- Every node creates a bdev pool container for every other node's tier
  (RegisterTarget's bdev create is broadcast): N^2 pool creates at startup,
  and each node maps N x TIER_MB of (sparse) shared memory. Not yet changed.
- `neighborhood_size` defaults to 32, which is exactly why 64-node pool
  creation reached nodes 0 and 32.
- build-fresh, which every evaluation binary linked against, is a Debug build.

## Log

- 2026-09-26: SWIM defaults changed (fdab7a1f): off everywhere, 300/150/3600 s.
- 2026-09-26: rc=11 identified as "no storage targets registered" (AGENTS.md
  and PITFALLS.md already said so; the 64-node diagnosis had guessed EAGAIN).
- 2026-09-26: host:port hostfile entries added (types.h, ipc_manager.cc,
  ipc_run2run.cc); bench_config.sh (one yaml generator), run_colocated.sh,
  pbs_devnode_aurora.sh written. Defect-5 fix and nap-unit fix built into
  libclio_cte_core_runtime.so. Probe fix written, building.
- 2026-09-26 06:48-07:00 UTC, dev node x4404c0s2b0n0 (job 8871567): host:port
  identity WORKS (node 0 on 9460, node 1 on 9468, same host; 4 ranks too).
  Fail-fast works (7 s instead of a 300 s cap). Launcher lessons, all fixed
  in run_colocated.sh: killing a rank's subshell left the benchmark and its
  embedded runtime alive (GNU timeout moves its child to a new process
  group) -> setsid per rank + timeout --foreground + kill by group; dev-node
  scripts must not poll files under /tmp (node-local). Every co-located run
  still segfaults ~7 s in, in the benchmark's main() (null deref at +0x38
  right after a 0x1000000000 constant), even with the peer not yet started;
  suspect the CLIO_MAIN_SEGMENT_SIZE override; cores in
  devnode/work/smoke2_core, gdb-oneapi works on the login node; a -g kmeans
  build (_x_dbg) is in progress.
- Range-split fix built into libclio_run_cxx.so (with the probe fix).
- 2026-09-26 07:05 UTC: the co-located segfault was an ABI mismatch, not a
  runtime bug. Adding Host::port (types.h) changed a struct that every
  benchmark binary compiles into itself through the runtime headers; the old
  binaries crashed inside main() (inlined accessor at a stale offset) while a
  freshly built kmeans passed: 1 rank small, and 2 co-located runtimes at the
  E5 sizes with matching checksums and remote page traffic. Consequence: all
  benchmark binaries are rebuilt under their existing names; launchers now
  refuse a binary older than the runtime libraries (bench_check_binary), and
  the build script writes a manifest line per binary.
- Noise to fix: kmeans's end-of-run TIER SPLIT check queries bdev pool (513,1),
  which does not exist in a one-tier config; it costs 5 s and an ERROR line.
- 2026-09-26 07:08 UTC: all five benchmark binaries rebuilt against the new
  headers (kmeans _x_ckpt2, grayscott _x_fc2_ct8, gmx, lbann, lammps_md).
  MANIFEST.tsv now records each build. Defect 3 verified fixed (run 24).
- 2026-09-26 07:13 UTC: defect-5 reproduction attempts at 2 and 4 co-located
  ranks (start delays, 16 tiers per node) never opened the window: every
  node's Create waits on the slowest peer's bdev create, so they finish
  together. A test hook (CLIO_CTE_REGISTER_DELAY_MS, per rank via
  BENCH_RANK_ENV_<r>) stretches one node's registration to make the window
  deterministic; runs 33/34 use it. kmeans's end-of-run tier check is now
  opt-in (KM_TIER_CHECK=1). bench_ladder.sh written (kmeans, grayscott).
