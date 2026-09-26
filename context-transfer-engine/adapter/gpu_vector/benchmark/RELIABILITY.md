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
| 4 | Ranks hang in GetOrCreateTag after the final checkpoint until the cap | Each rank embeds its node's runtime; a rank that exits takes the node's containers; peers whose tag lives there wait forever (SWIM off, so nothing marks the node dead) | Reproduced on the dev node (run 36: peer SIGKILLed at 10 s; the survivor finished 173 iterations, then "[stuck-wait] no completion after 60s" until the cap). First fix (count probes that complete with an error) did nothing: a probe to a dead node never completes, because no origin-side timer fails a sent task whose target never answers (run 38 hung exactly like 36). Fix now in code: a pending probe unanswered for kProbeSilenceSec (15 s = three intervals) marks its target dead; ScanSendMapTimeouts then fails the waiting tasks (30 s). Requires the probe on; the generator default is back to 5000 ms. VERIFIED: suite case peer_death (peer SIGKILLed at 10 s; the survivor logs "has not answered a liveness probe ... marking it dead" and fails before its cap). |
| 5 | Writeback "REFUSED (rc=11)" during seeding on a subset of nodes | PutBlob rc = 10 + 1: the CTE container is registered before its Create runs (pool_manager.cc:866 then :882); Create awaits one broadcast bdev create per target (slower with N); a peer that finished earlier sends puts into that window and finds target_list_ empty (core_runtime.cc:7680). kmeans gated only on a 500 ms sleep. | FIXED AND VERIFIED (runs 33/34): with CLIO_CTE_REGISTER_DELAY_MS=6000 on one rank, the peer's seeding puts hit the empty target list; wait 0 reproduces the 64-node signature exactly (REFUSED rc=11, FATAL 8, exit 134); the default 120 s wait completes the run with matching checksums. Skew alone (2 and 4 ranks, 16 tiers) never opened the window: every node's Create waits on the slowest peer. |
| 6 | One rank dies (GPU NotPresent fault, OUT_OF_RESOURCES); the other 63 livelock at ROUND CAP for the full cap | No failure propagation between ranks; 2,000,000-round cap is minutes long | Fixed: bench_watch_ranks / run_colocated fail-fast (first non-zero rank stops the rest; a 300 s cap became 7 s), CLIO_GV_WALL_CAP_MS in the yieldable driver, and dead-peer detection (defect 4) so a survivor fails instead of parking. The GPU host-address fault itself is still open. |
| 8 | Silent data corruption: a page read back is a MIX of two generations (header says gen 4, the tail still holds gen 3; first bad word 102608 of 131072 in one run, 121336 in another) with rc=0 | Found by the CPU-only `clio_cte_vector_stress` (context-transfer-engine/benchmark), not by the vector. PutBlob overwrites a blob's extents IN PLACE under the write token but never drained pinned readers (core_runtime.cc said so: "PutBlob never drains ... read/write concurrency on a blob is unchanged"); a GetBlob mid-ReadData on the same extents returned old and new bytes together. Both the scalar and the PodMultiPutBlob paths (rungs a and d of the stress ladder, 2 and 4 co-located ranks). For the vector this is a halo/stream page read during a neighbour's publish: wrong numbers, no error. | Fixed in code and built: PutBlobImpl drains readers after taking the write token (BeginDrainReaders + wait HasReadPins, the #753 discipline every extent-freeing mutator already used). Verifying with the stress reproducers (rungs a/d repeated). Still to check: the zero-IPC TryReadBlobShm fast path (non-generational host reads of local pages) validates only placement_gen_, which an in-place put does not bump. |
| 9 | Barrier miss: after the checkpoint at step 4, node 0 polled for a peer's barrier blob (`stressbar_1004_2`) for 300 s and never got it, while nodes 1 and 3 read the same blob and passed | Unknown yet. No "generation never reached" warning on any node, so the gets did not sit in the owner's 120 s generational wait; they failed some other way, repeatedly. Node 0 issued its get ~300 ms before the writer's put (it finished its checkpoint first). The same ReduceSum/GetPeers exchange gates every kmeans iteration and every benchmark's exit, so this is a candidate for the "exit=124 with the timed region complete" hangs at 16-64 nodes. | GetPeers now reports attempts and the last rc on timeout; the stress benchmark has `--barriers N` (barrier-only rounds). Reproducers queued (4 ranks x 300 rounds, and the failing rung). |
| 10 | Every clean exit of an embedded-runtime process ends with 900-5000 `RouteTask: RouteLocal returned 4 for pool=(1,0) method=14/21` ERROR lines | `RuntimeManager::ClientFinalize()` (which the CTE benchmarks and `CLIO_RUNTIME_FINALIZE`'s documented contract call) finalized the pool manager and closed the peer connection pool while the process's OWN server workers were still running: the admin pool vanished under the network worker, every kSend/kClientSend was re-queued and failed again until StopWorkers. Peers lose their DEALER connections early as well. | Fixed and verified (0 lines on a 2-rank run): ClientFinalize runs ServerFinalize first when the process is also the runtime, the order `~RuntimeManager` and `CLIO_RUNTIME_FINALIZE` already use. |
| 11 | Same torn pages through the zero-IPC fast path: one rank, non-generational host reads of LOCAL pages, 3 torn pages in 11 s (rung fast1) | `TryReadBlobShm` copies a blob's extents straight out of the RAM tier and re-checks only `placement_gen_`, which an in-place put never changes (the header said so: "concurrent same-blob overwrite vs view is torn-content-visible"). The runtime-side drain (defect 8) cannot see these readers: they pin nothing. This is the default read path for every host client of a local blob (adapters, CAE), not only the benchmarks. | Fix in code: a content seqlock. `BlobInfo::content_seq_` goes odd before a put's first in-place byte and even after its last, mirrored to `ShmBlobRecord::content_seq_` both times (cache layout v4); `TryReadBlobShm` refuses an odd value and discards a copy across which it changed; the zero-copy view stamp (`TryGetBlobViewShm`/`CheckBlobGenShm`) now carries both counters. Building; to verify with rung fast1/fast2. |
| 7 | Same config passes at 4 nodes, fails at 64 | Configs edited by hand per script; three scripts generate three yamls; binaries with ad-hoc suffixes and no provenance | Done: bench_config.sh is the one generator (E1/E4/E5 use it), MANIFEST.tsv per build, bench_check_binary refuses binaries older than the headers, bench_ladder.sh is the mandatory pre-scale gate, bench_colocated_tests.sh keeps the reproducers green, the clio-bench skill is the runbook. |

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
- 2026-09-26 07:20 UTC: committed 3a76539a (runtime/CTE) and 9690b048
  (harness). Defect 4 reproduced (run 36) and fixed in code (dead after 3
  unanswered probes); SetDead now evicts the peer's real port from the client
  pool. The stale-binary check caught the benchmarks being older than the
  relinked library (run 37); all five rebuilt again.
- 2026-09-26 07:27 UTC: bench_colocated_tests.sh (6 cases) first run: smoke,
  range_split, targets_race pass; targets_race_control reproduced the defect
  (rank 0 abort 134, owner refusals) but the check read the wrong log line
  (fixed); peer_death / probe_ooc were refused by the staleness check after a
  library relink with no header change -> the check now compares against
  headers (runtime, ctp, module client, CTE core, gpu_vector), not .so files.
  Ladder: kmeans's atomically summed checksum differs by ~1e-8 across rank
  counts, so the cross-rung gate is a relative tolerance (1e-6) now.
- 2026-09-26 07:31 UTC: bench_colocated_tests.sh ALL SIX CASES PASS (run 44):
  smoke, range_split, targets_race, targets_race_control (reproduces rc 11
  without the wait), peer_death (dead peer detected, no hang), probe_ooc
  (out of core, probe on, zero Gone). Ladder out-of-core rungs now sized
  above the 8-frames-per-block floor.
- 2026-09-26 07:32 UTC: bench_ladder.sh kmeans LADDER PASS (run 45): 1 resident,
  2 resident, 2 out of core (1514 evicts), 4 out of core (1511 evicts).
- 2026-09-26 14:30-15:10 UTC (dev node x4415c7s0b0n0, job 8872208): the
  CPU-only CTE stress ladder (context-transfer-engine/benchmark/
  clio_cte_vector_stress) found defects 8 (torn reads, RPC path), 9 (barrier
  miss, once), 10 (ClientFinalize under live workers) and 11 (torn reads,
  zero-IPC path). Rungs: a FAIL torn, b FAIL barrier, c PASS (4 ranks, 64
  threads, 4 GB/rank), d FAIL torn (scalar), e PASS (non-generational), f PASS
  (32768 x 64 KB pages, 1.5 M gets). After the PutBlob reader drain: a/d
  4/4 PASS, b 5/5 PASS, 400 barrier-only rounds PASS (0.5 ms per round at 4
  ranks). fast1 (1 rank, non-generational, 64 threads) then showed defect 11;
  content seqlock written, verification pending. A header edited 2 s after the
  stress binary linked made bench_check_binary refuse the next dev run and the
  first 16-node job (8872228): batch header edits before building.
