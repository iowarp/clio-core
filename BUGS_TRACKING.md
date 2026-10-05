# Bug triage tracker — iowarp/clio-core open issues

Working branch: `bugfix/oldest-first-triage`, branched fresh from `origin/dev`
(not from the dirty `jaime-issues` branch, so none of its local fixes are
assumed here — every bug below gets validated against this clean base).

Process per bug, oldest created-date first:
1. **Validate**: re-read the issue against the current `dev` tip. Confirm the
   referenced files/symbols/behavior still exist and still misbehave. Note if
   stale (renamed/removed/already fixed) vs. still real.
   - Checked separately: `origin/dev` is the live integration branch (last
     commit 2026-10-05, 1458 commits ahead of `origin/main`'s 2026-08-18 tip).
     `origin/main` only differs by a version-bump release merge — `dev` is
     the correct base.
   - The remote also already has many issue-numbered branches
     (`915-aggregateout-no-copy`, `927-rwlock-exclusion`,
     `929-dev-to-main-ci-flake`, `919-fix-flaky-ci-tests`,
     `923-924-startup-diagnostics`, `fs-sieve-flush-and-hang-repro`,
     `gpu2cpu-devicemem-queue`/`gpu2cpu-pinned-host-ring`, and others) — these
     are leads that in-progress or landed work may already exist for that
     bug; check their tip and any PR before assuming the issue needs fresh
     work here.
2. **Fix**: smallest correct change on this branch, one commit per bug
   (reference the issue number in the commit subject).
3. **Record** the outcome in the Status column below and in the Notes.

| # | Created | Title (short) | Status | Notes |
|---|---|---|---|---|
| 363 | 2026-03-21 | Install feedback: optional build features may block setup | **fixed (adjacent regression)** | Original repro (recipe.yaml/install.sh, WRP_CTE_ENABLE_COMPRESS etc.) is stale: those files/flags don't exist on dev anymore; CLIO_CORE_ENABLE_TESTS and CLIO_CTE_ENABLE_COMPRESS already default OFF. Found and fixed a live sibling bug in the same file: installers/conda/build.sh passed `-DWRP_CORE_ENABLE_CONDA=ON` but CMakeLists.txt defines `CLIO_CORE_ENABLE_CONDA` — stale rebrand prefix meant the flag was silently ignored. |
| 503 | 2026-06-07 | Re-enable disabled CTE distributed execution validation assertions | pending | |
| 579 | 2026-06-18 | cte_tag/cte_query force-net tests hang in CI | pending | |
| 597 | 2026-06-20 | xfstests generic/quick 606/639 pass — gaps & bugs | pending | |
| 641 | 2026-06-27 | Fallback-runtime ctests hang under Boost coroutines | pending | lead: branch `fix/gpu-task-hangs`? verify relevance |
| 646 | 2026-06-29 | Allocators run dry under GB-scale FUSE churn | pending | |
| 706 | 2026-07-08 | Flaky reorganize-to-disk allocation failure | pending | |
| 722 | 2026-07-10 | Daemon spins forever requeueing undeliverable response | pending | jaime-issues branch (uncommitted/local, not this base) has a commit claiming to fix this — re-verify independently here |
| 725 | 2026-07-11 | stale restart state kills fresh daemons silently | pending | same jaime-issues lead as #722 |
| 768 | 2026-07-18 | Windows TCP IPC latency + auto-select fastest IPC | pending | |
| 791 | 2026-07-22 | bdev leaks 64KiB block on Windows | pending | |
| 793 | 2026-07-22 | native GetBlob can hang holding GIL | pending | jaime-issues lead (local, uncommitted work claims GIL release fix) |
| 794 | 2026-07-22 | Tiered tests pass vacuously (ReorganizeBlob) | pending | |
| 796 | 2026-07-22 | CTE WAL does not persist blob timestamps | pending | |
| 800 | 2026-07-22 | Windows link-lock flake (~22% of runs) | pending | issue thread says mitigated (23%→7%) but root cause explicitly still open |
| 803 | 2026-07-22 | Sanitizer CI jobs cannot fail (ubsan 151/259 defects, still green) | pending | |
| 808 | 2026-07-23 | adapters(linux): FUSE smoke fails to bind port 9413 | pending | |
| 809 | 2026-07-23 | HDF5 VOL descriptor lingers past H5Fclose() | pending | |
| 836 | 2026-07-27 | kvhdf5 test targets fail post-link (CUDA+MPI) | pending | |
| 848 | 2026-07-28 | Windows CI one-different-test-per-run flake cluster | pending | lead: branch `940-singleton-test-windows-timeout`? |
| 853 | 2026-07-28 | Cluster Tests: CTE distributed SHM-cache tests fail | pending | |
| 856 | 2026-07-29 | Leader recovery crashes new SWIM leader (free(): invalid pointer) | pending | |
| 859 | 2026-07-29 | CTE pinned zero-copy read views (pin missing) | pending | lead: branch `943-reorganize-del-read-pin` |
| 863 | 2026-07-29 | ReorganizeBlob loses blob data under capacity pressure | pending | |
| 877 | 2026-07-31 | Windows shm names collide across runtime instances | pending | jaime-issues lead (local, uncommitted) |
| 882 | 2026-07-31 | icx (windows-2025) CI leg chronically flaky | pending | |
| 892 | 2026-08-04 | Cross-node data path bound at ~116 MB/s | pending | |
| 893 | 2026-08-04 | Sub-128KB deferred puts invisible cross-node | pending | |
| 896 | 2026-08-04 | Put to SWIM-dead node hangs submitting client forever | pending | |
| 907 | 2026-08-04 | test_fuse_ops hangs (zero output) on Windows CI | pending | |
| 915 | 2026-08-05 | AggregateOut delegates to Copy(): corrupts origin | pending | lead: branch `915-aggregateout-no-copy` exists upstream — check its tip/PR before redoing |
| 919 | 2026-08-05 | Fix flaky CI tests (bdev_fragmentation, cfs_rename, ...) | pending | lead: branch `919-fix-flaky-ci-tests` exists upstream |
| 924 | 2026-08-05 | Daemon dies at startup: pool created then HasPool says not found | pending | lead: branch `923-924-startup-diagnostics` exists upstream |
| 927 | 2026-08-05 | ctp::RwLock allows concurrent reader/writer | pending | lead: branch `927-rwlock-exclusion` exists upstream, plus `rwlock-batched-fairness` |
| 929 | 2026-08-06 | dev→main PR flakes ~47% (duplicated PoolManager + SWIM/fiber SEGV) | pending | heavy investigation already (10 comments); Class A fixed+merged, Class C already fixed pre-filing, Class B root-caused but NOT fixed as of last read. Many bisect/probe branches upstream (`bisect-*`, `hb-instrument*`, `probe-bad-*`, `probe2-*`, `cluster-bisect-revert-heartbeat`) — read their latest state before redoing the bisection. |
| 991 | 2026-08-18 | cr_shutdown_bt_* battletests intermittently hang on macOS | pending | |
| 995 | 2026-08-18 | CTE FUSE loses regions from concurrently growing files | pending | lead: branch `fs-sieve-flush-and-hang-repro` (separate worktree exists: core-fs-sieve) |
| 1000 | 2026-08-18 | DestroyPool orphans periodic tasks into RouteTask retry storm | pending | lead: branch `fix-routetask-nonworker-retry-drop` |
| 1028 | 2026-08-25 | clio FUSE: O_DIRECT vs page-cache/mmap coherence | pending | |
| 1029 | 2026-08-25 | clio FUSE: rmdir returns ENOTEMPTY (generic/070) | pending | |
| 1030 | 2026-08-25 | cte_replication_persist_integration loses/zeros disk replica | pending | |
| 1035 | 2026-08-27 | dev CI audit: redness is pre-merge gap, not flakiness | pending | |
| 1039 | 2026-08-27 | Cluster Tests: CTE cache coherence 4-node step times out | pending | |
| 1043 | 2026-08-27 | Windows checkout: schannel SEC_E_UNTRUSTED_ROOT not retried | pending | |
| 1049 | 2026-08-28 | Runtime wedges in unbounded response-requeue loop | pending | jaime-issues lead (local, uncommitted) |
| 1050 | 2026-08-28 | Windows: CTE client crashes (0xC0000409) under pytest | pending | |
| 1059 | 2026-08-28 | Record-scoped permanent GetBlob failure on committed data | pending | jaime-issues lead (local, uncommitted) |
| 1065 | 2026-08-29 | Client pool never releases connection (1024-client stall) | pending | PR #1168 reproduced this with a WILL_FAIL test, not fixed |
| 1086 | 2026-09-11 | gpu2cpu producer-only design fails on Intel GPUs | pending | lead: branches `gpu2cpu-devicemem-queue`, `gpu2cpu-pinned-host-ring` |
| 1096 | 2026-09-27 | Python binding holds GIL for whole RPC | pending | jaime-issues lead (local, uncommitted) |
| 1100 | 2026-10-01 | Windows bdevs allocate up front (no lazy allocation) | pending | jaime-issues lead (local, uncommitted) |
| 1156 | 2026-10-04 | safe_bdev disk-fault tests fail on macOS (rc=2, 0 bytes) | pending | |
| 1159 | 2026-10-04 | clio-fs small-file create rate falls 11x (1→6 nodes) | pending | |
| 1160 | 2026-10-04 | safe_bdev: 68-byte inode write waits behind 1 MiB stripes | **fixed upstream** | Commit 5806605ca ("safe_bdev: a write's parity update is batched... (#1160)"), already on this branch's base (origin/dev tip, merged hours before this session) — batches GatherSurvivors/ReadReplacedBytes/DeltaEncodeStripe/StoreDegradedParity into single round trips instead of per-row serial ones. No further code change needed here; issue just needs closing on GitHub once labeling access exists. |
| 1167 | 2026-10-05 | CTE core put refills lost range, stays marked lost | pending | newest bug, filed the day before this session |

**Note on "leads":** branch names and the jaime-issues commit log are starting
points only, not proof of a merged fix. Each must still be validated — read
the branch's actual diff/PR state and confirm it addresses the issue's root
cause — before marking a bug fixed.
