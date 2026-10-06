# Bug triage tracker — iowarp/clio-core open issues

Working branch: `bugfix/oldest-first-triage`, rebased on `origin/dev`
(b7d291934, 2026-10-05). Every open issue was re-read and classified; bugs
were checked against dev and against the sibling triage branches
(`jaime-issues`, `hyoklee-issue-triage`, issue-numbered branches) before any
work. Verification ran in the `iowarp/clio-core-devcontainer` image (Linux,
Debug, stackless coroutines) unless noted.

Legend: **fixed here** = commit on this branch, verified locally;
**fixed on dev** = already landed, issue only needs closing by hand (closing
keywords do not fire on dev PRs); **open** = still real, not fixed here, with
the reason.

## Bugs

| # | Status | Notes |
|---|---|---|
| 363 | fixed here | Stale `WRP_` flag names in installers (189f41bd0) + `installers/check_flag_names.mjs` regression check. |
| 503 | fixed here | Reproduced on the 4-node docker cluster: ops DID route cross-node, but a node running a task routed to it never set completer_ (only RouteLocal did), so all reported 0. Fixed in RouteTask's routed-task path; assertions re-enabled; avg completer 0 -> 1.75. Note: on this host the cluster needs a reduced config (4 threads, small segments/tiers) or nodes are OOM-killed. |
| 579 | fixed on dev; CI exclusion dropped here | Root cause (exit-time `zmq_ctx_term` hang) fixed by d7598a6e4/#627. Both tests ran 20x each locally (Debug) with no hang; 12916cfd9 re-enables them in the leak-check job (not run locally under that preset). |
| 597 | mostly fixed on dev | B1 d33aff8fb, B2 60f89b980/3e5126dbb, fsx 38a8174a4/ccc6328bf, mmap/timestamps/inode numbers landed. Left: exportfs (feature). |
| 641 | obsolete | The fallback-runtime feature was removed; its tests only exist with `CLIO_CORE_ENABLE_RUNTIME_FALLBACK=ON`. Nothing to fix. |
| 646 | fixed here | BuddyAllocator never merged adjacent free pages, so a mostly-free heap failed 1 MB requests (894 MB churn -> null). Now coalesces free pages before failing. All three hidden `[fuse_repro]` cases pass (also the ProducerConsumerAllocator one: 2910 nulls -> 0) and are un-hidden (~1 s). |
| 706 | fixed on dev; CI exclusion dropped here | No longer reproduces in its own deterministic environment: deps-cpu, --cpus=2, 64 MB /dev/shm, all three affected tests 20x each = 60/60 pass (reorganize place-then-swap 756097c94 + #794 placement asserts). Linux exclusion removed; macOS exclusions kept (not verifiable here). |
| 722 | fixed on dev | Bounded retry + drop + evict (d1dbb516f, 297bf813a). Log size cap is extra protection pending in the jaime-issues working tree. |
| 725 | fixed on dev | 3d7554a15 (restore hardening), 297bf813a (fail-loud port cluster), 5f597ac99 (docs). |
| 768 | fixed here (Linux-neutral) | Net worker lanes were registered before they existed (null), so EnqueueNetTask never woke the net worker. Fixed in f6cb94637. Linux PutGet/TCP unchanged (~2 ms/op both); the Windows tick-bound latency is not measured with the fix. |
| 791 | fixed on dev | 7eac0527e (fresh StatTargets before reading) + 7a76b4b1e (alloc/free alignment, #798). Native Windows Debug: cte_bdev_leak_stress_force_net 10/10. Windows CI on dev: last 5 runs x 7 jobs = 35 passes, 0 failed attempts (no retry masking). |
| 793 | fixed on dev | 4bafb3c68: GIL released around every RPC, `wait(max_sec)`. |
| 794 | fixed here (tests) | 6ff6cb70c: tiered tests assert actual placement from SHM records. Locally 64/64 in DRAM, 64/96 on file then 96/96 back in RAM. CI `-E` exclusions left until a CI run confirms them. |
| 796 | fixed here | 726a13ea9: WAL create/extend records carry a wall-clock stamp, snapshot entry type 6 carries times, restore converts to the new boot's steady clock. `cr_cli_cte_BlobTimes`: 0 hits without the fix, 2 in order with it. |
| 800 | no longer occurring | Scanned the last 48 Windows CI runs (~264 builds): the BUILD-RETRY fired 6 times in 3 runs, and every one was a genuine compile error on gpu-vector-rewrite (setenv / unistd.h on MSVC), not a link lock; 0 link-lock failures vs ~9 expected at the issue's 3.6%/job rate. Holder never identified; Defender exclusions + retry remain in place. |
| 803 | fixed here | Baseline from dev CI: asan and ubsan 330/330, 0 tests with defects (3 runs); msan 13 with defects. asan/ubsan now gate on defects, failed tests or an incomplete run; msan stays a report. Gate logic checked against real dev logs + synthetic cases. |
| 808 | fixed here | 4eeefd330: port guard probes every TCP state on base, base+1, base+3; start retry. Verified free/LISTEN/TIME-WAIT cases in the container. |
| 809 | fixed here | Root cause found on Linux with strace: the VOL connector closes synchronously, but the runtime's bdev health poll ran popen("df ..."), forking a shell that inherited the .h5 descriptor (HDF5 opens without O_CLOEXEC) and so its flock, which outlived H5Fclose. Device lookup now stat/statfs-based (89f8617fd): no execve during a VOL test (was sh/df/tail/awk), same device as df. Locking mitigation removed in a separate commit; macOS confirmation is CI-only. |
| 848 | no longer occurring | Causes fixed on dev: RwLock reader/writer overlap (cebb678bb, #927), DLL-copy race, detached_spawn. The remaining startup-wedge signature ('inline retry exhausted' / 'RouteLocal returned 4') appears in 0 of the last 35 Windows CI runs; the only test failure in that window was cr_safe_bdev_disk_fail_tests on gpu-vector-rewrite, i.e. the #1156 YAML bug fixed here. |
| 853 | fixed on dev | Cluster Tests run 37173255855 passes both steps. |
| 856 | partly fixed on dev | PR #918; remaining SIGSEGV tracked in #929; needs the leader_elect docker harness. |
| 863 | partly fixed on dev | Data-loss path fixed (756097c94); capacity drift never confirmed. |
| 877 | in progress elsewhere | Asks 1 and 3 on dev; fail-closed ServerInit is uncommitted in the jaime-issues checkout (another agent). |
| 882 | no longer occurring | Same window as #848: the icx (windows-2025) leg's only failure in 35 runs was cr_safe_bdev_disk_fail_tests (#1156 YAML, fixed here); no wedge, no flake. |
| 893 | fixed on dev | 2d769d309. |
| 896 | fixed on dev | fe21df453 (regression test not added; multi-node). |
| 907 | open | Windows + WinFsp hang, no root cause. |
| 915 | fixed on dev | fb735bdb6 (PR #1027). |
| 919 | fixed on dev | PRs #920, #932. |
| 924 | fixed on dev | 499b7c2fb (PR #932). |
| 927 | fixed on dev | cebb678bb (PR #1032). |
| 929 | partly fixed on dev | A, D fixed; B (leader recovery SIGSEGV) open, needs the 4-node harness. |
| 991 | needs repro | No 1800 s hang in recent macOS runs; `cr_shutdown_bt_churn` fails fast instead. |
| 995 | open (needs FUSE repro) | Candidate 002b51b4d is a no-op: since #1007 an open sieve page counts in `pending_count_`, so the early return never skipped one. Added a contract test (422596b40) that AwaitPendingPuts drains an open page. |
| 1000 | fixed on dev | 3c2f493de. |
| 1028 | open | FUSE O_DIRECT/page-cache coherence; design-sized (#1060 §3.3). |
| 1029 | open | Hypothesis: rmdir lacks the closer barrier; issue requires a CI-verified fix. |
| 1030 | fixed on dev | bfae06809, 1ae09daef, 79b0b2ea5, 13ee44409. Acceptance loop run here: `cte_replication_persist_integration` 20/20 passed. |
| 1039 | not recurring | No coherence timeout on dev since 09-26. |
| 1049 | fixed on dev | d1dbb516f + 297bf813a. |
| 1050 | open | Windows + pytest crash, no repro on current dev. |
| 1059 | fixed on dev | 7118e904d + 3d7554a15. |
| 1096 | fixed on dev + here | 4bafb3c68; leftover `clio_init`/`clio_finalize` GIL release in fcd79070d (checked: a Python thread keeps running while clio_init blocks). |
| 1100 | fixed on dev | 7118e904d. |
| 1156 | fixed here | 4e93ca9d1: POSIX AIO EAGAIN (macOS request cap) completes synchronously; safe_bdev test YAML single-quoted for Windows. Test fails without the fix. ctest stays Linux-only until CI confirms macOS/Windows. |
| 1160 | partly fixed on dev | 5806605ca batches parity updates; stripe-aware allocation only on gpu-vector-rewrite. |
| 1180 | diagnostics fixed here | 6c739c79e: the stall warning names every pool create in flight (name, module, age). Root cause still unknown (rare, 1/21). |

## Not bugs (features, designs, CI process)

26, 200, 223, 239, 240, 241, 250, 252, 266, 267, 268, 270, 305, 308, 310,
324, 333, 351, 369, 374, 435, 443, 484, 513, 525, 526, 539, 551 (done on
dev), 604, 612, 613, 637, 688, 693, 694, 695, 700, 710 (done on dev), 713,
770, 771, 787, 826, 859, 892, 933, 961, 966, 968, 986, 999, 1008 (done on
dev), 1013, 1015 (done on dev), 1035, 1036, 1043, 1044, 1060, 1076, 1086,
1111, 1112, 1159.
