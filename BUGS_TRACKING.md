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
| 503 | open (test debt) | `TEMP-DISABLED (#503)` assertion; needs a routing or test-design decision and the 4-node docker suite. |
| 579 | fixed on dev; CI exclusion dropped here | Root cause (exit-time `zmq_ctx_term` hang) fixed by d7598a6e4/#627. Both tests ran 20x each locally (Debug) with no hang; 12916cfd9 re-enables them in the leak-check job (not run locally under that preset). |
| 597 | mostly fixed on dev | B1 d33aff8fb, B2 60f89b980/3e5126dbb, fsx 38a8174a4/ccc6328bf, mmap/timestamps/inode numbers landed. Left: exportfs (feature). |
| 641 | obsolete | The fallback-runtime feature was removed; its tests only exist with `CLIO_CORE_ENABLE_RUNTIME_FALLBACK=ON`. Nothing to fix. |
| 646 | fixed here | BuddyAllocator never merged adjacent free pages, so a mostly-free heap failed 1 MB requests (894 MB churn -> null). Now coalesces free pages before failing. All three hidden `[fuse_repro]` cases pass (also the ProducerConsumerAllocator one: 2910 nulls -> 0) and are un-hidden (~1 s). |
| 706 | open (needs repro) | Partly explained (64 MB /dev/shm, over-subscribed tier per #794, reorganize rewrite 756097c94/879b47bb6); needs a fresh run in the constrained deps-cpu container. |
| 722 | fixed on dev | Bounded retry + drop + evict (d1dbb516f, 297bf813a). Log size cap is extra protection pending in the jaime-issues working tree. |
| 725 | fixed on dev | 3d7554a15 (restore hardening), 297bf813a (fail-loud port cluster), 5f597ac99 (docs). |
| 768 | fixed here (Linux-neutral) | Net worker lanes were registered before they existed (null), so EnqueueNetTask never woke the net worker. Fixed in f6cb94637. Linux PutGet/TCP unchanged (~2 ms/op both); the Windows tick-bound latency is not measured with the fix. |
| 791 | open (no repro) | Stale-cache measurement and alignment mismatch fixed on dev (7eac0527e, 7a76b4b1e); the exact 64 KiB residue is unexplained; Windows only. |
| 793 | fixed on dev | 4bafb3c68: GIL released around every RPC, `wait(max_sec)`. |
| 794 | fixed here (tests) | 6ff6cb70c: tiered tests assert actual placement from SHM records. Locally 64/64 in DRAM, 64/96 on file then 96/96 back in RAM. CI `-E` exclusions left until a CI run confirms them. |
| 796 | fixed here | 726a13ea9: WAL create/extend records carry a wall-clock stamp, snapshot entry type 6 carries times, restore converts to the new boot's steady clock. `cr_cli_cte_BlobTimes`: 0 hits without the fix, 2 in order with it. |
| 800 | open (CI infra) | Mitigated 23% -> 7% by build retry; the lock holder is unidentified; needs a Windows CI diagnostic run. |
| 803 | open (CI) | Findings summary landed (bcd773823), UBSan cast class fixed (#1094). Jobs still cannot fail (`|| true`, `exit 0`); gating needs a fresh sanitizer baseline. |
| 808 | fixed here | 4eeefd330: port guard probes every TCP state on base, base+1, base+3; start retry. Verified free/LISTEN/TIME-WAIT cases in the container. |
| 809 | open | Root cause unknown; mitigation holds; macOS only. |
| 848 | partly fixed on dev | RwLock (cebb678bb), DLL-copy race, detached_spawn fixed; Windows startup wedge remains. |
| 853 | fixed on dev | Cluster Tests run 37173255855 passes both steps. |
| 856 | partly fixed on dev | PR #918; remaining SIGSEGV tracked in #929; needs the leader_elect docker harness. |
| 863 | partly fixed on dev | Data-loss path fixed (756097c94); capacity drift never confirmed. |
| 877 | in progress elsewhere | Asks 1 and 3 on dev; fail-closed ServerInit is uncommitted in the jaime-issues checkout (another agent). |
| 882 | partly fixed on dev | Same startup-wedge class as #848. |
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
