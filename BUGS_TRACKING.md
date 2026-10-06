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
| 856 | fixed on dev | leader_elect 4-node docker harness: 6/6 pass, all four node runtimes alive after the restart phase in every run (the crash used to hit node2 10/10). CI step re-enabled here. |
| 863 | fixed on dev | Data loss fixed by 756097c94 (place the new copy before freeing the old). Hazard 2 (drift) re-tested with the issue's own repro (kRounds=4, churn burst 8): 2 runs x 4 rounds, every fill-to-brim + reorganize round succeeded; the churn trigger (blob layout seen empty mid-move) never fires any more because place-then-swap removed that window. |
| 877 | in progress elsewhere | Asks 1 and 3 on dev; fail-closed ServerInit is uncommitted in the jaime-issues checkout (another agent). |
| 882 | no longer occurring | Same window as #848: the icx (windows-2025) leg's only failure in 35 runs was cr_safe_bdev_disk_fail_tests (#1156 YAML, fixed here); no wedge, no flake. |
| 893 | fixed on dev | 2d769d309. |
| 896 | fixed on dev | fe21df453 (regression test not added; multi-node). |
| 907 | does not reproduce; CI exclusion dropped here | After installing the WinFsp developer feature: native Windows Debug build, test_fuse_ops 14/14 cases, ctest 8/8 passes (~1 s each), also with #1039's fix reverted. CI exclusion removed in its own commit; CI is where it hung, so CI confirms. Local runs need WinFsp\bin on PATH (else 0xC0000135). |
| 915 | fixed on dev | fb735bdb6 (PR #1027). |
| 919 | fixed on dev | PRs #920, #932. |
| 924 | fixed on dev | 499b7c2fb (PR #932). |
| 927 | fixed on dev | cebb678bb (PR #1032). |
| 929 | fixed on dev (class B) | A/D fixed earlier; class B (leader-election node SIGSEGV) no longer reproduces: 6/6 runs, no node crash. Cluster-tests leader_elect step re-enabled here. Local runs used 4 threads / 256 MB segments to fit the docker VM. |
| 991 | fixed here | Original 1800 s hangs no longer occur on macOS. Residual: cr_shutdown_bt_transports exit 137 (2/8 dev runs) because stop SIGKILLed a live-but-slow runtime 10 s after SIGTERM while the runtime bounds its own exit at 20 s. 28212434c waits 25 s; all 7 cr_shutdown_bt_* pass (bt_wedged still escalates). |
| 995 | likely fixed on dev | FUSE in a privileged container: concurrent growing-file writers (4 KiB-1 MiB chunks, 1 and 4 workers) 0 bad chunks; kernel shallow clone through the mount + git fsck clean; 64 MiB restart test clean in all 4 modes. Matches 7834cd773 (parallel puts creating the same blob lost leading blocks). Original report was aarch64 (not testable here). |
| 1000 | fixed on dev | 3c2f493de. |
| 1028 | does not reproduce here | 900+ runs of the generic/729 case (pinned CPU, hogs, random handler delays) and ~8 harness rounds of 209/451/647/729: all pass. Host kernel is WSL 6.6; FUSE direct-I/O semantics changed in 6.8+ (what CI runs) -- needs a 6.8+ host to reproduce. |
| 1029 | fixed here (adjacent defect) | Not reproduced (78 runs); CloserBarrier hypothesis refuted. b96113d51: a failed listing is now an error, not an empty dir (client readdir, server CollectDir after 8 races, ReaddirPath errno), plus rmdir-ENOTEMPTY diagnostics. 16 cfs/fuse unit tests + 3/3 real mounts pass. generic/070 stays quarantined until CI confirms. |
| 1030 | fixed on dev | bfae06809, 1ae09daef, 79b0b2ea5, 13ee44409. Acceptance loop run here: `cte_replication_persist_integration` 20/20 passed. |
| 1039 | fixed here | Root cause: the runtime silently DROPPED a client's request that arrived before its pool's Create finished (IpcCpu2Cpu::RecvIn and the ZMQ RecvIn); the coherence harness starts clients a fixed 5 s after the runtime, and a slow cte_main compose (~4.7 s) overran it, so early clients hung forever. af43335a6 holds such requests until PoolManager::IsClientAdmissible (dropped with a log after 120 s). New late-pool tests fail before / pass 10/10 after; 1 s-start cluster 0/3 before, 10/10 after; stock cluster 20/20; FUSE smoke 3/3. |
| 1049 | fixed on dev | d1dbb516f + 297bf813a. |
| 1043 | fixed here | Windows checkout now retried once after a failure (all 4 Windows jobs); plus the 3 other unretried Windows network steps (oneapi-ci clone, WinFsp download, adapters vcpkg bootstrap). Download loop tested locally. |
| 1050 | does not reproduce | Native Windows Debug build + Python 3.13 extension: a daemon + CTE client put/get flow (4 tags x 200 round trips, then stop) hosted by pytest with default fd capture 5/5 clean, with -s 3/3, bare script 3/3; daemon exit 0x0 every time, never 0xC0000409. The reported daemon-side crash loop fits the restart-on-corrupt-restore crash fixed by 3d7554a15 (#725) and the Windows bdev offset bug (#1059); confirming against clio-agent's own suite is still worthwhile. |
| 1059 | fixed on dev | 7118e904d + 3d7554a15. |
| 1096 | fixed on dev + here | 4bafb3c68; leftover `clio_init`/`clio_finalize` GIL release in fcd79070d (checked: a Python thread keeps running while clio_init blocks). |
| 1100 | fixed on dev | 7118e904d. |
| 1156 | fixed here | 4e93ca9d1: POSIX AIO EAGAIN (macOS request cap) completes synchronously; safe_bdev test YAML single-quoted for Windows. Test fails without the fix. ctest stays Linux-only until CI confirms macOS/Windows. |
| 1160 | fixed here | Parity batching (5806605ca) was on dev; the stripe-aware allocation (7e554a96) and one-write-per-consecutive-blocks put (7f4a095d5) existed only on gpu-vector-rewrite and are cherry-picked here (cluster-measured by their author: 47 -> 50 MiB/s, slow-stripe phases roughly halved). Locally: safe_bdev / cte_core / tiered / sieve / leak-stress suites 30/30; safe-bdev docker smoke (write 32 MB via clio-fs, replace a member, recover, verify) PASSED. |
| 1180 | diagnostics fixed here | 6c739c79e: the stall warning names every pool create in flight (name, module, age). Root cause still unknown (rare, 1/21). |
| 1186 | fixed on branch bugfix/new-issues-1186 | 9fbe9d620: checkpoint pool id 566.0; static_assert test keeps CTE well-known ids distinct. |
| 1187 | fixed on branch bugfix/new-issues-1186 | 700ed833d + db946b6f2: MPI-IO errors mapped to MPI classes, real MPI_File_sync, close reports latched errors, MPI_Status filled. Tests pass 6/6 (the earlier zero reads were the test deleting the live bdev file). |
| 1188 | fixed on branch bugfix/new-issues-1186 | d60119322: SyncFd runs SyncTag/size/parent-dir sync (shared with FUSE). cfs_fsync_durable, cfs_shm_read, fuse_cte_*, fuse_ops pass. |
| 1189 | fixed on branch bugfix/new-issues-1186 | 14a4319fc: one preset encoding, GetLibraryId model ids, real decompress time, model paths from YAML, dynamic mode in interposer PutBlob, kDecompress gated on transform flag. Test run standalone (libzfp SYCL init crashes in the devcontainer). |
| 1190 | fixed on branch bugfix/new-issues-1186 | 14d2e34d1: CTE backing compiled, async UAF fixed with CUDA events, errors propagated, blobs deleted on destroy. GPU test passes (RTX 5080). |
| 1191 | fixed on branch bugfix/new-issues-1186 | c45f55f93: cuFile offsets honored, descriptor copied, caller fd not closed, CMake target fixed. GPU test fails before / passes after. |
| 1192 | needs maintainer decision | Reaper disabled deliberately in 7c37a1bed; re-enabling risks unmapping a dead client's segment mid-PutBlob and pid checks across pid namespaces. |
| 1193 | fixed on branch bugfix/new-issues-1186 | b764fb4f0: append fsynced; replay sorted by time (per-file append order kept). Test cr_address_table_wal. |
| 1194 | fixed on branch bugfix/new-issues-1186 | b28836d7c: Spack variants map to real CLIO_* options; adapters conflict with ~elf; CI asserts libclio_cte_posix.so. |
| 1195 | fixed on branch bugfix/new-issues-1186 | 01775c5fb: conda variants use the release preset + IOWARP_CMAKE_ARGS; RPM license BSD-3-Clause; WRP_ flags renamed. |
| 1197 | fixed on gpu-vector-rewrite | 32ed13bcb (PR #1198): Gone verdicts park until SendIn finishes transmitting. |

## Not bugs (features, designs, CI process)

26, 200, 223, 239, 240, 241, 250, 252, 266, 267, 268, 270, 305, 308, 310,
324, 333, 351, 369, 374, 435, 443, 484, 513, 525, 526, 539, 551 (done on
dev), 604, 612, 613, 637, 688, 693, 694, 695, 700, 710 (done on dev), 713,
770, 771, 787, 826, 859, 892, 933, 961, 966, 968, 986, 999, 1008 (done on
dev), 1013, 1015 (done on dev), 1035, 1036, 1043, 1044, 1060, 1076, 1086,
1111, 1112, 1159.
