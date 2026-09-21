# The clio-coroc paged benchmarks on Aurora (Intel Data Center GPU Max 1550)

Six paged benchmarks, transpiled by `tools/coroc` and compiled with icpx
(oneAPI 2025.3.2) for `spir64_gen -device pvc`. Each runs as ONE single-node
job in the `debug` queue, capped at 90 s inside the queue's 5-minute minimum
walltime, pinned to one tile. Results as of 2026-09-21:

| benchmark | args                                              | result | notes |
|-----------|---------------------------------------------------|--------|-------|
| gmx       | `--page-kb 128`                                   | PASS   | 3 s, all three gates, 128 faults |
| lbann     |                                                   | PASS   | 5 s, all gates; needs the tile pin (below) |
| weights   | `--repeat 1`                                      | PASS   | 4 s, checksum OK, 437 faults, 309 evictions |
| grayscott | `--data-mb 512 --hbm-mb 128 --repeat 1 --steps 2` | PASS   | 4 s, v_checksum 339533.404295, 256 faults |
| kmeans    | `--data-mb 256 --hbm-mb 128`                      | PASS   | 17 s, checksum OK, 13 320 faults/evictions |
| lammps_md | `--lattice 12 --steps 10`                         | PASS   | 3 s, ballistic gate bitwise; 10 steps in 106 ms |

Sizes are chosen for the 90 s cap, not for the numbers in RESULTS.md, and
every one still exceeds its cache so the paging path is exercised.

## Two nodes

One rank per node (`pbs_newcoro_aurora_2n.sh`, `submit_2n_all_aurora.sh`),
same sizes, each benchmark splitting its problem with `--nodes 2 --node r`,
the runtimes joined by a hostfile from `$PBS_NODEFILE`:

| benchmark | result | notes |
|-----------|--------|-------|
| gmx       | PASS   | all gates on both ranks, 67 faults/rank (half of single-node), 15 s |
| lbann     | PASS   | loss + weight gates, "2 nodes" all-gather path, 322 faults/rank, 43 s |
| weights   | PASS   | both ranks the same global checksum, put_errors=0, 36 s -- once the config composed cte_core at 512.0 |
| grayscott | PASS   | v_checksum bit-identical to single-node on both ranks, 18 s |
| kmeans    | see 5  | crashed in the runtime's network send; fixed, rerun pending |
| lammps_md | OPEN   | runs to completion on both nodes, gate fails; see below |

Storage tiers on Flare and DAOS: `pbs_newcoro_aurora_2n_tier.sh` and
`submit_tier_all_aurora.sh` (16 GB through 8 GB of HBM into 8 GB on a
filesystem), results to follow.

## How to build and run

```
tools/coroc/build.sh                                   # the transpiler, spack LLVM 22
AOT=1 benchmark/build_newcoro_aurora_all.sh            # transpile + AOT for pvc, links against build-fresh
SKIP=lammps_md benchmark/submit_newcoro_aurora.sh      # one job per benchmark, sequential
benchmark/submit_one_aurora.sh weights mytag "" --repeat 1   # one benchmark, any tag
```

`build_newcoro_aurora.sh` links against `$W/build-fresh` by default (the
SYCL configure of this tree, `configure_fresh_aurora.sh`); `CLIO_SYCL_BUILD`
overrides. The runpath is baked in at link time -- see "wrong runtime" below.

The debug queue admits ONE queued job per user, so both submitters wait for
the slot rather than fail. `pbs_newcoro_aurora.sh` takes `BENCH_ZE_MASK`
(tile, default `0.0`, `none` for composite), `BENCH_TRACE=1` (SYCL_UR_TRACE=2)
and `BENCH_EXE` (a differently named binary).

## What was wrong, in the order it was found

1. **Device atomics on host USM.** The gpu2cpu submission queue lived in
   `malloc_host`, and the Max 1550 declares `usm_atomic_host_allocations=0`:
   every benchmark died on the first push (`AtomicAccessViolation` at the
   16 MB queue base, PDE level). Fixed by porting the device ring from the HIP
   backend (`gpu2cpu_init_sycl.cc`): `head_/tail_` in device memory,
   `entries_/ready_` in host memory written with plain stores.

2. **An awaiter that carried `this` across a park (rule R9).** `FlushWait{this}`
   is saved into the frame at a park and restored, not rebuilt, on the
   relaunch; `this` was the by-value DeviceVector parameter of the calling
   coroutine, i.e. private memory, which CUDA happened to hand back at the
   same address every launch and PVC did not. `Ready()` then read another
   block's flush state and re-fired a task slot whose previous submission
   the host was still running; the D2H copy of the new POD zeroed that
   coroutine's RunContext under it (`CoroHandle: null RunContext`; the
   gpu2cpu evlog showed S-after-C on every flush slot). The awaiters now
   hold the block's `BlockTasks` record, and `clio-coroc` rejects an awaiter
   that captures `this` or a local's address (R9, `test/co/bad_awaiter_this.h`).

3. **The wrong runtime.** Two rebuilds linked against the main checkout's
   `build/sycl` -- a runtime from before the ring port -- because the build
   script defaulted `CLIO_SYCL_BUILD` there. The kernels took the legacy
   host-memory queue and faulted exactly like (1), which cost the better part
   of a night of bisecting until `SYCL_UR_TRACE=2` placed the fault address at
   that queue's allocation. The default is now this tree's `build-fresh`.

4. **lammps_md** (three more, in order). IGC's backend segfaulted on the
   list-build kernel under every knob; the trigger was the pair of
   `nl.CoBeginFlush`/`CoEndFlush` awaits as written inside `BuildListCoro`
   (compiling them out was the one change that built all 20 images), and
   moving exactly those two awaits into their own `noinline` coroutines
   works -- an IGC bug on a code shape, layout-sensitive, so the stage split
   that came first is kept too. Then the process died in `exit()`: the
   detached fatal-channel watcher was issuing SYCL copies while the runtime
   unloaded its adapters (an `atexit` stop fixes it), which had been hidden
   behind the `SIGSEGV` handler `FatalMirror()` used to install -- Level
   Zero owns that signal for shared-USM migration, so that handler is gone.
   Finally the ballistic gate failed bitwise at 3-4 ulp: icpx defaults to
   `-fp-model=fast` on both passes, unlike nvcc and clang; the benchmark
   compile is `-fp-model=precise -ffp-contract=off` now.

5. **Device memory on the wire.** kmeans' first cross-node page flush died
   in `zmq_send(0xff000000112e0000, 65536)`: a task bound for the node that
   owns a blob carried the paged vector's frame -- GPU memory -- as its
   bulk buffer, and the transport copies bulk bytes with a host memcpy.
   `SaveTaskArchive::bulk` now stages device-only memory through a host
   buffer the archive owns for the send (`task_archive.cc`). The other
   four never hit it because their cross-node traffic is host memory.

6. **A stale benchmark binary interposing the runtime.** The rerun of (5)
   crashed inside the staging code itself: `staged_.emplace_back` with the
   vector's three pointers holding stack garbage, on libraries that were
   provably rebuilt. The archive's constructor was inline in the header,
   so the kmeans executable -- linked three hours before the member was
   added -- exported its own copy, and the dynamic linker picked it over
   the library's (the executable is first in lookup scope). The old copy
   constructed the old 1152-byte layout; the library's `bulk()` used the
   new 1176-byte one. The special members now live in `task_archive.cc`,
   so the layout has one owner and a stale binary fails to link instead of
   corrupting memory. Same lesson as the task-struct ABI note in AGENTS.md,
   one level up: after changing a header that crosses a library boundary,
   relink every executable, not only the libraries.

7. **Device memory on the wire, the reply.** With (5) and (6) in, rank 0
   died in `LoadTaskArchive::bulk` instead: a page fault on a blob the
   other node owns is a bdev ReadTask whose destination is the GPU frame,
   and the reply's bytes were copied into it with a host memcpy (rank 1
   then hung on its dead peer until the cap). The copy is
   `DeviceAwareMemcpy` now, which stays a plain memcpy for host
   destinations. The grayscott tiering run's rank-1 segfault in memmove
   (stale binary, so (6) or this) has the same shape: at 8 GB per node its
   pages cross nodes, which the 256 MB two-node run never did.

8. **Pages placed on the other node's HBM.** With the crashes gone the
   256 MB two-node kmeans ran past its 90 s cap while the runtime kept
   processing tasks. `CLIO_NET_TRACE=1` on a 64 MB-per-node run: rank 0
   spent 4.1 s of a 6.5 s kernel *serialising* sends -- 2016 sends, 105 MB,
   ~1.8 ms each -- and rank 1 spent 5 ms on the same amount. The 1.8 ms is
   the staged copy from (5): a device frame going out as bulk. It was not
   the transfer engine's remote-owner path (that bounces in the PutBlob
   coroutine and sends host memory, which is rank 1's 5 ms) but the
   placer: cte_core registers every node's bdevs on every node across a
   neighborhood window (default 4), so a GPU page could be written to the
   OTHER node's HBM, a bdev write whose bulk is the frame. Forcing the L0
   copy engines on (`UR_L0_USE_COPY_ENGINE=1` and friends) changed
   nothing. `targets: {neighborhood: 1}` in the two-node compose (both
   job scripts) removed the staged sends entirely (ser 3 ms on both ranks)
   and cut the kernel from 6.5 s to 4.5 s.
   Still open after that: 4.5 s for ~170 remote faults, against 65 ms
   per iteration for the same size on ONE node (`kmeans_1n_64`).

9. **Every SYCL copy on one shared queue.** `sycl_copy_probe` cleared the
   GPU side: a 64 KB host-initiated copy is 10 us pinned, 37 us pageable
   (heap, std::vector or MAP_SHARED memfd alike), with the tile idle or
   running a 64-group spin kernel, whatever queue or copy-engine setting.
   The runtime's own numbers disagreed: the bdev `[bwr]` profile
   (`CLIO_PUT_PROF=1`) put each HBM block write at 3.7-6.1 ms of copy
   *submission* and 2 us of wait, and the scheduler's per-method table
   (on-CPU time, timers pause across a park) had PodMultiGetBlob at
   3.75 ms and bdev Write at 5.9 ms per task. Cause: under SYCL,
   `GpuApi::MemcpyAsync` ignored its stream and ran every "async" copy as
   a synchronous `memcpy().wait()` on the one shared `SyclQueue()`, from
   every worker and the network receive thread at once; DeviceAwareMemcpy
   did the same; and the SYCL init never warmed the I/O stream pool, so
   `BorrowStream` handed out a single queue and every concurrent bdev copy
   yielded waiting for it. Now: MemcpyAsync enqueues on the caller's
   in-order stream (PollSync/StreamQuery wait on it), DeviceAwareMemcpy
   uses a per-thread in-order queue, and the pool is warmed like the HIP
   init does. Two-node 64 MB kmeans: kernel 4.5 s -> 2.9 s, GetBlob
   handler 3 ms -> 0.4-0.75 ms, bdev read await 6.7 ms -> 0.4 ms.

   Still open: the fault handler still spends ~40 ms awaiting a sub-get
   that itself takes under 1 ms, and a GPU submission still waits ~70 ms
   from route to handler end (clio-evlat C-P p50), against 7 ms on one
   node. The next report carries two more channels, the remote round
   trip (SendIn to reply) and the completion event's wait in the parent's
   event queue, to place the remaining 40 ms.

   Also found and kept on the way: PodMultiGetBlob's parked coroutines
   were counted as worker load until they finished (they now release it
   while parked), the device-side `__nanosleep` shim was a counted spin
   at 55 ns per iteration (removed; the GPU spin-waits), and a
   `neighborhood: 1` compose knob keeps cte_core targets node-local.

Two smaller things found on the way and kept: `__nanosleep` was an empty
function under SYCL, so `AllocatePage`'s transient-pressure backoff was 4096
instant retries and a trap; and the device-side fatal latch is device memory
now, mirrored into pinned host memory and printed by a SIGABRT handler, so a
FATAL on Level Zero reports its cause instead of dying as an unexplained
atomic fault.

## Open

- **lammps_md across two nodes**: each rank completes all 640 page
  iterations deterministically, but every owned atom ends one drift
  (dt*v per axis) and one half-kick (dt*g/2) AHEAD of the ballistic
  reference on both nodes -- i.e. one extra integrate launch in the
  multi-node stage-1 loop. Not a readback artefact (MD_SETTLE_MS=2000
  changed nothing) and not the network. The stage-1 gate now judges only
  the node's own z-slab (it compared every global slot before, which read
  as exactly half the energy); the extra launch is unlocated.
- **Composite (two-tile) mode**: lbann faults on an atomic to device memory
  when a root device is used unpinned; the other four pass either way. One
  tile is ALCF's recommended unit and is what the table reports.
- The runtime's `worker.cc` still carries the null-RunContext guards added
  while chasing (2). With the re-fire gone they should be unreachable; they
  stay as diagnostics rather than being removed blind.
- `submit_newcoro_aurora.sh`'s `report` prints the NEWEST log for a skipped
  benchmark, so a `RESULT lammps_md: FAILED` line in a driver run is the
  stale log, not a new run.
