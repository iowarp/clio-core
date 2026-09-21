# The clio-coroc paged benchmarks on Aurora (Intel Data Center GPU Max 1550)

Six paged benchmarks, transpiled by `tools/coroc` and compiled with icpx
(oneAPI 2025.3.2) for `spir64_gen -device pvc`. Each runs as ONE single-node
job in the `debug` queue, capped at 90 s inside the queue's 5-minute minimum
walltime, pinned to one tile. Results as of 2026-09-21, commit after 7a7c6a05:

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
Results as of 2026-09-21, commit after 651a3139: all six pass.

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

Two smaller things found on the way and kept: `__nanosleep` was an empty
function under SYCL, so `AllocatePage`'s transient-pressure backoff was 4096
instant retries and a trap; and the device-side fatal latch is device memory
now, mirrored into pinned host memory and printed by a SIGABRT handler, so a
FATAL on Level Zero reports its cause instead of dying as an unexplained
atomic fault.

## Open

- **Composite (two-tile) mode**: lbann faults on an atomic to device memory
  when a root device is used unpinned; the other four pass either way. One
  tile is ALCF's recommended unit and is what the table reports.
- The runtime's `worker.cc` still carries the null-RunContext guards added
  while chasing (2). With the re-fire gone they should be unreachable; they
  stay as diagnostics rather than being removed blind.
- `submit_newcoro_aurora.sh`'s `report` prints the NEWEST log for a skipped
  benchmark, so a `RESULT lammps_md: FAILED` line in a driver run is the
  stale log, not a new run.
