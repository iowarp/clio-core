# Microbenchmark and CPU-accounting plan (reviewer response)

Two gaps a EuroSys reviewer will raise, and the smallest experiments that
close them:

1. **What does one fault cost, and what does batching buy?** The paper says
   "~110 us round trip for ~6 us of copy" but never measures it. Section 4.4
   (page size) explains its result with per-request latency, so a reviewer
   needs to see that latency.
2. **How much CPU does Eternia's runtime take?** Every byte moves through
   CPU workers that busy-poll. If they use cores the baselines leave idle,
   E1's comparison is only fair once that is shown.

Everything runs on **Aurora, 1 PVC tile per process**, like the rest of the
paper. Record **5 repetitions per point** and plot median with min-max bars.
All binaries: coroc edition, `build_newcoro_aurora.sh` (AOT=1).

---

## Part A -- fault-path microbenchmark (new file, ~300 lines)

**Why a new file:** `clio_gpu_vector_access_bench.cc` measures resident
access only, and `clio_gpu_vector_{flush,overlap,stream}_bench.cc` still call
`FlushAsync`/`AwaitFlush`, which were removed on 2026-08-22. Write
`benchmark/micro/clio_gv_fault_micro.cc` from the kmeans newcoro skeleton
(runtime config writer, `YieldRunner`, `GV_LAUNCH_BOUNDS`, SYCL split). Reuse
kmeans' host code; only the kernels are new.

### A1. Latency and batching (the headline figure)

Kernel: `nblocks` blocks, each owning a disjoint page range. Each block loops:
`CoFetch(0, off, n_pages*epp)` -> `CoHoldPage` of the first element (forces the
wait) -> `UnpinRange`. Cache cold for every call: size the cache to exactly
the in-flight pages so every fetch is a miss; ClearCache between repetitions.

| axis | values |
|---|---|
| pages per fetch (batch) | 1, 2, 4, 8, 16, 32, 64 |
| page size | 64 KB, 256 KB, 1 MB, 4 MB |
| blocks | 1 (latency) and 64 (throughput) |
| host tier | DRAM |

Measure, per fetch: device time from `CoFetch` entry to hold return, using the
device cycle counter already used for E1's communication split (the same
counter, so the numbers compose). Report:
- **latency per request** and **latency per page** vs batch size (1 block);
- **effective GB/s** vs batch size (64 blocks);
- two reference lines on the same axes, same bytes:
  - `memcpy` pinned host -> device (`sycl::queue::memcpy` from USM host), the floor;
  - USM **shared** memory first-touch migration of the same bytes (Intel's
    UVM equivalent), with and without `prefetch` -- the single-node
    "just use unified memory" alternative a reviewer will ask about.

Expected shape (from the 4070 laptop numbers): per-request latency roughly
flat in batch size, so per-page cost falls ~1/n until copy time dominates.
The plot makes the "batching is the lever" claim concrete.

### A2. Where the round trip goes

Same kernel, batch 1 and 64, 1 MB pages, 1 block. Timestamp five points and
report the breakdown as a stacked bar:

1. kernel submit (device clock, just before `Send`)
2. GPU worker pops the ring entry (`Worker::ProcessNewTaskGpu`)
3. handler start (`PodMultiGetBlob` entry)
4. handler end (after the last `PodGetBlob` copy)
5. kernel observes completion (device clock after `AwaitFetch`)

Plus the relaunch gap when the block had parked: from completion flip to the
block running again (`gy::Yieldable` round log already records per-round
time). Reuse the existing profiling hooks where they exist (`CLIO_PUT_PROF`,
the yield driver's `RoundLog()`); add host timestamps only in the handler,
behind an env flag, reported by the bench at exit. Convert device clock to
wall time with one calibration pass at start.

### A3. Remote pages

2 nodes. Same kernel, pages owned by the **other** node (name the tag so the
hash lands remote; verify with a get to confirm owner). Batch 1 and 64, 1 MB.
Report latency and GB/s next to A1's local numbers. This is the per-fault
cost behind E1's communication share and E7's 32-node number.

### A4. Park/relaunch cost

Kernel that parks with zero data movement: fetch of an already-resident page
does not park, so instead await a flush of 1 page. Report ms per round and
rounds per fault for 1, 64 and 1024 blocks. `context-runtime/test/unit/gpu/
spike/spike_*` have the measurement loop; port the pattern.

**Figure for the paper:** one figure, two panels -- (a) A1 per-page latency vs
batch size, lines per page size, with memcpy and USM-shared references;
(b) A2 stacked breakdown for batch 1 vs 64, plus A3's remote bar. Put it at
the start of the evaluation, before E1: every later result is explained by it.

---

## Part B -- runtime CPU accounting (no new code in the benchmarks)

### B1. Cores consumed during E1

Rerun E1's kmeans and Gray-Scott at 4 and 16 nodes (the existing
`pbs_*_aurora.sh` decks) with a sampler beside each rank:

```bash
# per rank, started by the pbs script after the runtime is up
pid=<bench pid>
snap() { for t in /proc/$pid/task/*; do
  echo "$(cat $t/comm) $(awk '{print $14+$15}' $t/stat)"; done; }
snap > cpu_start.txt; T0=$(date +%s.%N)
# ... at the end of the timed region (bench prints a marker line; poll for it)
snap > cpu_end.txt;   T1=$(date +%s.%N)
```

cores used = sum over threads of delta(utime+stime) / CLK_TCK / (T1-T0).
Group threads by role using `comm` (name the runtime's threads if they are
not named yet: `pthread_setname_np` for scheduler, I/O, GPU, net-send,
net-recv workers -- one line each where they are created in
`work_orchestrator.cc`). Report **cores per node used by the runtime, by
role, against the 104 the node has**, and the same for the MPI baseline
(which should be ~1 core per rank).

Run with the E1 config as-is (`first_busy_wait` as used), and once with
busy-waiting disabled, to show how much of it is polling rather than work.

### B2. Does the runtime need those cores?

kmeans and Gray-Scott, 4 nodes: sweep runtime `num_threads` = 2, 4, 8, 16
and pin the process to that many cores plus 1 (`taskset`/`--cpu-bind`), so
the runtime cannot borrow others. Plot iteration time vs cores. If time is
flat down to a few cores, the answer to "is the comparison fair" is a
one-line sentence in E1 with this figure in the appendix: Eternia needs N
cores per GPU, which GPU applications leave idle.

---

## Paper text to write once numbers exist

- Evaluation opener (before E1): "Cost of a fault", 1 paragraph + Figure A.
- E1: one sentence with B1's cores-per-node, pointing at B2.
- Design, "Managing CPU Resources": replace the unmeasured claims with B1/B2.

## Budget

A1-A4: one PBS job, 2 nodes, < 1 hour of allocation.
B1: piggybacks on E1 reruns at 4 and 16 nodes, ~1 hour.
B2: 4 nodes x 4 core counts x 2 workloads x 5 reps, ~1-2 hours.
