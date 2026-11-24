# Prefetching on gpu_vector: workload-specific tier hints driven by the yield state

## 1. What "prefetch" means here, and what it does not

There are two completely different things a paged vector could mean by
prefetching, and conflating them is how this gets designed wrong.

**Not this:** pulling a page into the GPU page cache ahead of the demand
fetch. The vector already has that verb — `BeginFetch` / `AwaitFetch` — and
the kernel is the only thing that can use it correctly, because only the
kernel knows which frames it still holds pins on. A host-side agent that
claimed frames behind the kernel's back would evict pages a block is still
reading. `Vector::Prefetch` (host-side, table-stuffing) exists for warm-up
and stays what it is.

**This:** moving a page UP THE TIER STACK before the demand fetch asks for
it. The demand fetch will still happen, still block-collectively, still under
the kernel's control — but it will be served from HBM instead of from NVMe.
Nothing about the cache's frames, pins or eviction changes. What changes is
which bdev the CTE reads when the fault arrives.

That is entirely a CTE-side decision, it is expressed as a blob score, and
the CTE already has a batched API for it: `PodMultiScoreTask` →
`Runtime::PodMultiScore` → `PodReorganizeBlob` per record →
`ReorganizeBlobInternal`, which re-runs the DPE and physically moves the
blob's blocks when the score moves by more than
`performance_.score_difference_threshold`. Up to `kPodMultiMax` (64) records
per task.

So: **a prefetch is a bulk rescore, and a prefetcher is a function from "where
is this block in its computation" to "which pages should be hot, and which
should stop being hot".**

Demotion is not an afterthought. A tier is a fixed budget; promoting page P
into HBM means something leaves HBM, and if the prefetcher does not say what,
the CTE picks — and it picks by frecency, which is a statement about the past.
The prefetcher knows the future. It should spend it on both ends.

### The score vocabulary

Scores are clamped to `[0, 1]` by `ReorganizeBlobInternal`, and MaxBwDpe
splits tiers on `target_score <= blob_score`, ranking the preferred group
descending — so **higher score = faster tier**, given the Gray-Scott config's
tier scores (HBM 1.0, host RAM 0.2, NVMe 0.0). The prefetcher's three-value
vocabulary is therefore:

| name | default | meaning |
|---|---|---|
| `hot`  | 1.0 | will be read within the lookahead window — get it to HBM |
| `warm` | 0.2 | out of the window but read again this step or next — host RAM |
| `cold` | 0.0 | done with for a long time — let it spill to storage |

These are policy, not law: they are constructor arguments, because a run with
no NVMe tier wants `cold` = 0.2 (there is nowhere lower to go, and pushing to
0.0 would be a rescore that moves no bytes and still costs a task).

### A finding that fell out of building this: the fast tier was unreachable

The vector writes every page at blob score **0.5** (`kVectorBlobScore`, at
both `Add()` call sites in `SubmitFetch` and `SubmitFlushRanges`). MaxBwDpe
splits tiers on `target_score <= blob_score`, so a tier scored **above** 0.5
is excluded from the preferred group for every page the vector writes.

In the Gray-Scott config -- HBM 1.0, host RAM 0.2, storage 0.0 -- that means
**pages never land in HBM at all**. It is not subtle once looked for: the
benchmark's own TIER SPLIT line has been printing

    kHBM used=0MiB cap=64MiB remain=64MiB   <-- nothing landed in the fastest tier

and the source comment above the tier config asserts the opposite ("the
vector puts at blob score 1.0, so the HIGHER score is the preferred tier"),
which is what kept the printed warning from being read as one. Measured, on
identical settings:

| prefetch | kHBM used | v_checksum |
|---|---|---|
| `none`   | 0 MiB  | 671068.413736 |
| `gs`     | 52 MiB | 671068.413736 |

So on this configuration a prefetch hint is not merely an optimization of
tier placement -- it is the only thing that puts anything in the fast tier at
all. That makes the `hbm`-only control row of the evaluation load-bearing:
without it, the whole sweep could be measuring "prefetching turns HBM on"
rather than "prefetching places pages better".

Two things follow, and only the first is done here: the score is now a named
constant instead of a literal repeated at two call sites, so a prefetcher can
know what score a page starts at. Whether 0.5 is the RIGHT put score, or
whether the tier scores in the benchmark configs should straddle it
differently, is a separate question about the vector's default tiering and is
deliberately not changed here -- every existing published number depends on
it.

### Configuring a tier stack that a baseline can actually use

This matters for any comparison, not just for prefetching. `MaxBwDpe`
partitions targets into `target_score <= blob_score` (PREFERRED, sorted by
score DESCENDING) and the rest (FALLBACK, reached only when no preferred tier
has room), after first dropping any tier without space for the write. So:

**Every tier that should hold data by default must be scored at or below
0.5, and their relative order is their score order.** A stack of
`0.5 / 0.3 / 0.1` fills VRAM, then DRAM, then storage, purely by capacity --
which is what a "30% VRAM, 30% DRAM, 40% NVMe" configuration means. The
historic `1.0 / 0.2 / 0.0` does not: it fills DRAM, then storage, and reaches
VRAM only as a fallback that in practice never triggers.

That distinction is load-bearing for the evaluation. Comparing `none` against
`gs` under `1.0 / 0.2 / 0.0` would not measure prefetching at all -- the
baseline cannot put anything in VRAM, so the comparison silently becomes
"fast tier off" versus "fast tier on". `--hbm-score` / `--ram-score` /
`--nvme-score` exist so the baseline can be given a real three-tier
hierarchy, and the prefetcher's hot/warm/cold vocabulary must then be set to
those same three numbers.

**A hot score equal to the page's current score promotes nothing.**
`ReorganizeBlobInternal` compares the new score against the blob's current
one and returns early when the difference is below
`score_difference_threshold`. So with tiers at `0.5 / 0.3 / 0.1` and the
vector putting at 0.5, a `hot = 0.5` hint is a no-op on every page that has
not previously been demoted -- the page is already *eligible* for VRAM, and
whether it is actually *in* VRAM was settled by capacity at write time and
cannot be revisited by restating the same number.

Measured, promote-only at `hot = 0.5`: 8188 records sent, and a tier split
identical to the baseline to the decimal. The prefetcher ran, reported
healthy counters, and moved nothing.

The hot score must therefore be strictly ABOVE the vector's put score while
still being at or above the fast tier's score, so that the fast tier stays in
the preferred group and ranks first. With VRAM at 0.5 and the put score at
0.5, `hot = 0.6` satisfies both. This is the least obvious constraint in the
whole mechanism and the easiest to get silently wrong, because every counter
looks correct when it is.

A cold score should EQUAL the storage tier's score rather than merely be
below it. Below it, storage lands in the fallback group, whose ordering is by
`write_bandwidth_mbps_` -- a prediction with no notion of device type, which
`MaxBwDpe::SelectTargets` itself documents as having rated an HBM tier at
118 MB/s against host RAM at 1600.

### Sizing the lookahead: what "optimal" means on this workload

Gray-Scott's access order is a deterministic sweep -- block b reads planes
z0..z1 in order, four regions per z -- so the future is fully known and there
is no prediction problem to solve. The Belady-optimal tier placement is
simply "the fast tier holds the next |fast tier| accesses", and a prefetcher
that promotes exactly that far ahead and demotes everything behind it attains
it. There is no cleverer policy available for this pattern, only a
better-sized window, which is why `--lookahead 0` derives it instead of
guessing:

    hot set at any instant = blocks * 4 regions * (L + 2)   planes
    L = hbm_planes / (4 * blocks) - 2

Undersized, the fast tier sits part empty and misses are served from a slower
one. Oversized, promotions evict each other before they are used and the
migrations are wasted work. So it is a real optimum with a curve either side,
and the evaluation sweeps L around the derived value rather than asserting
it.

## 2. Where the hook fires

The yieldable driver's round loop is:

```
launch pending blocks -> Synchronize -> D2H the YieldBlockState array
                      -> recompute pending -> UploadPending -> service()
```

The D2H of the block state array is the moment. Every suspended block has
just published why it stopped, no kernel is resident, and the host is about
to decide what to relaunch. Any CTE traffic issued here overlaps with the
next round's kernel rather than competing with it.

That is what "trigger when coroutine return occurs" means concretely: the
hook is a callback on `Yieldable::Round()`, fired immediately after
`Memcpy(host_yield_, d_yield_)` and before the pending set is recomputed.

```cpp
// context-runtime/include/clio_runtime/gpu/yieldable.h
using YieldObserver =
    std::function<void(const YieldBlockState *states, u32 nblocks, u32 round)>;
void Yieldable<StateT>::SetYieldObserver(YieldObserver obs);
```

One seam, in the runtime, that knows nothing about the CTE or about vectors.
The vector binds itself into it.

## 3. The position problem, and the cursor

`YieldBlockState` carries `resume_point_`, and in macro mode that is the
`__LINE__` of the yield — which is a position in the SOURCE, not a position
in the DATA. In coroutine mode it is not even that: the resume point lives
inside the compiler-generated frame (`YieldLaneHeader::coro_resume_`, an
opaque frame address), and `resume_point_` stays 0 forever. Neither is
something a host prefetcher can turn into a page number.

The position has to be published by the workload, because only the workload
knows what its position IS. Gray-Scott's is `z`. LAMMPS's is a bin range.
LBANN's is an offset into W.

**`YieldBlockState` gains a two-word cursor**, stamped by the kernel and read
by the host:

```cpp
struct YieldBlockState {
  u32 resume_point_;
  u32 status_;
  u64 wait_tag_;
  /** Workload-defined position. Opaque to the driver, which only copies it. */
  u64 cursor_;
  u64 cursor_aux_;
};
```

Why here and not in the per-block user state (`Yieldable<StateT>::d_user_`):
this array is **already copied D2H every round**. The cursor rides for free —
16 more bytes per block on a copy that is 16 bytes per block today. Putting
it in `StateT` would mean a second D2H per round on the driver's critical
path, to carry the same two numbers, and would force every workload that
wants prefetching to also adopt a user-state type.

The device side is one call, usable at any nesting depth because
`YieldTls().block_state_` is the same object as `view.Y()`:

```cpp
// thread 0 writes; block-collective call
CTP_GPU_FUN void YieldPublishCursor(u64 pos, u64 aux = 0);
```

Gray-Scott calls it once per z-iteration, at the top of the loop, before the
first fetch. Cost: one store per z per block, off the critical path.

**The cursor is a hint and is treated as one.** A block that publishes
nothing leaves it 0; a prefetcher that sees a cursor it cannot interpret
emits no hints. Nothing about correctness depends on it — the demand fetch is
still what makes a page resident, and a wrong hint costs a wasted tier
migration, never a wrong answer. That property is what makes it safe to hand
this API to a workload author.

## 4. The vector-side API

```cpp
// clio_cte/gpu_vector/prefetch.h

/** One prefetch hint: "page `page_` should be scored `score_`." */
struct PrefetchHint { u64 page_; float score_; };

/** What a prefetcher writes into. Bounded — a prefetcher cannot flood. */
class PrefetchSink {
 public:
  void Hint(u64 page, float score);
  void HintRange(u64 pg_lo, u64 pg_hi, float score);
  u32 size() const;
  static constexpr u32 kMaxHintsPerRound = 4096;
};

/** The per-block yield state a prefetcher sees. */
struct YieldEvent {
  u32 block_;         // LOGICAL block id
  u32 round_;         // driver round, monotone within one kernel run
  u32 status_;        // kYieldSuspended | kYieldDone
  u32 resume_point_;  // macro mode only; 0 under coroutines
  u64 wait_tag_;      // the completion word it parked on, 0 = none
  u64 cursor_;        // what the kernel published
  u64 cursor_aux_;
};

class Prefetcher {
 public:
  virtual ~Prefetcher() = default;
  /** Called once per block per round, after the kernel returns. */
  virtual void OnYield(const YieldEvent &ev, PrefetchSink &out) = 0;
  /** Called once per round after every block's OnYield. Default: nothing.
   *  For prefetchers whose decision needs the whole grid's cursors. */
  virtual void OnRoundEnd(u32 round, PrefetchSink &out) {}
  virtual const char *Name() const = 0;
};
```

Registration is on the vector, which is what owns the CTE client and the tag:

```cpp
vec.RegisterPrefetcher(std::make_shared<GrayScottPrefetcher>(geom, policy));
vec.SetPrefetchPolicy({/*hot=*/1.0f, /*warm=*/0.2f, /*cold=*/0.0f});
runner.BindPrefetch(vec);     // installs vec's observer on the driver
```

More than one prefetcher may be registered; their hints merge into one sink,
last writer wins for a repeated page. That is deliberate — it lets a generic
stride prefetcher run underneath a workload-specific one as a floor.

### The rescore engine

Between the sink and the CTE sits the part that makes this affordable.

1. **Dedup against last-sent.** A hash map page → last score sent. A hint
   whose score differs from the last one sent by less than the CTE's own
   `score_difference_threshold` is dropped host-side. Without this, a block
   that yields eight times inside one z-iteration re-sends the same eight
   promotions eight times, and the CTE does the no-op comparison eight times
   after paying for eight task submissions. Gray-Scott at 16 blocks with a
   lookahead of 2 emits ~64 hints per round and, after dedup, sends on the
   order of one batch every few rounds.
2. **Batch.** Fill `PodMultiScoreTask` records (name = `PageName(pg)`, the
   same decimal string the fault path normalizes to; `score_` is the only
   other field `PodMultiScore` reads) to 64 and submit.
3. **Never block.** Submit async, keep the futures, reap them at the TOP of
   the next round. The driver's round gap is not a place to wait on the CTE:
   a rescore that has not landed by the next round is still useful, and one
   that failed is counted, not retried. Blocking here would put tier
   migration on the critical path, which is exactly what prefetching exists
   to take it off.
4. **Count everything.** `PrefetchStats { hints, deduped, sent, batches,
   errors, promotions, demotions }`, readable like `Vector::ReadStats`. The
   tiered evaluation is a comparison of these against fault counts and tier
   occupancy; without them a prefetcher that does nothing and a prefetcher
   that works look identical in the timings, which move ~26% run to run on
   this workload.

## 5. The Gray-Scott prefetcher

### Geometry

One page is exactly one XY plane, and each region base is a multiple of
`plane`, so page arithmetic is exact and trivial:

```
page(base, z) = base / plane + z
```

The four regions are `ubase`, `vbase`, `unext`, `vnext`, each `plane * nz`
elements. Block b owns `z in [z0, z1)`. The host knows all of this; it passes
it in at construction, and re-arms the region bases each step because the
step loop swaps them.

### What the kernel actually touches at z

Reading `StepCoro`: eight planes per z-iteration —
`u{z-1,z,z+1}`, `v{z-1,z,z+1}`, `unext{z}`, `vnext{z}` — with `z-1` and `z`
of u and v re-read at `z+1`. The sliding window is 8 planes, 6 of which are
inputs and 2 of which are freshly written outputs.

### The policy

At cursor `z` on block `b`, with lookahead `L`:

- **HOT** — `u`, `v` at `z+1 .. z+L`, and `unext`, `vnext` at `z .. z+L`.
  These are the planes the demand fetch will ask for within the next `L`
  z-iterations. `L` is the tunable, and the useful range is small: one
  z-iteration is one plane of arithmetic, so a lookahead of 8 planes at
  1 MB pages is 8 MB per block in flight, 128 MB across 16 blocks, and a
  tier migration has to complete inside that window to be worth anything.
- **COLD** — `u`, `v` at `z-2`. Plane `z-1` is still in the window (it is
  `zm` for the current z); `z-2` left it for good and will not be read again
  until the NEXT step's pass reaches it, which is `zper` z-iterations away.
  That distance is the whole justification for demoting it.
- **Outputs are never demoted below `warm`.** `unext`/`vnext` are read again
  next step — `StepCoro` fetches the output plane before write-holding it —
  and a page demoted to NVMe at the end of step s is a page faulted from NVMe
  at the start of step s+1. This is the one place where the obvious policy is
  wrong, and it is wrong in the direction that looks like a win in the fault
  count and a loss in the wall clock.
- **Slab edges are not special-cased.** A block near `z1` emits hints past
  its own slab; those pages belong to its neighbour, which is reading them as
  halo anyway, and dedup collapses the duplicate. In a distributed run they
  are another NODE's pages and the rescore simply targets a blob this node
  does not own — the CTE routes it by hash, which is correct, and the hint is
  as useful there as here.

### Bases change every step

`StepCoro` is relaunched per step and the host swaps `cu/nu`, `cv/nv`. The
prefetcher gets `SetRegions(cu, cv, nu, nv)` before each `runner.Run`, so it
is always hinting about the regions the CURRENT step reads. Getting this
wrong is silent: it would promote the wrong four regions and look exactly
like a prefetcher that does not help.

## 6. The generic control: StridePrefetcher

A workload-specific prefetcher that beats no-prefetch proves that prefetching
helps. It does not prove that WORKLOAD-SPECIFIC prefetching helps — for that,
the comparison has to be against a prefetcher that knows nothing.

`StridePrefetcher` keeps, per block, the last two cursors, takes the
difference as a stride, and promotes `cursor + k*stride` for `k in [1, L]` in
a single caller-named region. It is right about Gray-Scott's `u` and `v`
current planes, and blind to the other two regions and to the demotion
opportunity entirely. That gap is the measurement.

## 7. The tiered evaluation

Three tier stacks × four prefetch configurations, on the pressure geometry
that already exists (10240 MB grid, 1 MB pages, 16 blocks, cache driven to
the 8-plane window floor — see `gv_grayscott_pressure.yaml`, whose finding
was that Gray-Scott's cache axis is inert above the window; the TIER axis is
the one with headroom left).

| tier stack | HBM | RAM | NVMe |
|---|---|---|---|
| `hbm` | 1024 MB | — | — |
| `hbm+ram` | 1024 MB | 11264 MB | — |
| `hbm+ram+nvme` | 1024 MB | 4096 MB | 16384 MB |

| prefetch | what it is |
|---|---|
| `none` | today's behaviour, the baseline |
| `stride` | generic, lookahead 4 |
| `gs` | workload-specific, lookahead 4 |
| `gs-L` | workload-specific, lookahead swept 1/2/4/8/16 |

The `hbm` row is a control: with one tier there is nowhere to promote from,
so every prefetch configuration must come out identical to `none` within
noise. If it does not, the prefetcher is costing something it should not.

**Metrics.** Timings on this workload move ~26% run to run, so the primary
numbers are counts:

- `faults` and faults/z — unchanged by prefetching, and that is the point.
  A prefetcher that changes the fault count is doing something other than
  what it claims.
- **fault SERVICE time** — the number prefetching is supposed to move. Total
  driver round time (`Yieldable::KernelMs`) minus the compute, or more
  directly: rounds per z-iteration, since a block that parks on a slow fetch
  parks for more rounds.
- tier occupancy at the end of the run (the TIER SPLIT line already printed).
- `PrefetchStats`: sent, deduped, errors. A run reporting `sent=0` is not a
  prefetching run whatever its timings say.

**Measured, at 1024 MB / 256 KB pages / 8 blocks / 2 steps, three tiers:**

| | `none` | `gs`, L=4 |
|---|---|---|
| v_checksum | 671068.413736 | 671068.413736 |
| faults | 9264 | 9245 |
| kHBM used | 0 MiB | 52 MiB |
| host tier used | 1024 MiB | 489 MiB |
| pf_hints / sent / deduped | 0 / 0 / 0 | 241172 / 16416 / 224756 |
| pf_promote / pf_demote | 0 / 0 | 8352 / 8064 |
| pf_errors / pf_notfound | 0 / 0 | 0 / 2032 |

The checksum is identical, which is the invariant. Faults are unchanged,
which is the point -- the same pages are demanded either way. The dedup rate
is 93%, which is what makes this affordable: a block parks several times
inside one z-iteration and re-derives the same hints each time. And
`pf_notfound=2032` is expected rather than a fault: the lookahead runs past
the written frontier, and on step 0 the two output regions have no blobs at
all because the seed writes only u and v.

Timings are NOT quoted here. This workload moves ~26% run to run and these
are single runs; the counts above are the evidence, and the wall clock is
what the tiered sweep's repeats are for.

## 8. RESULT: at 30/30/40, the optimal tier-prefetch policy is the null policy

Measured. 2048 MB grid, 256 KB pages (256x256x2048 planes over four fields),
8 blocks, 16 slots/block, 4 steps, best of 2. Tiers scored 0.5 / 0.3 / 0.1
with capacities 614 / 614 / 922 MB, which places the data 30% VRAM, 30% DRAM,
40% NVMe. `hot = 0.6` (above the put score, so promotions actually migrate).

| | ms | faults | pf_sent | promo | demo | vram | dram | nvme |
|---|---|---|---|---|---|---|---|---|
| none            | **4243** | 34885 | 0     | 0     | 0     | 27.6% | 27.4% | 45.0% |
| gs L=8 promote  | 4356 | 34916 | 8188  | 8188  | 0     | 27.4% | 27.5% | 45.0% |
| gs L=8 full     | 5270 | 35276 | 65268 | 32762 | 32506 | 25.0% | 30.0% | 45.0% |
| gs L=74 promote | 4699 | 36714 | 8188  | 8188  | 0     | 27.6% | 27.3% | 45.0% |
| gs L=74 full    | 5587 | 37740 | 67116 | 34610 | 32506 | 26.0% | 29.9% | 44.0% |

Checksum identical (`1511716.301085`) in all five, so the mechanism is sound.
Every prefetch configuration is SLOWER than no prefetching, monotonically in
both the lookahead and the amount of demotion.

### Where the time actually goes (channels 8 and 9)

`clio_evlat_add` channels `reorg_stall` (a read waiting on a writer) and
`reorg_move` (one migration, end to end) were added to answer this directly,
under `CLIO_EVLAT=1`:

| | ms | get_total avg | reorg_stall | reorg_move |
|---|---|---|---|---|
| none            | 4210 | 66 us | -            | - |
| gs L=8 promote  | 4334 | 61 us | n=1          | n=4108  avg 202 us |
| gs L=8 full     | 4604 | **20 us** | n=11 avg 3.7 ms | n=61196 avg 271 us |

Two things follow, and the first corrects an earlier claim in this document.

**The demand path is NOT stalled by reorganization.** 11 waits out of 38,966
gets (0.03%), about 40 ms of a 4604 ms run. The write-token discipline is
real -- `GetBlobImpl`'s torn-layout guard does block a reader while a
migration holds the token -- but in practice a prefetcher's migrations and
the kernel's faults almost never land on the same blob at the same time.
Reorganization behaves as background work, which is what it is supposed to do.

**The prefetch demonstrably WORKS.** Mean fault service time falls from 66 us
to 20 us, a 3.3x improvement, because pages are in a faster tier when
demanded. That is the mechanism doing exactly what it was built to do.

It still loses, and the reason is throughput, not latency and not stalls:
61,196 migrations x 271 us is **16.6 seconds of migration work inside a 4.6
second run** -- roughly 3.6 workers busy moving bytes -- and each migration
reads and writes a whole page, so it adds ~31 GB of bdev traffic to a run
whose own traffic is ~17 GB. The saving is 46 us per fault on 39k faults
(1.8 s); the cost is far larger.

**A tier promotion is a copy, and on this workload it is an EXTRA copy.**

Gray-Scott's sweep touches every page exactly once per step per tier. The
three reads of a plane within a step (as z-1, z, z+1) are served by the GPU
page cache -- that is what the sliding window IS -- so from the tier's point
of view each page is read once. A demand fetch of a page on NVMe reads it
from NVMe. A promotion of that page reads it from NVMe and writes it to VRAM,
and the demand fetch then reads it from VRAM. Same NVMe read either way, plus
a VRAM write, plus an eventual eviction. Two transfers where there was one.

The ceiling is therefore zero, not merely small. With 30% of capacity in
VRAM and every page needed every step, at most 30% of accesses can avoid the
slow tier -- and the baseline ALREADY achieves exactly that, for free, because
the DPE fills tiers in write order and write order here IS access order (both
are z ascending). An optimal prefetcher can change WHICH pages are in VRAM
but not HOW MANY, so it cannot reduce the bytes read from NVMe. It can only
add to them.

Promote-only makes the same point from the other side. It sends 8188 records
-- 4 regions x 2048 planes, i.e. every page promoted exactly once -- and then
dedup correctly suppresses the rest forever. **Promoting every page to hot is
identical to promoting none**: the scores are uniform again and capacity
decides, which is what the near-unchanged tier split shows. A tier hint only
carries information if it is differential, which is what makes demotion
structurally necessary and, here, structurally unaffordable.

### The stall gate: reorganize only once per X stalls

If the problem is migration volume, cap it against the signal that says a
migration could pay. A stall -- a block suspended on a completion word that
has not been written -- means the last access was served by a tier the block
had to WAIT for, and those are the only accesses a promotion can repay. At a
30/30/40 split, 60% of the ungated promotions are no-ops in value terms (the
page was already in VRAM or DRAM) and full price in cost; gating on stalls is
SELECTION, not merely rate limiting.

`PrefetchPolicy::stall_every_` / `--pf-stall-every X`. The gate must DROP,
not defer: with an incremental emitter the planes promoted per step are fixed
by how far the cursor travels, so batching emissions changes only when they
are sent. A closed gate therefore lets the window slide past unhinted.

Same configuration as above, single runs, L=8:

| X | ms | pf_sent | migrations | fault avg | vram |
|---|---|---|---|---|---|
| none (no prefetch) | 4253 | 0 | 0 | 67 us | 30.0% |
| 0 (ungated) | 4577 | 65268 | 61204 | **19 us** | 25.0% |
| 4   | 4563 | 65012 | 61062 | 19 us | 25.0% |
| 16  | 4276 | 17408 | 16142 | 58 us | 30.0% |
| 64  | 4259 |  5340 |  4994 | 60 us | 30.0% |
| 256 | 4298 |  1432 |  1346 | 65 us | 30.0% |

**The gate works exactly as designed, and it is not enough.** Migrations fall
by 45x from X=0 to X=256, wall clock converges to the baseline, and at X >= 16
`reorg_stall` returns to zero and the tier split returns to the untouched
30/30/40. But it never goes BELOW the baseline: the ms column at X >= 16 is
flat within this workload's noise.

The reason is that **cost and benefit fall at the same rate**, so the gate
slides along a line that never rises above the baseline:

| | migrations | fault-latency saved | per migration | cost per migration |
|---|---|---|---|---|
| X=0  | 61204 | 48 us x 38977 = 1.87 s | 30 us | 214 us |
| X=64 |  4994 |  7 us x 38977 = 0.27 s | 55 us | 148 us |

Selection IS working -- the benefit per surviving migration nearly doubles,
30 us to 55 us, which is precisely the stall signal keeping the promotions
that were worth making. It is simply not enough: a migration still costs
~150 us of worker time to save ~55 us of fault latency, so the policy needs
roughly another 3x in per-migration value to break even, and there is not
another 3x available from better selection alone.

What the gate DOES buy is safety. At X >= 16 the prefetcher is free: it costs
nothing measurable, stops perturbing the tier split, and stops colliding with
the demand path. So a conservative X is the right default for leaving a
prefetcher registered on a workload where it might help -- it cannot make
things worse, which the ungated policy demonstrably can.

### The conditions under which this WOULD pay

Nothing above is an argument against tier prefetching in general. It is an
argument that this workload, at this ratio, is the wrong place for it. It
pays when:

- **There is tier-level reuse.** If the GPU page cache cannot hold the reuse
  window, a page is faulted from the tier several times per epoch, and one
  promotion is amortized over N reads instead of paying for one. Gray-Scott
  at slots >= 8 has no such reuse by construction -- that is exactly what
  `gv_grayscott_pressure` established about the cache axis.
- **The run is latency-bound rather than bandwidth-bound.** The promotion
  costs the same bytes but off the critical path. This run is bandwidth-bound
  (faults are unchanged while time tracks migration volume), so moving the
  cost off the critical path does not help: there is no idle bandwidth for it
  to use.
- **The access order is not the write order.** The baseline's placement is
  optimal here only because the DPE happened to fill tiers in the order the
  sweep reads them. A workload that writes in one order and reads in another
  -- a transpose, a permuted epoch, a graph traversal -- gets no such gift,
  and that is where a prefetcher has something to correct.

`faults` rising slightly with the lookahead (34885 -> 37740) is a second-order
effect worth naming: prefetching does not change which pages are demanded,
but it does add host time to the round gaps, which changes how blocks
interleave and therefore how well they share the one page cache.

### The migration algorithm, and one optimisation that is not one

`ReorganizeBlobInternal` moves a blob by:

1. taking the blob's exclusive write token (the same one PutBlob, DelBlob and
   Truncate take), then draining in-flight readers;
2. allocating a host shared-memory buffer of the whole blob;
3. `ReadData` of the entire blob into that buffer;
4. `ClearBlob` -- frees the old blocks, KEEPS the metadata entry so a
   concurrent get never sees "not found";
5. `ExtendBlob` (DPE re-place at the new score) + `ModifyExistingData` to
   write the blob out to its new tier, with up to three attempts, the last
   falling back to the ORIGINAL score into the capacity just freed;
6. republish the SHM mirror and log the transaction.

So a migration is a full read plus a full write, staged through host memory:
~271 us for a 256 KB page, which is close to bandwidth-limited for 512 KB of
traffic. There is one shortcut: if the new score is below the best PERSISTENT
replica's score, the primary is dropped rather than moved, because a copy
that good already exists on storage.

**`PodMultiScore` awaits its records serially, and that is deliberate.** The
obvious optimisation is to dispatch them the way `PodMultiGetBlob` dispatches
large batches -- 64 heavy records pinning one fiber for ~17 ms looks like an
obvious convoy. It was tried and it is a regression:

| | ms | reorg_move avg | reorg_stall |
|---|---|---|---|
| serial (current) | 4604 | 215 us | n=0 |
| dispatch-then-await | 5350 | 1956 us | n=12 |

Bulk tier migration is bandwidth-bound, not latency-bound. Running 64 at once
does not finish them sooner -- it queues them against each other on the same
bdevs, multiplies each one's latency ~9x, and starts colliding with demand
reads on the write token (`reorg_stall` appears only in the dispatched run).
Serializing is what keeps reorganization behaving like background work: one
migration's worth of bandwidth at a time, leaving the rest to the data path.
The code carries this table so the experiment is not repeated.

### Two implementation defects this exposed

Both were found by disbelieving healthy-looking counters, and both are fixed:

1. **The dedup map was cleared every step.** It models the score the CTE
   holds for a page, which a change of page MEANING does not invalidate.
   Clearing it re-sent every page its own current score at each step boundary
   -- rescores the CTE no-ops -- and computed the promote/demote tally against
   an assumed prior rather than the known one, reporting `promote=896` against
   `demote=32512` on a run that was sending 66048 records.
2. **The window was re-emitted every round, not every advance.** A block
   parks about five times per z-iteration with an unchanged cursor, so the
   whole lookahead was regenerated each time and thrown away by dedup --
   12,265,244 hints at L=74. Emitting only the planes that ENTER the window
   cut that to 40,928, a 300x reduction, with identical resulting placement.

## 9. MEASURED: cluster-batched GNN, and the cost ratio that decides everything

Gray-Scott is the wrong shape for this (steady arrival, access order == write
order, once-per-step reads). The right shape is a BURST: a phase whose working
set arrives at once and is knowable in advance. Cluster-GCN is that shape, so
`clio_gnn_burst_bench` was built to measure it -- ogbn-products (2.4M nodes,
934 MiB features), METIS into 2048 partitions, feature rows renumbered
partition-contiguous, bursts run in a greedy overlap order, tiers 15/15/70.

Two prerequisites had to be established first, and both are measurable from
the graph alone (`gnn_page_skew.py`, `gnn_burst.py`):

- **Random minibatch sampling has NO page-level signal.** 1024 seeds x 13.7
  degree touch 7.2% of nodes scattered uniformly, so at 32 rows/page 90% of
  pages are touched per batch and at 256+ rows/page it is 100%. Top-decile
  access share is 0.10 -- exactly uniform. Nothing to select, so neither
  organization nor pinning can help. This is the failure
  `gnn_pagerank_cache.py` already documented.
- **You need cluster batching AND a matching layout.** Either alone leaves a
  batch touching ~100% of pages; together they collapse it to 41-57%.

### The measurements

| pages | arm | ms | fault avg | migrations | vram |
|---|---|---|---|---|---|
| 100 KiB | none | **1963** | 89 us | 0 | 14.9% |
| 100 KiB | organize, free->NVMe | 2219 | 73 us | 26627 | 14.0% |
| 100 KiB | organize, free->DRAM | 2154 | **68 us** | 26627 | 11.5% |
| 800 KiB | none | **957** | 221 us | 0 | 14.9% |
| 800 KiB | organize, free+fetch | 1134 | 257 us | 7794 | 12.2% |
| 800 KiB | promote-only, lead 2 | 975 | 246 us | 890 | 15.0% |

Checksums identical within each page size. Organization measurably improves
fault latency (89 -> 68 us, -24%) and measurably loses on wall clock.

### Why: a migration costs ~4x a fault, and needs ~8x to pay

The ratio is stable across page sizes, so it is proportional cost, not fixed
overhead -- coarser pages do not rescue it:

| page | migration | fault | ratio |
|---|---|---|---|
| 100 KiB | 376 us | 89 us | 4.2x |
| 800 KiB | 1031 us | 219 us | 4.7x |

A migration is read-whole-blob-to-host + free + re-place + write-whole-blob
under the write token; a fault is one transfer. So a migrated page must be
reused >= 4 times before displacement merely to break even. Cluster batching
at churn 42-46% gives a page a residency of **2.2-2.4 bursts**. Short by ~2x.

Gray-Scott, for contrast, gives 1.0 -- so GNN moves the ratio from 4:1 against
to 2:1 against. Real progress, still a loss.

### The actual blocker: promotion cannot evict

The deeper finding, and the one that makes the 4x into an 8x:

    none:          vram=14.9%   pf_sent=0
    promote-only:  vram=15.0%   pf_sent=890   (7806 deduped)

**VRAM never moves.** The DPE fills tiers in write order at ingest, so the
fast tier is FULL, and `MaxBwDpe::SelectTargets` filters targets by
`remaining_space_ >= data_size` with no eviction path. A promotion into a full
tier therefore does not displace anything -- it silently lands in DRAM or
storage. Promote-only is nearly free (890 migrations, +1.9%, within noise) and
buys nothing, because nothing can land.

So a useful promotion requires a preceding free, and a free is a full
migration in its own right: 2 migrations ~ 8x a fault, against 2.2x reuse.
That is the whole result.

An ordering bug of ours sat on top of this and had to be fixed first: frees
and fetches submitted in one call were merged into a single page-ordered batch
and raced, so the organizer pushed data DOWN (dram 14.9% -> 5.4%) and got
nothing UP. Two ordered submissions fixed the direction but not the economics.

### What would make it pay

**A cheap eviction.** The CTE already has one, and the gpu_vector simply does
not qualify for it: `ReorganizeBlobInternal`'s issue-#886 path drops a
blob's primary WITHOUT MOVING ANY BYTES when a persistent replica already
holds those bytes at least as well -- it frees the blocks and keeps the
replica. If the vector kept a persistent replica on storage, the fast-tier
copy would be a true CACHE: freeing it would cost a metadata update instead of
a read+write, the 8x would collapse toward 1x, and 2.2x reuse would be
comfortably profitable.

That is a concrete design change with a concrete predicted effect, and it is
the thing to try next. The secondary levers are lower churn (a better
partitioner, or bursts ordered by overlap fraction -- already worth ~2x) and a
migration path that does not stage through a host buffer.

Two results stand on their own regardless:

- **Greedy burst ordering is free and halves churn** (93% -> 53% random vs
  greedy on arxiv; 58.6% -> 45.8% once ordered over the full partition set
  rather than a sampled subset). It needs no migration machinery at all.
- **Layout is a precondition, not an optimization.** Without
  partition-contiguous renumbering every batch touches ~100% of pages and no
  policy of any kind can help.

## 10. What this does not do, and why

- **No device-side rescore.** The device COULD submit a `PodMultiScoreTask`
  the way it submits `PodMultiGetBlobTask`. It should not: a rescore issued
  from inside the kernel competes with the fault path for the same queue,
  and the decision it encodes is about tiers, which is host knowledge. The
  yield gap is free host time and there is no reason not to use it.
- **No automatic prefetching.** Registration is explicit and per-vector.
  A prefetcher that fires by default would change the tier placement of every
  existing benchmark, including the ones whose published numbers assume the
  frecency organizer is the only thing moving blobs.
- **No feedback loop.** The prefetcher does not learn from whether its
  promotions were used. That is the obvious next thing — hint accuracy is
  measurable (did a promoted page get faulted before it was demoted?) — but
  it needs the hit/miss attribution to be built first, and a design that
  measures nothing should not also adapt.
