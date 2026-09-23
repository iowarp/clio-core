# The paging defect: diagnosis and fix plan

Written 2026-09-23 from the E4/E5 runs on Aurora. Nine occurrences across
kmeans, weights and lbann, all at 4 nodes, all with a file tier in the
composition and the frame cache under pressure.

## What is observed

Three symptoms, which turn out to be one mechanism seen at different points:

| symptom | where | example |
|---|---|---|
| `DEVICE FATAL 7 (fetch returned an error; its pages were left EMPTY -- a generational get names a generation the writer has not published)` | `PublishFetch` | kmeans, 256 KB page, Lustre-heavy |
| `DEVICE FATAL 2 (HoldPage: page not resident)` | `CoHoldPage` | weights at 2 and 4 frames, lbann at 64 and 512 |
| silently wrong data, checksum off by 2%-65% | nowhere -- the run passes its gates and lies | weights on Lustre-heavy and on balanced at 8 frames |

## What is NOT a defect

**The GPU "segfault at 0x0" is the trap working as designed.** I called this a
second defect earlier and that was wrong. `sycl_compat::Trap()` has no SPIR-V
trap instruction available, so outside NVPTX it deliberately writes to null
to fault the kernel (`sycl_cuda_compat.h`, `Trap()`). The reason is latched
into the host-readable mirror before the trap, which is why the host still
prints the DEVICE FATAL line. Nothing here needs fixing.

What IS lost is the *device printf* that accompanies each trap -- the rich
one that prints the set census ("frames holding it=N fetching=N free=N").
The driver abort discards the printf buffer, so the only evidence that
survives is the mirror's three arguments. That is a diagnostics gap, not a
crash.

## The mechanism

`PublishFetch` (device_vector.h, the `get_failed` branch) handles a fetch
whose get was refused:

```cpp
const bool get_failed = bt->fetch->GetReturnCode() != 0;
if (get_failed) {
  for (each fetched slot) { valid_lo = 0; valid_hi = 0; atomicSub(&p->fetching, 1u); }
  FatalNote(kFatalGetFailed, ...);
  bt->fetch_n = 0;
  bt->fetch_busy = 0u;
  return;                       // <-- NO TRAP
}
```

The comment above it states the intent, and the intent is right: leave the
frames empty so a later `Covers` check traps loudly rather than publishing
garbage. It also names the cause -- "a get naming a generation the writer has
not published yet is refused, which is exactly right".

Two things follow from that code, and they are the defect:

1. **The kernel keeps running after a failed fetch.** `FatalNote` latches a
   code into the mirror and `PublishFetch` returns. There is no `__trap()`.
   So the block proceeds to `CoHoldPage`, and what happens next depends on
   luck:
   - the frame still carries `page_num == pn`, so `Find` succeeds; on a
     **write** hold `Covers` is not checked at all, and the kernel writes
     into a frame whose contents were never fetched -> **silent wrong data**;
   - on a **read** hold `Covers` fails -> `kFatalNotCovered` (code 3);
   - if the frame was meanwhile taken for another page, `Find` fails ->
     `kFatalNotResident` (code 2).

   That single branch explains all three symptoms, including why weights
   (which holds for write while seeding and for read while gathering) shows
   both a wrong checksum and a code-2 trap depending on the cache size.

2. **`FatalNote` latches only the first code**, by `atomicCAS`. So when a
   refused get (7) precedes the trap it caused (2 or 3), the mirror reports 7
   and the trap site is invisible, and vice versa. The codes we collected are
   therefore not independent failures; they are the same event reported from
   whichever site won the race.

**It is not Lustre-specific.** The clean E5 pass, on a DRAM+DAOS composition
with no Lustre at all, reproduces it: weights checksums OK at 16, 32 and 128
frames of cache and MISMATCHes at 64. Non-monotone in cache size, which is
what a race looks like and not what a capacity threshold looks like. Any
file tier will do; the filesystem behind it is irrelevant.

**Why a file tier and cache pressure are both needed.** A generational get is
only issued when a page must be re-read; with a cache that holds the working
set, pages are never re-fetched and the path is unreachable. kmeans at one
page per block never re-fetches and is the one workload that survived every
cache size.

## Fix plan

Ordered so that each step makes the next one cheaper. Steps 1 and 2 are
small and stop the corruption; step 3 is the real fix.

### 1. Stop the silent corruption (small, do first)

In the `get_failed` branch, make the frames unusable rather than merely
empty, and stop the kernel there:

- clear `page_num` to `kNoPage` as well as the valid range, so a subsequent
  `Find` cannot resolve the frame and a **write** hold cannot slip past the
  `Covers` check that only guards reads;
- `__trap()` after `FatalNote`, so the failure is reported at the fetch that
  failed rather than at whatever unrelated site notices later.

This converts "wrong answer" into "loud failure", which is the property the
surrounding comment already claims and does not yet have.

### 2. Make the evidence survive (small)

- Pass the set census into the mirror. `FatalNote` already takes `a4, a5, a6`
  and `ReportSetFull` already computes exactly this for the set-full case;
  the not-resident path passes only three arguments. Add frames-holding,
  fetching and free counts so a trap is diagnosable without the lost printf.
- Record the refused generation and the blob name in the `kFatalGetFailed`
  arguments (the generation is already in `bt->fetch_gen_sub`).

### 3. Find out why the get actually fails (CORRECTED -- see below)

**My first version of this section was wrong and is retracted.** I took the
explanation from the comment above the branch ("it is the generational path
that makes this reachable in practice") and applied it to every symptom
without checking. Two things falsify that for weights:

- **weights fetches at generation 0.** `CoFetch(0, ...)` in both its seed and
  gather loops, and the runtime's generational wait is guarded by
  `generation_ != 0`, so weights never enters that path at all. Only lbann
  passes real generations.
- **The runtime already waits.** `GetBlob` parks the reader until the named
  generation arrives, bounded at 10 seconds, and only then returns 1. So a
  refusal is not a lost race of microseconds that a retry would paper over;
  it is a writer that did not publish in ten seconds, or a different failure
  entirely.

So the mechanism in "The mechanism" above stands -- a failed get let the
kernel run on, and that is what turned one fault into three symptoms -- but
the CAUSE of the failing get is still open, and is probably not the same for
weights (generation 0, some other GetBlob error) as for lbann (real
generations, possibly the 10 s timeout).

Step 1 is also the diagnostic for this: the trap it adds prints the page
number and the generation of the fetch that was refused, which is exactly
the information the old silent path threw away. The next run of the failing
cells will name them.

### 3b. Then, if it is a late generation (the original idea, still plausible for lbann)

A get naming a generation the writer has not published yet is a **transient**
condition, not an error: the writer is a peer block that will publish. The
current contract turns a race into a hard failure. Options, cheapest first:

- **Wait and retry in the fetch path.** On a refusal whose cause is an
  unpublished generation, re-issue the get after a yield, bounded by the same
  stall budget `SubmitFetch` already uses for allocation pressure
  (`kMaxFetchStalls`). This matches how the allocator already treats
  transient pressure and needs no protocol change.
- **Distinguish the two refusals.** The runtime currently returns one
  non-zero code for "generation not published" and for a genuine error. The
  retry above is only correct for the former, so the CTE get needs to say
  which it was.

### 4. Reproducer and regression test

A single-node case is enough and costs nothing to run: one file bdev sized
below the working set, a gather kernel over more pages than the cache holds,
and a generational fetch. Assert that either the data is right or the run
traps -- never that it finishes with a wrong checksum. This belongs beside
the existing gpu_vector tests rather than in the benchmark suite.

## What this blocks

E5 sweeps the cache deliberately, so it lands in this path by construction:
five of its twenty cells died and the timings that survived cannot be
trusted on any composition that has a file tier. E4's weights row has one
failed cell for the same reason. Everything with a DRAM-only or DRAM+DAOS
composition and a cache that holds the working set is unaffected.
