# eternia-MD: a from-scratch LAMMPS reimplementation on gpu_vector

We stop extending LAMMPS. This document designs a standalone MD code that
implements the same physics as the LAMMPS melt benchmark (lj/cut, NVE,
periodic box) entirely on eternia `gv::Vector` APIs, living in
`context-transfer-engine/adapter/gpu_vector/benchmark/` as
`clio_lammps_md_paged_bench`, with correctness validated against stock LAMMPS
output. Cache sizes are FULLY CONFIGURABLE throughout; the design assumes
nothing about them except per-structure lower bounds that make the working
set residentable.

## 1. The principle that changes everything

The hooked-pair-style experiment proved one thing conclusively: the cost was
never the paging — it was computing on state the device does not own. Every
step re-shipped positions in and forces out; every rebuild re-shipped the
list, because LAMMPS builds state on the CPU and host arrays are its plugin
ABI.

So the design principle here is the same one KOKKOS uses, taken further:
**device-canonical state on paged vectors, for the entire run.** The host's
only jobs are configuration, initial-condition load, reading back a handful
of thermo scalars, and validation dumps. In steady state, ZERO bytes of
simulation state cross PCIe per step. Unlike KOKKOS, every large structure
sits on a paged vector, so the per-rank capacity ceiling is the tier stack
(HBM → RAM → NVMe), not VRAM.

## 2. Data structures and their page layout

The single most important layout decision: **atoms are stored bin-major with
padded bins, and pages contain whole bins.** The box is divided into cells
("bins") of edge ≥ cutoff + skin. Each bin gets a fixed atom capacity
(configurable, with a counted overflow check that refuses loudly rather than
corrupting); an atom's index is (bin, slot-within-bin). Empty slots carry a
sentinel. Every reneighbor re-sorts atoms into this order.

Why this layout is the keystone:

- **It kills the ghost problem at the root.** There are no ghosts at all:
  a single periodic box uses minimum-image arithmetic in the pair kernel
  (a few cheap conditional subtracts per pair — the kernel is memory-bound,
  they are free). No ghost tail, no two-range holds, no index-stability
  games. j-atoms of any bin live in that bin's 27 stencil bins, period.
- **It makes every working set STATIC and knowable.** The pages a force
  block needs are exactly the pages containing its stencil bins — computable
  from geometry at rebuild time, not discovered by scanning a list.
- **It IS the spatial sort.** LAMMPS sorts every 1000 steps as an
  optimization; here bin order is the invariant, refreshed every rebuild.

The structures:

| structure | element | layout | pages | mutated |
|---|---|---|---|---|
| `x` | float4: x, y, z, type-in-`.w` | bin-major, padded | whole bins per page; page size configurable | integrator, every step; re-sorted at rebuilds |
| `v` | float4: vx, vy, vz, spare | identical indexing to `x` | same geometry as `x` | integrator, every step |
| `f` | float4: fx, fy, fz, spare | identical indexing | **page = one block's work unit** (a contiguous run of bins); the decomposition never splits a page — one exclusive writer per page | force kernel writes, integrator reads, every step |
| `neigh` | int (packed j-index) | CSR: atom i's neighbors are one contiguous run | the LARGEST pages (streamed once per step, zero reuse; cost is per request) | rebuilt every reneighbor, ON DEVICE |
| resident index class (plain device arrays, O(atoms) or O(bins) with small constants — deliberately NOT paged; they are the machinery that drives the paging) | | | | |
| — bin counts | int per bin | | | every rebuild |
| — CSR offsets | int per atom+1 | device prefix sum | | every rebuild |
| — stencil table | 27 bin offsets + periodic wrap flags | | | once |
| — per-slice page sets | the page IDs each force block must hold | from geometry | | every rebuild |
| — box + LJ tables | scalars, ntypes² | kernel arguments | | once per run |
| accumulators | PE, KE, 6-term virial, pair count | device scalars, block-reduced | | every step |

Note `v` is now a first-class paged structure — the integrator is ours.

## 3. The pipeline — every kernel, every access pattern

All kernels are block-collective device coroutines using the standard verbs:
`Held<T> h = co_await HoldPage(...)` (the hold is the pin), `FlushAsync` /
`AwaitFlush` for written pages. All are self-contained; the host loop is
`for step: integrate1(); maybe_reneighbor(); force(); integrate2(); maybe_thermo();`.

**K1 — integrate, first half (kick + drift).** Each block owns a contiguous
run of bins (= its f page = its x/v page span). Holds: its x pages (write),
v pages (write), f pages (read). Pattern: pure sequential stream, perfect
page locality, skip sentinel slots. Applies `v += dt/2·f/m; x += dt·v` with
periodic wrap of positions.

**K2 — reneighbor (every N steps, or displacement-triggered):**
- *K2a bin-count + permutation:* stream x, compute each atom's new bin,
  count, prefix-sum, produce the permutation old-index → new (bin, slot).
  Refuse loudly on bin overflow.
- *K2b apply permutation:* gather-scatter x and v into the new order
  (double-buffered through a second vector region or staged per page run;
  f is transient and just gets re-zeroed). This is the one genuinely
  scattered write phase, and it is page-bounded: an atom moves at most one
  bin per rebuild interval, so source and destination pages are neighbors.
- *K2c list build:* for each atom i, scan the 27 stencil bins (their pages
  held for the duration of the block's bins — a static, geometry-known page
  set), emit j-indices with `r² < (cutoff+skin)²` through WRITE-ALLOCATE
  holds on the neigh vector (never fetch a page about to be overwritten),
  minimum-image applied in the distance test. Counts pass + prefix sum +
  fill pass (standard two-phase CSR build). Flush the written neigh pages.
- *K2d metadata:* per-block stencil page sets and the working-set lower
  bound check (section 5), refused with numbers if violated.

**K3 — force.** Per block: zero its f page (write-allocate hold), hold its
own x pages and its stencil x pages (the static page set from K2d), then
stream its atoms' CSR neighbor runs from the neigh vector (sequential pages,
each entry exactly once): gather j position+type in one float4 load, LJ
arithmetic, accumulate force in registers, per-pair energy/virial into
block-local accumulators, tree-reduce, atomically merge. Newton off / full
list: every atom's force computed entirely by its owning block — no write
sharing anywhere in the design.

**K4 — integrate, second half.** Same holds and streaming shape as K1:
`v += dt/2·f/m`.

**K5 — thermo (every K steps).** Streaming reduction over x/v (KE,
temperature) plus the accumulated PE/virial; host reads back ~10 scalars.
This is the ONLY steady-state device-to-host traffic.

## 4. What is out-of-core

| structure | OOC? | pattern under pressure |
|---|---|---|
| `x` | yes | integrator streams it (pages fault in walk order, drop clean — read-mostly for K3, whose stencil sets bound the gather); the only structure with reuse both within and across kernels |
| `v` | yes | pure streaming, touched only by K1/K4/K5 — the easiest structure in the code |
| `f` | yes | write-allocate in, flush out, one exclusive page per block |
| `neigh` | yes — the showcase | written once per rebuild (write-allocate), streamed once per step, zero reuse; by far the largest structure (~an order of magnitude over x) |
| index class, LJ, accumulators | no, by design | O(atoms)/O(bins) small-constant arrays; paging the index of the paging is circular |

Scale framing: single rank, capacity = the tier stack. Every host-side
staging path (initial load, validation dumps) streams through bounded
buffers — nothing ever materializes a whole structure in one allocation.
(Multi-rank halo exchange is out of scope here and is the one thing this
benchmark deliberately does not model.)

## 5. Cache sizing — configurable, lower bounds only

Per vector, all configurable: page size, cache grouping (per-block / shared
/ sharded), slots per cache. The design imposes ONLY the lower bounds that
make each kernel's working set residentable, validated against the real
geometry every rebuild and refused with numbers otherwise:

- `x`: ≥ the largest per-block stencil page set + the block's own pages,
  × blocks sharing the cache, + headroom slot. Known exactly from geometry.
- `v`: ≥ the block's own page span (streaming only).
- `f`: ≥ one page per block sharing the cache.
- `neigh`: ≥ a small constant of in-flight streamed pages per block.

Two regimes, one code path, both enforced:
- **fits** (cache ≥ page count): upload/placement is complete and every
  timed kernel runs with zero faults and zero evictions — asserted after
  every step, hard error on violation.
- **doesn't fit**: pages fault under the lower-bound guarantee; the pair
  integrity counter (entries walked vs CSR total) and bin-population
  counter make any skipped work a loud failure.

## 6. Correctness against stock LAMMPS — the plan in detail

The physics contract: identical pair set, identical force law, identical
integrator ⇒ trajectories match stock up to floating-point summation order.
Validation is layered so each layer isolates one failure class:

1. **Imported initial conditions.** We do not reimplement LAMMPS's lattice
   and velocity RNG. Stock runs the deck with a step-0 dump of x and v; the
   bench reads that dump. Both codes then integrate the SAME state.
2. **Step-0 statics (tight).** With identical inputs, PE, pressure, and
   pair count at step 0 involve no time integration — they must match the
   stock log to float summation-order tolerance, and the pair COUNT must
   match exactly. This validates binning, minimum image, the list, and the
   force law in isolation.
3. **Short-horizon trajectory (tight).** Over ~10–50 steps (before Lyapunov
   divergence amplifies representation noise), per-atom max |Δx| and |Δv|
   against a stock dump within float tolerance. This validates the
   integrator and force accumulation.
4. **Double-precision validation build.** The vectors are typed; the bench
   compiles a double-element variant used only for validation, removing
   representation tolerance from layers 2–3 almost entirely. The float
   build is what gets benchmarked.
5. **Long-horizon invariants (statistical).** Over the full 250-step melt:
   NVE total-energy drift within a stated bound, net momentum ≈ 0, and
   temperature/PE time-averages against stock within thermal noise. Chaotic
   per-step energies are NOT compared late in the run — that comparison is
   meaningless and we will not pretend otherwise.
6. **Cross-regime bitwise identity.** The same run at resident and at
   out-of-core cache configurations must produce IDENTICAL results —
   paging must be invisible to physics. This is the vector abstraction's
   own correctness gate, independent of LAMMPS.
7. **CI shape.** A small deck (~4k atoms) with committed golden references
   (stock dump + log) runs layers 1–3 and 6 as a ctest alongside the other
   gpu_vector tests; the 256k melt is the perf benchmark with layer 5
   checks inline.

## 7. Benchmark harness

`clio_lammps_md_paged_bench` (benchmark/) + `test_gpu_vector_md` (test/):

- CLI: box/lattice/temperature or `--init <dump>`; steps, dt, cutoff, skin,
  rebuild cadence; per-vector page/cache/slot configuration; `--dtype
  float|double`; `--validate <dump> <log>`; `--resident-assert` (regime
  enforcement on).
- Output: thermo table, the per-phase time ledger (integrate / rebuild /
  force / thermo, device-timed), fault/evict counters per vector, and the
  validation verdicts.
- Stock reference: a helper script runs the deck under a stock LAMMPS
  binary to (re)generate golden files; small goldens are committed.

## 8. Build order

1. Layout + K1/K4 + K5 on x/v (integrator alone, ballistic test: exact vs
   closed form).
2. K2 binning/sort/CSR + K3 cell-direct force (no neigh vector yet, stencil
   scan computes pairs directly) → validates layers 1–2 with the fewest
   structures.
3. K2c/K3 switch to the device-built Verlet list on the paged neigh vector
   (the skin's rebuild amortization + the streaming showcase).
4. Regime enforcement + OOC configurations + layer 6.
5. Golden references, ctest wiring, perf ledger vs stock KOKKOS on the same
   deck.

What this deletes relative to the hooked integration, by construction: all
per-step uploads and downloads, the host flatten, the blob round-trip, the
ghost machinery, the two-range problem, and every "who owns the state"
ambiguity. What remains honest: the force kernel's arithmetic cost and the
paging machinery itself — which is exactly what a gpu_vector benchmark
should be measuring.

---

## Status (2026-08-20)

Implemented as `clio_lammps_md_paged_bench` (design above; code in
context-transfer-engine/adapter/gpu_vector/benchmark/). Stages 1-3 are
done and gated; stage 4 is partly done; stage 5 is not started.

### Correctness, resident: DONE, every gate green

| gate | result |
|---|---|
| statics vs stock LAMMPS | PE/atom **-6.7733683** vs stock's published **-6.7733681** |
| pair count vs an independent host double reference | **exact** (6,912,000 at 256k) |
| resort continuity (same state, new layout) | **bitwise**, PE rel 0.00e+00 |
| NVE energy drift | 6e-7 cold; 2.4e-3 hot, matching stock's own 1.9e-3 in double |
| ballistic trajectories vs a host float replica | **bitwise** |
| resident contract (zero faults/evicts) | asserted every step |
| gpu_vector suite | 12/12 |

### Performance, resident, 256k atoms

**5.50 ms/step against stock KOKKOS's 3.68 -- 1.5x.** For contrast the
earlier approach of hooking a pair style into LAMMPS was 12.5x. Zero bytes
of simulation state cross PCIe per step. What moved it: a coroutine
register ceiling (clang allocates 130 regs regardless of body, capping
occupancy at 7 blocks/SM), a resident fast path in the core that skips the
pin/fence/LRU when nothing can evict, sizing the coroutine lane to the
measured frame, a transposed+padded neighbour list (CSR lost to brute
force -- per-thread runs scatter a warp over 32 cache lines), and
amortizing the stencil holds over a CHUNK of rows.
The list build is recurring, not one-time: ~22 ms per rebuild against
stock's ~11, on the same `neigh_modify every 20` cadence.

### Checkpointing: implemented, and it LOSES when resident

`--ckpt N` flushes the live state into the tier stack, page-granular, no
host staging, verified by reading it back. It costs ~11 ms per 18.7 MB
against ~0.4 ms for the device-to-host copy a non-paged code must make
first. Two structural reasons: the flush goes through one block (the flush
path uses the block table's batch slots, and this design maps every CUDA
block onto one shared table), and when everything is resident there is
nothing for paging to save. It should win out of core; that is not
claimed until measured.

### Out of core: mostly working, one defect left

Working: the integrator is **bitwise exact** out of core at every block and
slot count tried; the first force call (statics) is exact on every run; a
full single step is exact 4/4; the resort passes its own gate every repeat
after being inverted from a scatter into a **gather** (each block owns its
destination row, so every destination slot has exactly one writer).

Remaining: with the resort disabled, roughly one run in three computes some
step's forces from the wrong data (up to 1.8% energy error). Disproved so
far -- read-after-write through the store (forcing durability does not
help), the resort, the integrator, the neighbour list, duplicate slots, and
shared-vs-per-block tables. The reader is seeing a WRONG FRAME. Reproduce:
`--md --lattice 40 --steps 4 --rebin 0 --temp 3.0 --cap 48 --page-kb 96
--blocks 4 --threads 64 --rowchunk 2 --slots 40`, repeated ~3x.

### Core defects found and fixed along the way

- **A silent round cap.** `RunToCompletion` stopped at max_rounds and
  returned normally, so a livelocked kernel was indistinguishable from a
  finished one and callers computed on partially executed work. Now
  `Yieldable::HitRoundCap()`, surfaced by CoroRunner, and the LAMMPS pair
  style refuses a partial kernel.
- **A fault-path livelock.** `RetryLostFetch` only retried `BeginFetch`,
  which fails exactly when no slot can be claimed -- and nothing in that
  loop ever freed one, because `StartEvictionAsync` runs once. Now
  re-armed on a declined fetch. Every out-of-core config that used to lose
  work now runs every iteration with bitwise-zero mismatches.
- Earlier the same day: a hold-vs-claim TOCTOU (Dekker handshake) and an
  identity placement way that makes "it fits, therefore it cannot fault" a
  theorem rather than a lottery.
