---
name: gpu-vector-data-organizer
description: Use when designing, implementing, or evaluating a data organizer (tier-placement policy) for CLIO -- the CTE core's DataOrganizer factory, the gpu_vector paged benchmarks, or an application that wants its hot data in fast tiers and cold data in slow ones. Covers extracting the workload's access pattern, deciding between a custom DataOrganizer and ReorganizeHint phases, wiring it into the factory and config, and proving (or disproving) the benefit with single-node tiered tests. Triggers on "data organizer", "organizer", "tiering policy", "prefetch policy", "ReorganizeHint", "frecency", "which tier", "hot/cold data", "page placement".
---

# GPU Vector / CLIO Data Organizer

CLIO places blobs across tiers by **score**: the CTE core's `MaxBwDpe` puts a
blob on the tiers whose `score <= blob_score`, preferring the highest-scored of
those. A data organizer is a policy that rescores blobs over time so that what
the application is about to touch sits in a fast tier and what it is done with
sinks. It is application-specific by nature: the right policy follows from the
access pattern, and a policy that is right for a stencil is wrong for a graph.

There are exactly two levers, and most real organizers use both:

1. **A `DataOrganizer` in the CTE core factory** -- a periodic coroutine that
   sees every blob's stats and rescores them (`ReorganizeBlobInternal`).
   Reference implementation: `FrecencyDataOrganizer`.
2. **`ReorganizeHint(int)` from host code** -- the application tells the
   organizer which *phase* it is in. The integer is opaque to the core; the
   organizer reads it with `server->OrganizerHint()` and decides what it means.

Read `REFERENCE.md` in this directory before writing code: it holds the file
pointers, the score-direction rule everyone gets backwards once, and the
measured migration costs that decide whether a policy can pay for itself.

## Phase 1 -- Extract the access pattern (before any code)

Find the kernels that touch the paged data and answer, for each, with
evidence from the source (not a guess):

- **Order**: sequential, strided (what stride, in pages?), pseudo-random
  (seeded how?), pointer-chasing (graph/list -- the *data* decides the order).
- **Reuse distance**: is a page touched once per pass (streaming, e.g. kmeans),
  re-read every pass (weights), or reused within a sliding window (grayscott's
  3-plane stencil)? Reuse distance vs cache size is what decides whether a
  cache hit is even possible.
- **Read/write mix**: read-only, write-only, read-modify-write. Writes go
  through `CoBeginFlush`/`CoEndFlush` at the write site; the *lazy* path --
  `FlushResidentToCte()` and any checkpoint/persist that walks the whole vector
  -- is a separate, sequential, write-only pattern that usually deserves its
  own phase.
- **Phases**: does the workload alternate between coordinated regimes (build
  list -> force pass -> resort; forward -> backward -> update)? Each boundary
  is a `ReorganizeHint` candidate.

Deliverable: a *simplified kernel* -- a few lines that state exactly which
parameters drive the address sequence (iteration counter, block id, offset,
stride, RNG seed, dataset structure). If you cannot write it, you do not yet
understand the pattern. Attach it to the design.

The six paged benchmarks under
`context-transfer-engine/adapter/gpu_vector/benchmark/` are worked examples
whose headers already state their pattern (streaming, full-reuse, stencil,
scatter, MLP weights, MD neighbour lists). Start from the closest one.

## Phase 2 -- Dataset research (only when the data decides the order)

Skip this for counter-driven patterns. For graphs, maps, lists, sparse
matrices, KV stores: the access order is a property of the dataset, so read
about the dataset -- degree distributions, locality, community structure,
whatever governs which pages are neighbours in access time. Cite what you
find; a policy built on an assumed distribution should say so. The GNN
benchmark (`benchmark/gnn/`) is the in-tree case where this mattered.

## Phase 3 -- Design and implement the organizer

Decide the policy from Phase 1, then pick the lever(s):

**Custom `DataOrganizer`** when the policy can be computed from blob stats
(`OrganizerBlobStat`: current score, last read/modified, access count, size)
plus the phase hint. Steps:

1. Subclass `DataOrganizer` (`core/include/clio_cte/core/data_organizer/`);
   implement `Reorganize(Runtime *server, u32 replica_id)` as a coroutine and
   `GetName()`. Copy the shape of `FrecencyDataOrganizer`.
2. Inside `Reorganize`: `server->CollectOrganizerBlobStats(replica_id, stats)`
   (already partitioned across replicas), compute a target score per blob,
   and `CLIO_CO_AWAIT(server->ReorganizeBlobInternal(tag, name, score, rc))`
   only when it moves by more than `score_difference_threshold`. Read
   `server->OrganizerHint()` once per round if the policy is phase-aware.
3. Register the name in `DataOrganizerFactory::Get`
   (`core/src/data_organizer/data_organizer.cc`) and add the source to the
   CTE core CMakeLists.
4. Config: `organizer: <name>`, `organizer_tasks`, `organizer_period_ms`
   under the `clio_cte_core` compose entry. Document new keys in
   `context-runtime/config/clio_default.yaml` and
   `docs/docs/deployment/configuration.md` (AGENTS.md rule).

**`ReorganizeHint` phases** when the application knows something the stats
cannot show -- "the next pass is sequential", "we are about to checkpoint",
"iteration k of a strided sweep". Call `CLIO_CTE_CLIENT->ReorganizeHint(k)`
(broadcast; returns the task code) at the phase boundary, and give the
organizer a small enum of meanings. Keep the vocabulary tiny and written down
next to the organizer: the core will not validate it.

**Device-side prefetch** is the third, narrower tool: a kernel can raise the
score of pages it is *about* to need through `PodMultiScore`
(`gpu_vector/prefetch.h`, `PrefetchPolicy`). Use it for lookahead inside a
pass; use the organizer for decisions across passes.

Rules that are not optional:
- Higher score = faster tier. A page starts at `kVectorBlobScore = 0.5`, so a
  tier scored above 0.5 is unreachable unless something promotes the page.
- Under 100 lines per function, docstrings on every new function, BSD header
  on every new file, `simple_test.h` (never Catch2) for tests, and check
  `Create` return codes in tests -- all per AGENTS.md.
- Never log in lock paths; the organizer runs on a worker.

## Phase 4 -- Prove it, single node, with tiers

A policy is only worth keeping if a measurement says so. Build the smallest
test that can show the effect:

1. **Two or three tiers with a small fast tier**, so the working set does not
   fit: copy `test/unit/test_data_organizer.cc`'s fixture (16 MB DRAM at 1.0
   over a file tier at 0.2, short `organizer_period_ms`) or run a paged bench
   with `--hbm-mb` small and `--nvme-mb` set so pages actually spill.
2. **A/B the policy**: `organizer: none` vs yours, same deck, same seed.
   Report faults, evictions, and wall time -- the benches print
   `faults= evicts= ms=` lines, and the harness's `REQUIRE_EVICTS` idea
   applies: a run with zero evictions proves nothing about placement.
3. **Cover the intended regime and one it should not help**: e.g. a streaming
   pass (no reuse -> a promoter cannot help) and a reuse pass. The point is a
   map of where the policy pays, not a single green number.
4. **Cost the migrations**: a promotion is a page read plus a page write
   (~271 us for 256 KB measured) and can save at most the difference between
   a slow-tier and a fast-tier fault (~46 us measured). If the policy promotes
   more pages than it later hits in the fast tier, it loses; gate on stalls
   or on the hint rather than on position alone.
5. Write the result down either way. "Always helps" is fine; "never helps" is
   fine *with the reason* (reuse distance exceeds the fast tier, migration
   cost exceeds the fault delta, the pattern is undetectable from stats).
   Put the numbers in the organizer's header comment and the test's docstring.

Register the test in `context-transfer-engine/test/unit/CMakeLists.txt` with
a `cte` label, and rebuild everything (not one target) after touching any
task struct.
