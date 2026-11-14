# Handoff: shm write-through cache (branch `cte-shm-cache`, draft PR #1371)

For the next agent picking this up. Read `DESIGN.md` (same directory) first;
this file says where the work stands and what to do next. Delete this file
before the PR leaves draft.

## Goal (the user's words, paraphrased)

Local HPC runs (HDF5 VOL/VFD, many small I/Os on one node) pay a task round
trip per CTE op. Add a CTE chimod at the TOP of the chain that owns a
shared-memory segment; clients create blobs, write and read them directly in
that segment, and the chimod aggregates the buffered writes into a few tasks
down the chain. The client should have exactly two paths: with the cache
(SHM) and without it (plain tasks). Target: highly efficient reads, writes and
most metadata ops, correct across nodes.

## Decisions the user made (do not re-litigate)

- New chimod ON TOP of the existing `cache` chimod (565.0 -> 563.0).
- Data lives in chimod-owned, contiguous SHM buffers; metadata (blobs, tags)
  also in SHM. No WAL, no tiering.
- Writes log the EXACT modified byte ranges on ONE SHM MPSC queue; one
  background task drains it constantly and aggregates into single tasks.
- Eviction: anything not dirty is a candidate; frecency.
- Read misses fill the cache from the CTE.
- Must be coherent ACROSS NODES. The user simplified this to: track who
  caches a blob and invalidate them on write (no ownership/recall protocol).
- Tag creation is NOT done in the cache: it stays a task (ids come from the
  core). The user rejected name-hashed ids (renames keep ids; long paths).
- Remove the old client fast paths and the deferred-put pipeline
  (`AsyncPutBlobDefer`/`AsyncGetBlobDefer`, sieve, batching, `TryShmGet`):
  the cache replaces them. No-cache path = plain tasks.
- clio-fs gets its OWN similar SHM cache later (`filesystem/SHM_CACHE_DESIGN.md`);
  clio-fs skips the CTE cache and talks to the pool below it. Double
  buffering in corner cases is acceptable.

## What exists (5 commits on top of dev @ 0454b1892)

| Commit | What |
|---|---|
| ctp `Mutex::TryLock` fix | `TryLock` never lowered `try_lock_` on success, so every later TryLock on that mutex failed. Found by cache eviction (it TryLocks other stripes). Foundation header: everything rebuilds. |
| `MultiPutDesc::flags_` | Wire-format change. Core, compressor, replication batch handlers pass per-record flags (needed for `kCtePutDroppable`, which the VOL uses). Full rebuild, no `--target`. |
| the cache | see map below |
| clio-fs cache design doc | design only |
| exec bit on runner | |

Code map:
- `core/include/clio_cte/core/shm_wtc.h` + `shm_wtc_impl.h`: segment layout
  and every SHM operation (TryPut/TryGet/TryGetSize, eviction, fill,
  invalidate, flusher reads). Shared by client and chimod. 64-stripe blob
  table, pins instead of locks during memcpy, lock order stripe -> tag ->
  allocator, eviction uses TryLock on other stripes.
- `core/include/clio_cte/core/core_client.h`: client cache path (`Wtc()`,
  `WtcTryPut`, `WtcTryGet`, `WtcDonePut`, `WtcPlainContext`,
  `WtcAbsorbableFlags`) hooked into `AsyncPutBlob` (both overloads),
  `AsyncGetBlob` (both), `AsyncGetBlobSize`. `TryShmGet` stands down when
  bound to the cache.
- `core/src/content_transfer_engine.cc`: `CLIO_CTE_CLIENT_INIT` auto-binds to
  565.0 when the segment is mappable; `CLIO_CTE_SHM_CACHE=0` opts out.
- `shm_writethru_cache/src/`: `_runtime.cc` (create/segment/locator, tags,
  SyncTag, BarrierForward), `_data.cc` (task-path data verbs, fill,
  register sharer), `_flush.cc` (FlushLog, HomeFlush, Invalidate, sharer map).
  `autogen/..._lib_exec.cc` is hand-written dispatch (like the cache chimod).
- `context-runtime/config/clio_default.yaml`: cache ON in the default compose.
- Tests: `test/unit/test_shm_wtc.cc` (ctest `cte_shm_wtc`);
  `test/integration/shm_wtc_coherence/` (4-node docker, `run_tests.sh`);
  `benchmark/clio_cte_shm_wtc_bench.cc` + `cte_config_shm_wtc.yaml`.

## Validation state (honest)

- Built (targeted) and ran `test_shm_wtc` in the Linux devcontainer BEFORE
  the TryLock fix: the main test case PASSED (SHM round trip, write-through
  to core, exact ranges for gap writes, coalescing, read-miss fill, task-path
  ordering, DelBlob, 4 concurrent writers). The eviction case HUNG; root
  cause was the TryLock bug (census: 6 clean resident buffers, 0 evictable
  because TryLock always failed).
- NOT RUN: the eviction case after the fix; the full build; existing CTE
  unit tests; any multi-node test (HomeFlush/Invalidate/RegisterSharer and
  the DelTag broadcast have never executed); the benchmark.
- NOT DONE: merging `origin/dev` (15+ commits ahead at handoff).

## Next steps, in order

1. Merge `origin/dev`; full build with no `--target` (ABI change).
2. `ctest -R cte_shm_wtc`. Then the broader CTE unit suite: the default
   compose now includes the cache and `CLIO_CTE_CLIENT_INIT` binds to it, so
   any test using the default config runs through it. Separate genuine
   regressions from tests that assert on the old fast paths.
3. 4-node: `context-transfer-engine/test/integration/shm_wtc_coherence/run_tests.sh`
   (needs `CLIO_CORE_ENABLE_DOCKER_CI=ON`), plus the existing
   `coherence/` and `distributed/` suites.
4. Benchmark: runtime with `benchmark/cte_config_shm_wtc.yaml`, then
   `clio_cte_shm_wtc_bench` (prints task vs shm per-op ms).
5. Phase 4 (deletion of old fast paths / deferred pipeline). OPEN QUESTION
   for the user before doing the clio-fs part: clio-fs (`filesystem_client.h`,
   `adapter/libfuse/fuse_cte.cc`) depends on the deferred pipeline for
   write-behind, read-your-writes, pending size and fsync errors, and skips
   this cache. Deleting it before the clio-fs cache exists makes clio-fs
   writes synchronous. Non-clio-fs callers (lmcache, YCSB JNI, clio_cte_bench,
   bench_chain, Python bindings, test_multiput_defer, test_put_sieve, one VFD
   comment) can move now. List: `grep -rnE 'AsyncPutBlobDefer|AsyncGetBlobDefer|AwaitPendingPuts|AwaitPutsUntilSpace|DeferErrorCount|Defer[A-Z]\w*|FsyncDeferred|AwaitBatchPuts'`.
6. Docs: `context-runtime/config/clio_default.yaml` is updated; the `docs/`
   submodule (`docs/docs/deployment/configuration.md`) is not.

## Things I would review first (unverified assumptions)

- Flusher local group sends `MultiPutBlob` to the next pool with
  `PoolQuery::Local()`, assuming the core's MultiPutBlob routes each nested
  put to its owner (the old defer batch did the same). Confirm multi-node.
- `PoolQuery::DirectId(container)` for HomeFlush/Invalidate assumes one
  container per node in pool 565 (compose `pool_query: local`).
- HomeFlush passes the writer container in `route_tag_id_.major_` (minor
  `0xFFFFFFF7`) and restores the route before forwarding; core MultiPutBlob
  must not depend on that field after dispatch.
- The writer is invalidated too whenever another container also wrote the
  blob: N-1 disjoint writers exchange invalidation messages every flush.
- `WtcDonePut`/`WtcTryGet` allocate a task object per op to return a
  completed Future; likely the dominant client cost now. Candidate for a
  cheaper completed-future type.
- FlushLog period 50 us: check idle CPU (dev recently had idle-CPU work).
- A `MultiPutBlob` record whose put flags conflict with the cached blob's is
  reported as failed instead of forwarded.
- Processes that cannot map the segment (TCP clients, remote nodes) stay on
  the core pool unless `CLIO_CTE_POOL=565.0`; writes through the core bypass
  invalidation.
- The HDF5 VFD goes through clio-fs, so it does not benefit from this cache;
  the VOL does.

## User working preferences that apply here

- PRs target `dev`. Stage explicit paths (no `git commit -a`).
- Foundation-level rebuilds (ctp/runtime headers, task ABI): use the native
  Windows build, not a full devcontainer rebuild; never bind 0.0.0.0 (use
  127.0.0.1); do not trigger Windows Defender.
- Cross-node fixes: build a failing multi-node reproducer first.
- Timing output in milliseconds. New functions need doxygen docstrings,
  new C/C++ files need the BSD header, functions under 100 lines.
