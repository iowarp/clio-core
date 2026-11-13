# CTE redesign: separate volatile and persistent cores, with a policy-named cache

Status: plan, not started. Written 2026-11-12. Nothing here is implemented.

## 1. Problem

One `clio_cte_core` instance today holds every tier: VRAM (`hbm`), RAM,
NVMe and long-term file or object stores. It also owns every durability
decision:

| Responsibility in the core | Where (`core/src/core_runtime.cc`) |
|---|---|
| Persistence filtering at placement (`min`/`max_persistence_level`, TTL health) | `ExtendBlob` target filter, ~`:10111-10180` |
| Never lower a blob's durability on reorganize | `DurabilityFloor` `:8191`, used at `:4081` |
| Periodic flush of volatile primaries to persistent tiers | `FlushData` `:7849` -> `MoveBlobToPersistent` `:7997` -> `RelocateBlob` `:8073`; timer started in `Create` `:1144`; stop hook `RegisterStopFlush` `:1191` |
| fsync durability | `SyncTag` `:8253` (called from `filesystem_runtime.cc:642`) |
| Read-buffering tier | `read_cache: true` devices, `PromoteToReadCache` `:5362`, `REPLICA_READ_CACHE` |
| Replica slots with persistence semantics | `REPLICA_PERSISTENT` / `REPLICA_CACHE` / `REPLICA_READ_CACHE`, replica write path `:2918-3080` |
| Restart: drop volatile blocks at WAL replay | `:9184-9250`, `:9290-9300`, `DropRestoredCacheReplicas` `:9396` |

Two goals pull against each other: putting data on the fastest tier, and
never losing durable data. Mixing them in one module produced a real data
loss:

- A score-1.0 `hbm` tier sat next to a `long_term` file tier.
- To get model pages into VRAM, a reorganize had to move them off the file
  tier. That needed an escape hatch from `DurabilityFloor`
  (`accept_durable_moves`, added experimentally and to be deleted).
- The moved pages' file blocks were freed. KV-cache writes reused them,
  and the VRAM copies were gone at exit.
- Result: corrupted pages in the stored model.

The rule this plan establishes: **a core is either entirely volatile or
entirely persistent. Moving data between the two is the job of a cache
module above them, never of a core.**

## 2. Target architecture

```
            (CAE / gpu_vector / filesystem / clients)
                              |
                 compressor (optional, 512.0)
                              |
            write_thru_cache | write_back_cache   <- policy modules
                 /                         \
        next_volatile                    next_persistent
             |                                  |
      cte_volatile (core)              [replication] -> cte_persistent (core)
      hbm, ram, nvme scratch              file / object / long-term NVMe
      no WAL, empty at start              WAL, every blob durable
```

### 2.1 Cores
- **Same `clio_cte_core` module, two compose entries.** Each core's tiers
  are uniform in durability, so a core needs no persistence logic: it places
  by score, moves data between its own tiers, and evicts on request.
- **`cte_volatile`:** `hbm`, `ram` and optional NVMe used as scratch space.
  No metadata log, so it is empty after every start. It may refuse a put
  for lack of space, and that is a normal outcome.
- **`cte_persistent`:** file, object and long-term NVMe targets. Metadata
  log on. A blob it accepts is durable.
- **Config validation:** a core whose targets mix volatile and `long_term`
  persistence levels is a config error. This replaces the runtime durability
  rules.

### 2.2 Policy-named cache modules
- Rename the cache module after the policy it implements. Proposed:
  - `clio_cte_write_thru_cache`: strongly consistent; the ack follows the
    persistent write.
  - `clio_cte_write_back_cache`: the ack follows the volatile write; dirty
    data is flushed later.
- Both are thin modules over one shared base library, so lookup, fill,
  coherence and eviction are written once.
- **Config:**
  - `next_volatile`: pool id of the volatile core.
  - `next_persistent`: pool id of the persistent chain. May be omitted for a
    purely volatile deployment; the cache then forwards everything to
    `next_volatile`.
  - Policy keys:
    - `fill`: `on_read_miss` | `on_hit_count:N` | `none`
    - `evict`: `lru` | `lfu`
    - `flush_period_ms` (write-back only)
    - `locality` (see 4.1)
- **Two next pointers is new.**
  - `CoreInterposer` (`core/include/.../core_interposer.h:50-61`) has
    exactly one `interposer_next_pool_`.
  - Add a two-target base (or generalize to N named targets) instead of
    hacking the cache module alone.
  - Every module still forwards whole tasks; no new task types are needed
    for this.

### 2.3 Operation semantics

| Op | write_thru_cache | write_back_cache |
|---|---|---|
| PutBlob | 1) put to persistent; 2) update/insert volatile copy tagged with the persistent version; 3) ack. On volatile refusal: ack anyway (the copy is optional). | 1) put to volatile, mark dirty; 2) ack. If volatile refuses: put straight to persistent (write-around). |
| GetBlob | volatile hit with a current version -> serve. Miss -> read persistent, serve, then fill volatile per `fill`. | same; a dirty copy is always current |
| DelBlob / Truncate | persistent first, then drop the volatile copy | drop both; a dirty copy is dropped without flushing |
| Flush / fsync (`SyncTag`) | no-op (already durable) | write dirty copies of the tag to persistent and wait |
| Eviction | volatile is full -> drop clean copies in policy order | dirty copies are flushed before they are dropped |
| Stop | nothing | flush all dirty copies (replaces `RegisterStopFlush`) |

- **Coherence:** the volatile copy records the source version (the
  persistent blob's `last_modified_`). A read checks the copy against the
  persistent version only when the cache cannot prove the copy is current.
  On a single node every write goes through the cache, so a version bump the
  cache itself made is enough. Multi-node invalidation reuses today's
  `RegisterReplicaContainer` owner-invalidates-copies protocol, moved to the
  cache layer.
- **Compressed data:** the compressor sits **above** the cache, so both
  cores store the same compressed bytes. A VRAM copy is then the compressed
  page itself, and nvCOMP decodes straight from it into the gpu_vector frame,
  with no staging.
  - `clio_cte_locate` (`context-runtime/src/ipc/ipc_gpu2cpu.cc:880-912`) has
    a single registry entry today. Key it by pool (or let the cache register
    the volatile core's locator), so the compressor's direct-decode lookup
    asks the volatile core.

### 2.4 What gets deleted from the core
Delete each item only after its replacement in the cache module passes the
same tests.
- `FlushData` periodic task, `MoveBlobToPersistent`, `RelocateBlob`'s
  durability use, `RegisterStopFlush`
- `DurabilityFloor` and its use in `ReorganizeBlobInternal`
- `read_cache` device option, `PromoteToReadCache`, `REPLICA_READ_CACHE`,
  the read-cache LRU/skew counters
- `accept_durable_moves` (experimental; delete outright)
- Persistence filtering in `ExtendBlob`. Keep the TTL-health filter, which
  is about device failure, not durability policy.
- `REPLICA_CACHE` slot use by the cache module; the volatile core holds the
  copy as its own primary
- WAL replay's volatile-block dropping, which can't happen once the volatile
  core has no WAL

Keep: `REPLICA_PERSISTENT` replicas, which belong to the replication module
and live entirely inside the persistent chain.

## 3. Migration plan

Each phase builds, passes the existing CTE suite, and is independently
revertible. Task structs are ABI: after any task-struct change, rebuild
everything (`AGENTS.md`).

1. **Two-target interposer base.**
   - Add the base class and config parsing for `next_volatile` and
     `next_persistent`.
   - Unit test: a stub module routes puts and gets to the right pool.
2. **`write_thru_cache` module** (new; leave the old `clio_cte_cache` in
   place).
   - Implement the table in 2.3 for single-node operation.
   - Tests:
     - durability across restart (persistent intact, volatile empty)
     - volatile refusal still acks
     - read fill on miss
     - stale copy never served after a write
     - eviction under a small volatile core
3. **Config validation.** Reject cores that mix volatile and persistent
   targets; give a clear error naming the offending device. Update
   `context-runtime/config/clio_default.yaml` and
   `docs/docs/deployment/configuration.md` (AGENTS.md rule).
4. **Move fsync.** Route `SyncTag` and the filesystem's fsync through the
   cache module. `write_thru_cache` needs nothing; prepare the hook for
   write-back.
5. **`write_back_cache` module.**
   - Dirty tracking, periodic and stop flush, flush-before-evict.
   - Tests:
     - crash before flush loses only unflushed writes
     - fsync makes them durable
     - no dirty copy is ever dropped silently
6. **Compressor integration.**
   - Put the compressor above the cache.
   - Make the locator per-pool.
   - Test: a compressed page in the volatile core decodes straight from
     VRAM (path counter `tier_d2d` > 0, `spill_stage` = 0 on hits).
7. **Multi-node coherence.** Move the owner-invalidation protocol from the
   core's `REPLICA_CACHE` handling into the cache base. Rerun the coherence
   integration suite (`test/integration/coherence`).
8. **Fold in the old cache module.**
   - Its per-node raw local copy becomes a `locality: submitter` option
     of the policy modules.
   - Retire `clio_cte_cache` and update the default compose and
     `clio_repo.yaml`.
9. **Delete the core paths** listed in 2.4. Rerun the full suite, including
   filesystem fsync tests and the WAL restart tests.

## 4. Open questions for the implementer
1. **Locality vs durability:**
   - Is the old cache module's per-node copy (a copy on the submitting node)
     a `locality` option of the new modules, as proposed?
   - Or a separate module stacked above them?
2. **Where replication sits:**
   - Proposed: under `next_persistent` only, since volatile copies need no
     replicas.
   - Confirm that failover reads (replication serving a dead owner's blob)
     still work with the cache above.
3. **Fill rule default:** `on_read_miss` is simplest, but a one-pass scan
   (an LLM prompt touches 25k pages) floods the volatile core. Consider
   `on_hit_count:2` as the default for large stores. The hotset ranking
   (`core/src/data_organizer/hotset_organizer.cc`) could move here.
4. **Data organizers:**
   - Organizers that reorganize within one core are unaffected.
   - Organizers that existed to cross the durability boundary (promote
     durable pages to VRAM) become cache fill policies.
5. **Naming:** `write_thru_cache` / `write_back_cache`, or
   `cache_write_through` / `cache_write_back`? The pool name and the module
   name should match.

## 5. Interim configuration (until this lands)

For workloads that need data in VRAM now: use one **fully volatile** CTE
core, with `hbm` at score 1.0 plus NVMe `file` targets marked volatile for
extra space, and no metadata log. Data is loaded into it each run (for
clio-infer, by assimilating in the same runtime that runs inference) and is
gone at exit. Do not put a durable store and a volatile tier in one core
again.
