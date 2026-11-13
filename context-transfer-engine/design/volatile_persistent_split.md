# CTE redesign: separate volatile and persistent cores, with a policy-named cache

Status: in progress on branch `cte-volatile-persistent-split` (from `dev`).
Written 2026-11-12; amended the same day (section 6) with three additions:
the volatile core runs without a WAL, the core loses its async data flush
entirely (the write-back cache owns it), and a cache whose volatile core is
full can fall back to the persistent core.

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
  - **The WAL is disabled, not just unused** (amendment 6.1). A volatile
    core ignores `metadata_log_path` (with a warning if one is set), opens
    no transaction-log shards, writes no snapshot, starts no
    `FlushMetadata` timer, and treats a restart as a fresh start. Logging
    metadata for bytes that cannot survive a restart only costs put latency
    and produces a log whose replay must then throw the blocks away.
- **`cte_persistent`:** file, object and long-term NVMe targets. Metadata
  log on. A blob it accepts is durable.
- **Config validation:** a core whose targets mix volatile and `long_term`
  persistence levels is a config error. This replaces the runtime durability
  rules.
  - A core states its class with `durability: volatile | persistent`. When
    the key is absent the class is inferred from the devices (all
    `volatile` -> volatile, none `volatile` -> persistent).
  - Errors, each naming the offending device: a `volatile` device in a
    persistent core, a `temporary`/`long_term` device in a volatile core,
    and a `ram`/`hbm`/`pinned` bdev marked anything but `volatile`.
  - `temporary` counts as persistent: it survives a process restart, which
    is the boundary the WAL protects.

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
    - `fallback_to_persistent` (see 6.3)
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
| PutBlob | 1) put to persistent; 2) update/insert volatile copy tagged with the persistent version; 3) ack. On volatile refusal: drop any copy, ack anyway (the copy is optional). | 1) put to volatile, mark dirty; 2) ack. If volatile refuses: evict, retry once, then write around to persistent when `fallback_to_persistent` (6.3), else fail with `kPutNoSpaceRc`. |
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
- All asynchronous data flushing (amendment 6.2): the `FlushData` periodic
  task and its timer in `Create`, `MoveBlobToPersistent`, `RelocateBlob`'s
  durability use, `RegisterStopFlush`, the move step of `SyncTag`, and the
  `flush_data_period_ms` / `flush_data_min_persistence` config keys
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

## 6. Amendments (2026-11-12)

### 6.1 A volatile core has no WAL
- `durability: volatile` (stated or inferred) clears `metadata_log_path`
  at config load and logs a warning when it was set. Every WAL path in the
  core already keys off that path being non-empty, so this one decision
  turns off: the per-worker transaction-log shards, the snapshot, the
  periodic `FlushMetadata` task, restart restore and replay, and
  `DiscardPersistentMetadata`.
- A restart of a volatile core is a fresh start: no snapshot to restore,
  no blocks to reserve, and nothing for WAL replay to drop. This makes the
  "drop volatile blocks at replay" code (`DropRestoredCacheReplicas` for
  volatile tiers, the replay filters) dead for volatile cores; it is deleted
  in phase 9 with the rest of 2.4.
- Test: a volatile core given a `metadata_log_path` creates no log files,
  and a restarted runtime finds it empty.

### 6.2 The core does no asynchronous flushing
- The core stops moving bytes between durability classes on its own. Once
  a core is uniform there is nothing for a flush to move, so instead of
  waiting for phase 9 this lands with config validation (phase 3):
  - `FlushData`'s periodic timer and `RegisterStopFlush` are removed from
    `Create`; `MoveBlobToPersistent` and its helpers are deleted.
  - `Method::kFlushData` stays as a verb (benchmarks and tools call it),
    with new meaning on the core: sync every device and the WAL — "make
    what this core accepted durable". On a volatile core it is a no-op.
    The write-back cache interposes it as "flush all dirty copies, then
    forward".
  - `SyncTag` on the core loses its move step: it syncs the devices that
    hold the tag's blocks and the WAL. Moving dirty data is the write-back
    cache's job, which runs before it forwards `SyncTag`.
  - `flush_data_period_ms` and `flush_data_min_persistence` are removed
    from the core config; `fsync_mode` stays (the cache reads the reply).
- The write-back cache owns all of it: a periodic flush (`flush_period_ms`,
  default 10000, 0 = off), a stop hook bounded by the stop grace period
  (replaces `RegisterStopFlush`), flush on `SyncTag`/`FlushData`, and
  flush-before-evict.
- Write-through has nothing to flush; its `FlushData`/`SyncTag` forward to
  the persistent chain unchanged.

### 6.3 Volatile-full fallback to the persistent core
- Policy key `fallback_to_persistent: true | false` (default `true`).
- The sequence when the volatile core refuses a put with
  `kPutNoSpaceRc`:
  1. Evict in policy order until the bytes the put needs are freed: clean
     copies are dropped; dirty copies are flushed to persistent, then
     dropped (never silently).
  2. Retry the volatile put once.
  3. Still refused:
     - `fallback_to_persistent: true` -> write around: if this blob has a
       dirty copy, flush it first, drop the copy, then put to persistent.
       The blob is now clean and uncached; a later read may fill it.
     - `false` -> the put fails with `kPutNoSpaceRc` (for deployments that
       must never silently slow down to disk speed). The volatile copy, if
       any, is left unchanged.
- Write-through always writes persistent first, so it never needs the
  fallback for durability; a volatile refusal just drops the stale copy.
  The key is accepted and ignored there.
- A cache with no `next_persistent` has nothing to fall back to; it
  returns the refusal.
- Fills on read follow the same eviction step but never fall back (a fill
  is optional): a refused fill serves the read from persistent and leaves
  the blob uncached.
- Tests: a write-back put larger than the volatile core's free space
  succeeds with fallback on and fails with `kPutNoSpaceRc` with fallback
  off; after fallback, the blob reads back correctly and survives restart.

### 6.4 Decisions on the open questions (section 4)
- Names: `clio_cte_write_thru_cache` and `clio_cte_write_back_cache`,
  module names `write_thru_cache` / `write_back_cache`, matching pool names.
  One shared base class (`CacheBase`, header-only in `cache_base/`) holds
  lookup, fill, eviction and the two-target forwarding.
- Pool ids: the persistent core keeps the canonical `512.0`
  (`kCtePoolId`), so every client that addresses the core directly reaches
  the durable store; the volatile core is `520.0` (`kCteVolatilePoolId`);
  the policy caches are `566.0` (write-through) and `567.0` (write-back).
- Routing: the policy caches route blob verbs to the blob's owner
  container exactly as the core does (the `CoreInterposer` delegation), and
  reach the volatile core on that node with `PoolQuery::Local()`. Every
  operation on a blob therefore passes through one cache container, which
  is what makes the single-node coherence argument in 2.3 hold on a
  cluster too. The submitter-locality copy is phase 8's `locality` option.
- Fill default: `on_read_miss`. `on_hit_count:N` is implemented for large
  one-pass scans.
- Replication stays under `next_persistent` only.
