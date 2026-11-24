# The CTE data path: core, client, interposers, shm cache, write-back

Path shorthand: `CTE/` = `context-transfer-engine/`,
`CORE/` = `CTE/core/include/clio_cte/core/`,
`WTC/` = `CTE/core_shm_cache_writethru/`,
`RT/` = `context-runtime/`. Line numbers were checked on branch
`cte-shm-cache`. Where `WTC/DESIGN.md` or `WTC/HANDOFF.md` disagree with the
code, the code is described here and the conflict is listed in section 8.

## 1. Concepts

- **Tag**: a named group of blobs, `TagId = clio::run::UniqueId`
  (`CORE/core_tasks.h:802`). A blob is addressed by (TagId, blob name).
- **Blob**: bytes with a score and placement; written by `PutBlob` (offset,
  size), read by `GetBlob`.
- **Target**: a storage device = a bdev pool wrapped by `TargetInfo`
  (`CORE/core_tasks.h:303-325`): score (0-1, normalized log bandwidth),
  capacity, perf metrics, persistence level, bdev type. Tiers are configured
  per core under `storage:` with `score` (-1 = automatic) and
  `persistence_level` (`RT/config/clio_default.yaml:469-475`).
- **Score** of a put: `-1.0` = let the engine place it, `0.0-1.0` = explicit
  tier (`CORE/core_tasks.h:2604`).
- **Owner (home) of a blob**: `BlobHash(tag_id, blob_name)` (tag major,
  minor and `std::hash` of the name, combined boost-style,
  `CORE/blob_placement.h:57-68`) modulo the number of containers. The core's
  `ScheduleTask` (`CTE/core/src/core_runtime.cc:1264+`) routes blob verbs with
  `HashBlobToContainer` (`:11507-11530`): `DirectHash(BlobHash)`, `Local` for
  CAE node-local ingest tags, or `DirectId(FailoverContainer(owner))` when the
  owner is dead and failover is on. Tag creation routes by
  `DirectHash(NameHash(tag_name))`; target management routes Local.
- Core methods (`CTE/core/clio_mod.yaml`): blob data (`kPutBlob 15`,
  `kGetBlob 16`, `kDelBlob 18`, `kMultiPutBlob 48`, POD variants 43/44/50/51),
  tags (`kGetOrCreateTag 14`, `kDelTag 19`, `kGetTagSize 20`, `kTagXattr 58`,
  `kUpdateTagNames 55`, `kMultiDelTag 61`), metadata/placement (22, 23, 25,
  17, 47, 60), durability (`kSyncTag 56` = fsync of a tag, `kFlushMetadata
  33`, `kFlushData 34`, `kSetDurability 59`), queries (30, 31, 35),
  `kDynamicReorganize 46`. Note the manifest drift warning in
  `module-anatomy.md` section 2.

### Payloads

- `PutBlobTask` (`CORE/core_tasks.h:2558-2745`): `blob_data_` is a
  `ShmPtr<>`, optional vectored `segments_`; `SerializeIn` ends with
  `ar.bulk(blob_data_, size_, BULK_XFER)` (skipped for emulated puts).
- `GetBlobTask`: `BULK_EXPOSE` in `SerializeIn` (`:3032`), `BULK_XFER` in
  `SerializeOut` (`:3047`).
- `MultiPutBlobTask` (`:3317-3435`): one buffer `data_` + a serialized vector
  of `MultiPutDesc` (tag, name, offset, size, payload offset, flags, score);
  routed by its first record. Put flags: `kCtePutReplace 0x1`,
  `kCtePutDroppable 0x2`, `kCtePutAppend 0x4` (`:2287-2338`).

## 2. The client has two paths

`CORE/core_client.h` (design comment at `:593-610`):

1. **Shared-memory cache path** -- when the process has mapped the node's shm
   cache segment: synchronous `PutBlob`, `GetBlob`, `GetBlobSize` and
   `GetOrCreateTag` are served by copying between the caller's buffer and the
   segment. No task, no runtime allocation, nobody waits.
2. **Task path** -- everything else, and every `Async*` method ("a Future
   needs a task", `:608-609`).

### Connecting

- The client learns the segment from the **CTE core's Create**: the core
  writes `ShmCacheKeys{runtime_pid_, cache_pool_, root_off_, entry_pool_}`
  (`CORE/core_tasks.h:176-200`) into `CreateParams::cache_keys_` both on
  Create and for an existing pool (`WriteCreateOut` / `FillExistingCreate`,
  `CTE/core/src/core_runtime.cc:947-961`). `FindFrontCache` (`:963-1009`)
  finds the shm cache whose next-pool chain reaches this core.
- `ConnectCache(keys)` (`core_client.h:623-646`) refuses (task path) if the
  keys are invalid, if the client's `pool_id_` is neither the cache pool nor
  the entry pool (its tasks would bypass the cache), if there is no metadata
  allocator or the runtime pid differs (TCP client, remote node), or if the
  segment cannot be attached. Views are shared per process and never freed.
- `CLIO_CTE_CLIENT_INIT` (`CTE/core/src/content_transfer_engine.cc:56-145`)
  get-or-creates the core and binds the client to the front pool
  (`FrontPool()` = entry pool or cache pool) and the cache; on failure it
  keeps the core pool and logs that it uses the task path. `CLIO_CTE_POOL=maj.min`
  selects a pool explicitly. Runtime-internal clients (`Client(next_pool_id)`)
  never connect.

### `PutBlob` / `GetBlob`

- `PutBlob` (`core_client.h:724-750`): **always tries the cache first**
  (`TryPut(..., may_block=false)`); on `kOk` returns 0 with no task. On any
  other status (no cache, `kMiss`, `kTooLong` (name > 200), `kNoSpace`) it
  falls back to `AsyncPutBlob` + `Wait`. Score and context are not carried by
  the cache path.
- Task-path staging (`:503-562`): in a co-located runtime the caller's
  pointer is wrapped with `ShmPtr::FromRaw` -- no copy; in a client process,
  one `AllocateBuffer` + memcpy.
- `GetBlob` (`:770-816`): `TryGet` when flags are 0, no context, non-empty;
  otherwise `AsyncGetBlob` (runtime mode reads straight into the caller's
  buffer; client mode copies out of a staging buffer in `PostWait`).
- `GetBlobSize` and `GetOrCreateTag` similarly try the segment first.

**Default compose caveat**: the default chain runs the shm cache with
`md_only: true` (`RT/config/clio_default.yaml:740-764`) because the write-back
cache below holds the bytes. Then `TryPut`/`TryGet` always miss, client puts
and gets take the task path, and only sizes, tag lookups and tag attributes
come from shared memory. Data mode is `md_only: false` (e.g.
`CTE/benchmark/cte_config_shm_wtc.yaml`).

## 3. Interposers

`CORE/core_interposer.h`:

- `CoreInterposer : clio::run::Container` -- a module that sits in front of a
  "next pool" (`interposer_next_pool_`, default the core `kCtePoolId`) and
  receives the same core verbs (same method ids).
  - `GetNextPool()` returns the next pool (`:69`); `CoreContainer()` is the
    local container of that pool.
  - `ScheduleTask` delegates to the next pool's container, so blob verbs get
    the core's owner routing unless the interposer overrides it
    (`:104-111`).
  - `ForwardToCore(method, task)` runs the next pool's `Run(method, task)`
    **inline on the same task** (no new task, no copy), rc=1 if the container
    is missing (`:120-139`).
  - `GetTaskStats` returns the core's stats (`CoreTaskStats`) for core verbs;
    override only for your own verbs and delegate the rest (`:80-88`).
  - Forward* defaults implement the generated switches' default cases for
    core verbs (`:145-216`).
  - **Rules** (`:34-36`): compose the interposer **after** its next pool;
    number the interposer's own verbs **>= 100**.
- `DualTargetInterposer` (`:232-255`): a volatile pool plus a persistent
  chain (used by the write-back / write-through caches).
- Interposers in the tree: `cache/`, `checkpoint/`, `compressor/`,
  `indexer/`, `replication/`, `core_shm_cache_writethru/`
  (`CoreInterposer`); `cache_base/` (`DualTargetInterposer`) with
  `core_persist_write_back/` and `core_persist_write_thru/`; CAE
  `summarizer`.

### The default stack (`RT/config/clio_default.yaml`)

```
clients -> indexer 564 -> shm cache 568 -> write-back 567 -> volatile core 540
                                                         \-> replication 561 -> persistent core 512
```

(indexer `next_pool_id 568.0` `:787-791`; shm cache `next 567.0`,
`entry_pool_id 564.0` `:753-758`; write-back `next_volatile 540.0`,
`next_persistent 561.0`, `flush_pool 568.0` `:666-690`; replication
`next 512.0` `:614-618`.) The compressor (562) is commented out; enable it by
pointing the shm cache's next at 562.0. CAE 400, stream 565 and filesystem 560
point at 564. Composition order: bottom first.

## 4. The shared-memory cache module (`core_shm_cache_writethru`, pool 568)

### Segment

- One POSIX shm segment per cache pool per node,
  `clio_wtc_<pid>_<major>_<minor>` (`CORE/shm_wtc.h:376-379`), discovered by
  clients through a `Locator` registered in the runtime's metadata segment
  directory (`WTC/src/core_shm_cache_writethru_runtime.cc:186-209`).
- `Format` (`CORE/shm_wtc_impl.h:69-190`) lays out: `Root` (counters,
  `pushed_`/`flushed_`, stats, locks, 64 stripe mutexes), `BlobEntry[max_blobs]`
  in 64 stripes, `TagEntry[max_tags]`, name slots, the dirty-range log
  (`mpsc_ring_buffer<DirtyRange>`), and an owner map per 4 KiB granule.
- **Prefaulted** at Format with `SystemInfo::BulkFault` -- otherwise the
  writing client absorbs ~256 faults per MiB (`shm_wtc_impl.h:76-79`).
  Extent mode can also prefault in the client (`CLIO_PREFAULT`).
- **Extent mode** (`data_pool`, `md_only: false`): blob bytes live in leased
  4 KiB granules of a node-local RAM bdev instead of the segment.

### Write: `View::TryPut` (`shm_wtc_impl.h:835-927`)

Refuse (`kTooLong` / `kMiss` in md_only or with non-droppable flags /
`kNoSpace` when the log is nearly full and the caller may not block); under
the stripe lock find-or-insert the entry, check conflicts, ensure buffer
capacity (evicting by frecency, `hits * 2^(-age/half_life)`; a **pinned**
entry that must grow is refused with `kBusy`, untouched -- never wait for a
pin under a stripe lock: the flusher's zero-copy pin drops only after the
home acks, and the home's ack waits for an Invalidate here that needs that
stripe lock; callers back off unlocked, bounded, then take the task path /
write around), bump the
generation and seqlock, pin; **copy the bytes outside the lock** (a pin, never
a lock, protects the copy); unpin; `pushed_++` then push the dirty range to
the log.

### Read: `TryGet` (`shm_wtc_impl.h:929-988`)

A hit needs a resident entry that is complete or whose valid range covers the
request; seqlock-protected copy-out with up to 256 retries; misses are counted
by reason (absent, nonres, range, busy).

### Flusher (`WTC/src/core_shm_cache_writethru_flush.cc`)

- `FlushLog` is a periodic task (`flush_period_us`, default 50) per container;
  one runner; up to `flush_depth` batches in flight; pauses while chain reads
  wait.
- `PopAndCoalesce` (`:121-165`) pops dirty ranges (up to 8192 ranges / 65536
  pops / `flush_batch_bytes`), drops superseded ones, sorts and merges
  adjacent ranges per blob.
- **Zero-copy path**: `PinRecords` + `View::PinRange` (`shm_wtc_impl.h:1458-1479`)
  pin the segment bytes and `SendPinned` (`:242-270`) sends tasks whose
  payload `ShmPtr::FromRaw` points **into the segment**: local home ->
  `MultiPutBlob` to the next pool with `Local`; remote home -> this pool's
  `kHomeFlush` with `DirectId(home)`. (In the current code this sends one task
  per record.)
- **Staged path** (extent mode, or when pinning fails): one staging buffer per
  home, waiting (never dropping) for allocation; the local home is split into
  `flush_shards` shards by blob hash so one blob's ranges stay ordered; one
  `HomeFlush` per remote home.
- `LandGroups` awaits every future, retries failed homes record by record up
  to `flush_retries`, and records a sticky per-tag error on final failure;
  `flushed_` advances in pop order.
- `HomeFlush` (`:681-719`) on the home forwards the batch to the core and
  invalidates the blob's sharers **before acking**.

### Coherence and invalidation

- Each blob's **home** keeps a sharer directory of other containers that
  cache it (`HomeApply`, `:601-650`); a foreign write invalidates them.
- **Batched per home** (`WTC/src/core_shm_cache_writethru_data.cc:184-251`):
  one `InvLane` per home node, keys de-duplicated, **at most one Invalidate in
  flight per home**; keys arriving meanwhile form the next batch.
- `invalidate_on_flush: true` (default compose): writes send no invalidation
  of their own; the write-back cache's flush lands through pool 568 on the
  home, which invalidates then.
- Barriers: `DelTag` and `SyncTag` wait for `flushed_ >= pushed_` (polling
  every 10 us); `SyncTag` then syncs every remote home its ranges landed on and
  forwards to the core; sticky flush errors override rc 0
  (`WTC/src/core_shm_cache_writethru_runtime.cc:373-462`).

### Config keys (`WtcConfig`, `WTC/include/.../core_shm_cache_writethru_tasks.h:58-186`)

`next_pool_id` (null -> 512.0), `capacity` 256 MiB, `log_depth` 0 (auto:
`max(65536, capacity >> 10)`, cap 2^28; 65536 in md_only), `flush_batch_bytes`
16 MiB, `max_tags` 4096, `max_blobs` 65536, `max_fill` 16 MiB, `evict_sample`
16, `half_life_ms` 10000, `flush_period_us` 50, `flush_retries` 3,
`flush_shards` 8, `flush_depth` 4, `max_sharer_entries` 65536,
`invalidate_on_flush` false, `md_only` false, `entry_pool_id`, `data_pool`,
`data_capacity` 0. All appear in `RT/config/clio_default.yaml:740-785`. Env:
`CLIO_WTC_TRACE=1`, `CLIO_PREFAULT`, `CLIO_CTE_POOL` (not in the yaml).

### Stats at stop

A stop hook logs (warning level) `[shm-cache stats] node= puts= put_MiB=
refused: log= slot= buf= busy= miss= waits= flush_batches= flush_ranges= flush_MiB=
flush_skipped= fills= md_fills= get_hits= get_misses= (absent= nonres= range=
busy=) invalidations= evictions=`
(`WTC/src/core_shm_cache_writethru_runtime.cc:94-136`); clients read the same
counters with `Client::CacheStats()`.

## 5. Write-back and write-through (`cache_base`)

`CTE/cache_base/` (`CacheBase : DualTargetInterposer`), used by
`core_persist_write_back` (pool 567 in the default stack) and
`core_persist_write_thru`.

### Policies

- **Write-back = eventual consistency.** A write is acknowledged once it is in
  the volatile copy on the node that made it (`ScheduleTask` routes
  Put/MultiPut/DelBlob/TruncateBlob `Local` under write-back,
  `CTE/cache_base/src/cache_base.cc:167-186`). Background flushes send dirty
  ranges to the blob's owner and on to persistent storage. Reads go to the
  owner; **unflushed writes become visible elsewhere only after a flush
  (fsync)**. A read before a flush is not expected to be coherent.
- **Write-through = immediate visibility**: forward to persistent first, then
  update the volatile copy. Use it for data that must be visible at once.
- **Metadata is applied on its owner even under write-back.** Special puts
  (replace, append, put-if-absent/version) and **`kMetaBlob`** puts (clio-fs
  directory and inode blocks -- shared read-modify-write state) go to the
  owner after landing the cache's own buffered bytes (`:94-111`,
  `:1304-1315`), avoiding lost updates.
- One volatile copy per blob, held by its owner; per-blob operations are
  serialized by a coroutine-friendly per-blob lock (`Lock` polls with
  `yield(kLockPollUs)`, `:219-230`).

### Flushing

- Dirty extents per entry; `dirty_tags_` indexes dirty blobs by tag so a sync
  reads one tag's set instead of scanning every entry (`cache_base.h:542-548`).
- Triggers: periodic (`flush_period_ms`, default 10000), continuous
  (`flush_dirty_mb`, default 256: start a flush once that many bytes were
  dirtied), fill level (`flush_fill_pct`, default 80, checked at most every
  10 ms), and stop (bounded by the grace period; warns that unflushed data is
  lost) (`:136-165`, `:290-322`).
- `FlushAll` (`:997-1035`) flushes **foreign-owned blobs first, then local
  ones**, so a lock on a locally-owned blob is never held while waiting on
  another container. Local: up to 64 ranges in flight. Foreign: one buffer per
  owner, sent in 8 MiB batches through `flush_pool` with `DirectId(owner)`,
  capped at 256 MiB in flight.

### fsync (`SyncTag`, `:1845-1904`)

- Concurrent syncs of the same tag share one round.
- The leader **waits** for the flush of that tag (`FlushTagShared` ->
  `FlushAll`) before forwarding SyncTag to the persistent chain: fsync makes
  data visible and durable, and returns only after the flush completes.
- **One broadcast per fsync**: the client's `AsyncSyncTag` defaults to
  Broadcast and sets `kSyncFannedOut` (`CORE/core_client.h:1962-1973`,
  `CORE/core_tasks.h:6209`); each container then flushes only its own copies.
  Only an un-fanned sync fans out itself. This exists because a sync that
  broadcast from every receiver was N^2 messages.
- Errors: `kSyncNoSpaceRc = 28`, `kSyncIoRc = 5`
  (`CORE/core_tasks.h:6196-6198`). Core `fsync_mode: "deferred"` returns
  without syncing devices (`CTE/core/src/core_runtime.cc:7712-7714`).

### Config keys (`PolicyCacheParams`, `CTE/cache_base/include/clio_cte/cache_base/cache_config.h:97-232`)

`next_volatile` (required), `next_persistent` (null = volatile-only), `fill`
(`on_hit_count:2` | `on_read_miss` | `none`), `evict` (`lru` | `lfu`),
`flush_period_ms` 10000, `flush_fill_pct` 80, `flush_dirty_mb` 256,
`fallback_to_persistent` true (write through when volatile is full instead of
failing with `kPutNoSpaceRc`), `flush_pool` (null = this pool). Write-through
ignores `flush_period_ms` and `fallback_to_persistent`. All in
`RT/config/clio_default.yaml:643-695`.

### Tests must match the policy

Under write-back verify after fsync; under write-through verify immediately.

## 6. Data-path rules (from real defects)

1. **Go through the abstraction you were asked to use.** If a request says
   "use the shm cache module", the change goes through it -- even if a bypass
   measures faster today.
2. **Two client paths, cache first.** New synchronous client verbs that can be
   served from the segment belong on the cache path, with a task-path
   fallback; `Async*` verbs are task-only.
3. **No staging copies.** Flushers and forwarders pass pointers to bytes
   that already live in shared memory and pin them until the send completes;
   co-located clients wrap caller pointers (`ShmPtr::FromRaw`). A copy in a
   single-threaded flusher is a serial bottleneck.
4. **Prefault** shared segments at creation.
5. **Size defaults for the working set** (segment capacity, `max_blobs`, log
   depth) and check every configuration that inherits a default you change: a
   `max_blobs` that did not fit a 256 MiB segment disabled the cache and
   crashed the runtime.
6. **Create must survive a missing optional resource** ("serving tasks only"
   when the segment could not be created): every path checks it.
7. **Policies that depend on where a task executes belong in `ScheduleTask`**
   (write-back's Local routing is there), not in the handler.
8. **Batch per home** for flushes and invalidations; never one message per
   page or per write.
9. **Exposing Create outputs**: anything clients need from Create (cache keys)
   must also be written for an existing pool (`FillExistingCreate`).

## 7. Where config is documented

Every key above must appear, with its default and a comment, in
`RT/config/clio_default.yaml`, and in the module's compose-key table under
`site/pages/modules/`.

## 8. Known conflicts between design docs and code

1. `CORE/shm_wtc.h:483-486` says clients call `TryPut(may_block=true)`; the
   client passes `false` (`core_client.h:739`): a nearly-full log sends the
   client to the task path instead of blocking.
2. `WTC/DESIGN.md:56` says a full cache never diverts a write to the task
   path; `kNoSpace` does go to a task (the cache runtime orders it and
   eventually writes around).
3. `WTC/DESIGN.md:118` says one MultiPutBlob per home; the pinned path sends
   one task per record, the staged local path is sharded.
4. `max_name`, `inval_timeout_ms` (DESIGN.md) have no implementation.
5. `kTagAttrs` (103) is missing from `WTC/clio_mod.yaml` (present in the
   generated header).
6. Homes are computed as hash mod N with `DirectId`, duplicating the core's
   placement; the shm cache's `HomeOf` does not apply `FailoverContainer`
   while `CacheBase::OwnerOf` does
   (`WTC/src/core_shm_cache_writethru_runtime.cc:279-289`).
