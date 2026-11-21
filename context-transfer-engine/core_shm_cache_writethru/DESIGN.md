# core_shm_cache_writethru — design

Status: proposal (branch `cte-shm-cache`).

## Problem

HDF5 VOL/VFD reach the CTE through task APIs. Every `PutBlob`, `GetBlob`,
`GetBlobSize`, `GetOrCreateTag` allocates a task, serializes it, wakes a
worker, walks the interposer chain (cache → indexer → compressor →
replication → core → bdev) and signals completion. For large, distributed I/O
that overhead is noise. For an HPC rank doing many small writes against a
node-local filesystem it dominates.

The client already tries to dodge this with fast paths (`TryShmGet` over the
core's metadata mirror, the deferred-put registry). They live at the END of
the chain, so they must second-guess everything above them: compressed?
replica? generational? emulated? `CLIO_FORCE_NET`? Writes have no fast path at
all (core_client.h:863 — a client cannot pin blocks against the DataOrganizer).

## Idea

Put a write-through buffer at the TOP of the chain, owned by a new chimod,
in a shared-memory segment that clients map directly.

- The client creates blobs **in the segment** (tags still come from the
  core, see Tags): metadata into SHM
  tables, data into chimod-owned contiguous SHM buffers. No task.
- Every write appends the exact byte range it modified to one SHM
  **MPSC log**. A single background task drains the log continuously and
  aggregates it into **single tasks** (`MultiPutBlob`) down the existing
  chain.
- No WAL, no tiering, no placement: it is a temporary buffer. Durability is
  the chain's job once a range lands.
- Because the chimod owns the bytes, nothing below it can move them, so
  the reason a client write fast path was impossible before does not apply.

The chimod also implements every CTE method as a `CoreInterposer`, so task
callers bound to its pool (remote nodes, tools, the no-mmap TCP client) see a
coherent view.

## Client: two paths

```
Client::PutBlob(...)
  if (shm_ != nullptr) return shm_->Put(...);   // done-future
  return task path;                               // plain task, nothing else
```

- **No cache** (`shm_ == nullptr`): every call is a plain task. `TryShmGet`,
  `TryReadBlobShm`, the RAM-bdev mapping, the replica/transform/generational
  guards and the whole deferred-put pipeline are removed from the client
  (see "Replaces the client deferred-put pipeline").
- **Cache** (`shm_` attached): the client uses the segment. Ops the segment
  cannot express (name too long, flagged/emulated/replica op) go as a task
  to the **cache pool**, which orders them against the log (barrier first).
  A full cache never diverts a WRITE to the task path; see Capacity.

Attach: the chimod registers its root in `MetadataDirectory`. The client
attaches at `Init` when its pool's directory slot points at a
core_shm_cache_writethru root. TCP/remote clients never attach and use the task
path.

## Segment layout

One `PosixShmMmap` segment, named `ShmSegmentName(runtime_pid, pool_id)`
(the RAM-bdev pattern), formatted as a `CLIO_TASK_ALLOC_T` multi-process
allocator the way the runtime's metadata segment is. Everything in it is
reached by offset; every type in it is POD or a ctp SHM container.

```
ShmWtcRoot     version, ready, node_id, limits, counters,
               log_pushed_seq, log_flushed_seq, table lock
TagTable       open-addressed; key = TagId; value = flags (kCreated),
               sticky_rc (first failed flush of this tag)
BlobTable      open-addressed; key = (TagId, blob name); value =
               buffer (ShmPtr), cap, size, pending, write_gen, seqlock,
               hits, last_access_ns, score, inval_epoch,
               flags (kComplete, ...)
Sharers        open-addressed; one entry per blob HOMED on this node that
               some node caches: set of caching nodes
DirtyLog       ctp::ipc::mpsc_ring_buffer<DirtyRange>   (THE write log)
blob buffers   from the segment allocator, one contiguous buffer per blob
```

Limits are config: `capacity` (segment bytes), `log_depth`, `max_tags`,
`max_blobs`, `max_name` (longer BLOB names take the task path), `max_fill`,
`evict_sample`, `half_life_ms`.

## Write path

```
PutBlob(tag, blob, off, size, src):
  lock;   entry = find-or-insert(tag, blob)
          grow entry buffer to >= off+size (alloc new, copy, free old)
          entry.pending++;  entry.write_gen++
          entry.size = max(entry.size, off+size)
  unlock
  memcpy(entry.buffer + off, src, size)           // entry seqlock odd/even
  log.Push(DirtyRange{entry slot, off, size})     // waits if the log is full
  log_pushed_seq++
```

The client returns after the push. Exactly the bytes written are logged: a
write that leaves a gap, or touches 4 KB of a 1 MB blob, logs only that
range. Nothing is ever flushed per blob, and holes in a buffer never leave
the cache.

## Background flusher

One continuous task on the chimod (`FlushLog`; yields when the log is
empty) is the ONLY consumer of the log:

```
loop:
  pop up to flush_batch_bytes worth of ranges
  sort by (slot, off); coalesce overlapping/adjacent ranges of one blob
  copy each coalesced range's CURRENT bytes into one staging buffer
  ONE MultiPutBlob per home node, addressed to the cache pool (it routes
  to each blob's home; the home forwards to next_pool_id and invalidates,
  see Coherence)
  on completion: entry.pending -= ranges popped for it;
                 log_flushed_seq = seq of the last popped range
```

"Current bytes" is safe: if a writer overwrote a range after logging it,
the flusher may ship the newer bytes early, and the newer write logged its
own range, which ships again. The chain converges to the last write, and a
range is never sent with bytes older than the log entry that named it.

The flusher copies into staging rather than pointing the task at the blob
buffer, so a buffer can grow or be evicted while a batch is in flight.

Ranges whose writes carried a different `Context` (codec, persistence) go
in separate batches, as `MultiPutBlobTask` already requires.

### Barrier

`log_pushed_seq` / `log_flushed_seq` give a cheap barrier: wait until
`log_flushed_seq >= log_pushed_seq as observed now`. It waits on the
flusher, which is always running; it never flushes by itself. Used by:
- reads that must go to the chain (below);
- `SyncTag`, `DelTag`, `DelBlob`, `FlushData`, H5Fflush/H5Fclose;
- methods that must see the chain: `TagQuery`, `BlobQuery`,
  `GetContainedBlobs`, `GetTagSize`.

### Eviction (capacity)

Anything not dirty is a candidate. A blob is dirty while `pending > 0` (it
has logged ranges the flusher has not landed); a dirty buffer is never
evicted.

Policy: **frecency**. Each entry keeps `hits` and `last_access_ns`, both
bumped on every read and write hit (relaxed atomics, no lock):

```
frecency = hits * 2^(-(now - last_access_ns) / half_life)
```

When an allocation fails (buffer for a write or a fill):
1. sample up to `evict_sample` clean, resident entries starting at a
   rotating cursor in BlobTable (approximate, Redis-style; no global
   ordered structure to keep consistent across processes);
2. free the buffer of the lowest-frecency sample; repeat until the
   allocation fits or `evict_max_rounds` is reached;
3. still no room (everything resident is dirty): wait on the flusher, then
   retry. Never fall back to a task write; a direct task write could reach
   the chain before an older logged write of the same bytes and then be
   overwritten by it.

Evicting a buffer keeps the metadata entry (name, id, size) and clears
`kComplete`. When the BlobTable itself is full, clean entries with no
buffer are dropped the same way (lowest frecency first).

A full log blocks producers in `Push` (`WAIT_FOR_SPACE`) until the flusher
pops.

## Reads

Every resident buffer carries `kComplete` when it holds the WHOLE blob.
`kComplete` is set when:
- the blob was created here under a tag whose `GetOrCreateTag` reported
  `created_` (the tag is new, so no older bytes exist anywhere): holes read
  as zeros, which is what the chain would return;
- the blob was filled from the chain (below).

Read rules:
- `kComplete` and the request is within `size`: memcpy from SHM, bump
  frecency. No task.
- otherwise: task to the cache pool. The chimod waits on the barrier,
  reads from the chain, answers the request, and **populates the cache**:

```
fill(tag, blob):
  gen0 = entry.write_gen                 // bumped by every logged write
  bytes = GetBlob(whole blob) from next_pool_id    (size <= max_fill)
  lock
  if entry.write_gen == gen0:            // no local write raced the fetch
    allocate buffer (may evict); copy bytes; set kComplete
  // else: a newer local write landed; skip the fill (next miss retries)
  unlock
```

  Blobs larger than `max_fill` are served but not cached.
- `GetBlobSize`: `entry.size` when `kComplete`; otherwise barrier + chain,
  and the answer is recorded on the entry.

## Coherence (across nodes)

Caching only needs one thing to stay coherent: when a blob changes, every
other node caching it drops its copy.

**Home.** Each blob has a home: the cache container on
`HashBlobToContainer(tag, blob)`, the owner the core already routes to
(CoreInterposer delegates ScheduleTask to the core). The home keeps a
`Sharers` entry per blob listing the nodes that cache it.

**Register.** A read miss is a task routed to the blob's home anyway. The
home adds the requesting node to the blob's sharers when it serves the
fill. Blobs created by local writes register the writer node the same way,
on their first flush.

**Invalidate.** Flushed ranges go to each blob's home (the flusher batches
per home node). The home forwards the batch down the chain and, once it
lands, sends `Invalidate(blob list)` to every sharer except the writer.
A sharer clears `kComplete`, bumps the entry's `inval_epoch`, and frees the
buffer if nothing is pending for it. A buffer with unflushed local ranges
keeps its bytes for the flusher, but is no longer served. The node is
removed from the sharers set; it re-registers on its next miss.

**Races.**
- Fill vs invalidate: a fill records `inval_epoch` before fetching and
  installs the bytes only if it is unchanged (same check as `write_gen`).
- Reader mid-memcpy vs invalidate: the client re-checks the entry seqlock
  after copying and retries through the task path if it moved.
- Invalidate vs ack: the home acks the flush batch only after the
  invalidations are acked, so a writer's barrier/SyncTag means "every other
  node's cached copy is gone".

**Visibility.** A remote reader sees a write once the writer's flusher has
landed it at the home (continuous, so normally within one batch interval).
For a guaranteed read-after-write across nodes, the writer syncs
(`SyncTag`, H5Fflush/H5Fclose), which waits on its log barrier, and the
log barrier now includes the invalidation acks.

**Tag-wide operations.** `DelTag` and `DelBlob` invalidate the same way.
`GetTagSize`, `GetContainedBlobs`, `TagQuery`, `BlobQuery` barrier the
local log and forward; they read the chain, which every node's flusher is
continuously feeding.

**Failures.** A sharer that is gone cannot ack; the home drops it from all
sharer sets when the runtime reports the node failed (or after an
`inval_timeout_ms`), and does not block on it.

**Capacity.** A Sharers entry is freed when its set empties. If the table
is full, the home invalidates the lowest-frecency entry's sharers to free
one.

## Tags and metadata

The cache holds tag and blob METADATA, not only bytes: a cache that sends
every tag lookup, name resolution or size query to the runtime keeps the
overhead it exists to remove. With `md_only: true` it holds metadata only
(the data cache below, e.g. the write-back cache, holds the bytes).

### Chain position

```
clio-fs / CAE / stream -> indexer (564) -> shm cache (568) -> write-back (567)
                                            -> replication (561) -> cores
```

The indexer sits ABOVE the cache: it reads and writes through it, and
indexing precedes the data transformations below. Clients bind to the
cache's `entry_pool_id` (the indexer) so their tasks pass every module above
the cache, and map the segment for the shared-memory paths. A put absorbed
in shared memory (data mode) bypasses the indexer at put time; the flusher
notifies the indexer when that write lands (index on flush).

### Tag homes

Every tag has a HOME among the cache's containers: hash(tag id) % N, the
same rule blobs use (see Known gaps for the hash-layout caveat). The home
is the authority this cache uses for the tag's metadata:

- **Names.** clio-fs publishes hierarchical names as records
  "$tagid{parent}/leaf" (`UpdateTagNames`, `RenameTag`). The cache routes
  each record to its tag's home -- never to the publishing node, never as
  a broadcast. The home applies it to its own core (durable, as before) and
  keeps it in its tag table.
- **Size and times.** Each node's cache tracks the size and mtime changes
  its own writes make to a tag and reports them to the tag's home with the
  flushed batch; the home folds them in and pushes the new values to the
  nodes caching the tag.
- **Sharers and invalidation.** A node that fetches a tag's metadata
  registers at the home; a rename, a name change, a size/time change or a
  DelTag invalidates (or updates) every registered copy before it is
  acknowledged -- the blob protocol, applied to tags.

### Segment

```
TagTable   key = TagId; value = stored name (parent id + leaf), size,
           atime/mtime/ctime, flags (kCreated, kNameValid, kAttrValid),
           seqlock, inval epoch, sticky_rc
NameIndex  key = hash(absolute or stored name) -> TagId (GetOrCreateTag hits)
```

### Verbs

| Verb | Hit (no task) | Miss |
|---|---|---|
| GetOrCreateTag (existing name) | NameIndex | task to the chain, fill |
| GetTagName(id) | resolve the parent chain through TagTable | fetch the missing ancestor from its home, fill |
| GetTagSize / times | TagTable | fetch from the home, fill |
| GetBlobSize / blob info | BlobTable | fetch from the blob's home, fill |
| TagQuery | -- | to every cache container: each scans the records it is home for, resolves them through the cache, applies the regex |
| UpdateTagNames / RenameTag | -- | records routed to their homes; homes invalidate copies |

Tag creation still asks the authority (the core assigns ids); the NameIndex
only memoizes existing name-to-id bindings, invalidated by rename and
DelTag through the home.

### Restart

The core does not journal published names (a restarted core drops them,
`kResetPublished`, and relies on publishers re-sending). So each home also
persists the records it is home for as small blobs under a reserved tag
(`_shm_tag_names`, blob name = packed tag id, payload = stored name and
stamp) in the chain below, and a restarted home rebuilds its TagTable --
and re-applies the names to its core -- from them at startup. No peer
republish is needed.

## Replaces the client deferred-put pipeline

The client's deferred pipeline (core_client.h, roughly lines 1040-2960:
issues #862, #892, #1007) does in each PROCESS what this cache does once per
NODE: buffer writes, coalesce small ones (64 KiB sieve pages, small-put
batches into MultiPutBlob), serve read-your-writes from pending writes, flow
control on in-flight bytes, and report write errors at sync time. It is
deleted, not kept beside the cache.

| Deferred API (removed) | Replacement |
|---|---|
| `AsyncPutBlobDefer`, `AsyncPutBlobVectoredDefer`, `DeferPutLarge`, sieve pages, small-put batch, staging pool | `AsyncPutBlob` (cache path: SHM buffer + log; FlushLog coalesces and batches) |
| `AsyncGetBlobDefer`, `DeferTryServe` (read-your-writes) | `AsyncGetBlob` (cache path serves local writes from SHM) |
| `AwaitPutsUntilSpace`, `DeferAwaitOldest`, window knobs (flow control) | writers block on a full log / a cache with nothing clean to evict |
| `AwaitPendingPuts`, `DeferAwaitKey`, `FsyncDeferred`, `FlushAll` | barrier: `SyncTag` (or the per-tag barrier) |
| `DeferTakeKeyError`, `DeferPeekKeyError`, `DeferErrorCount` | per-tag sticky error in the cache (below), returned by `SyncTag` |
| `DeferMaxPendingEnd` (size including unflushed writes) | the cache's blob `size` (covers buffered writes) |
| `DeferRegisterWrite`, `DeferKeyHash*`, `DeferReapCompleted*` | none needed; the cache owns completion |

### Write errors

A deferred write's failure used to surface at the next await on its key.
With the cache, the write has already returned to the client when the
flusher's `MultiPutBlob` fails a range. The flusher records the first
nonzero rc per tag in the TagTable entry (`sticky_rc`, plus a node-wide
error count). `SyncTag` and the per-tag barrier return it and clear it,
matching fsync/close reporting EIO for an earlier write. The flusher does
not drop a failed range silently: it retries `flush_retries` times before
recording the error.

### Callers to migrate

| Caller | Today | After |
|---|---|---|
| clio-fs: `filesystem_client.h`, `adapter/libfuse/fuse_cte.cc` | defer registry for writes, read-your-writes, file size, fsync errors | plain `AsyncPutBlob`/`AsyncGetBlob`; `SyncTag` for fsync/close; cache blob size |
| `llm-hooks/lmcache/src/lmcache_store.cc` | `AsyncPutBlobDefer` + `AwaitPendingPuts` | `AsyncPutBlob`; barrier where it awaited |
| `benchmark/ycsb/clio_ycsb_jni.cc`, `benchmark/clio_cte_bench.cc`, `test/integration/coherence/bench_chain.cc` | defer as the async write mode | `AsyncPutBlob` with the cache enabled |
| `wrapper/python/core_bindings.cc` | exposes `AsyncPutBlobDefer`, `AsyncGetBlobDefer`, `AwaitPutsUntilSpace`, `DeferErrorCount` | removed; Python `AsyncPutBlob` gets the cache path; `SyncTag` for errors |
| `test/unit/test_multiput_defer.cc`, `test/unit/test_put_sieve.cc` | test the pipeline | replaced by cache tests covering the same cases (coalescing, read-your-writes, flow control, error at sync) |
| `adapter/vfd/H5FDclio.cc` | comment only (an at-exit crash in the registry) | drop the comment |

### Default compose

Without the cache, these callers fall back to one plain task per write,
which is slower than the deferred pipeline they had. Since the cache is the
replacement, it is ON in the default compose (`clio_default.yaml`) once
phase 2 lands, not off.

## Methods

| Method | Cache path |
|---|---|
| GetOrCreateTag | task, forwarded (records TagId + `created_`) |
| PutBlob, MultiPutBlob | SHM buffer + log |
| GetBlob | SHM if `kComplete`, else to home: barrier + forward + fill + register |
| GetBlobSize | SHM if `kComplete`, else to home: barrier + forward |
| DelBlob, DelTag | barrier, invalidate sharers, forward |
| Invalidate (new) | drop `kComplete` (and the buffer if clean) |
| ReorganizeBlob, GetBlobScore | update entry score, forward |
| Query / contained-blob / tag-size / stat methods | barrier, forward |
| everything else | `ForwardToCore` |

## Compose

```yaml
- mod_name: clio_cte_core_shm_cache_writethru
  pool_name: clio_cte_core_shm_cache_writethru
  pool_query: local
  pool_id: "568.0"
  next_pool_id: "563.0"        # existing cache chimod
  capacity: 256M               # segment bytes (buffers + tables + log)
  log_depth: 65536             # DirtyRange entries
  flush_batch_bytes: 16M       # max payload per MultiPutBlob
  max_tags: 4096
  max_blobs: 65536
  max_name: 128
  max_fill: 16M                # larger blobs are read but not cached
  evict_sample: 16             # clean entries sampled per eviction
  half_life_ms: 10000          # frecency decay
  max_sharer_entries: 65536    # blobs homed here that other nodes cache
  nospace_wait_ms: 250         # a put refused for room waits this long for
                               # earlier ranges to flush, then writes around
  inval_timeout_ms: 30000      # stop waiting on a silent sharer
```

On by default once phase 2 lands (it replaces the client deferred-put
pipeline); VOL/VFD and clio-fs bind to the top pool of the compose.

## Phases

1. Chimod skeleton: clio_mod.yaml, autogen, interposer forwarding, segment
   create/attach, DirtyLog, FlushLog task. Unit test through tasks only.
2. Client cache path (Put/Get/GetBlobSize in SHM), frecency
   eviction, read-miss fill, VOL/VFD
   bound to the cache pool, unit tests for both paths, microbenchmark
   against today.
3. Cross-node coherence: sharer registration,
   per-home flush batches, Invalidate. Multi-node Docker reproducers
   (read after remote write + sync, invalidate racing a fill, sharer
   death).
4. Delete the no-cache fast paths AND the deferred-put pipeline from the
   client; migrate clio-fs, lmcache, YCSB/bench, Python bindings and the
   defer/sieve tests (see the migration table); move gpu_vector host
   readers onto plain tasks; rebench against the deferred pipeline.
5. Docs: `clio_default.yaml`, and a module page under `site/pages/modules/`.

## Implementation notes (as built)

Where the code settled details the sections above left open.

- **Files.** Segment layout and every SHM operation:
  `core/include/clio_cte/core/shm_wtc.h` (+ `shm_wtc_impl.h`), shared by
  the client and the chimod. Client path: `core::Client::ConnectCache`,
  `PutBlob`, `GetBlob`, `TryGetCached`, `GetBlobSize`, `GetOrCreateTag` in
  `core_client.h` (the `Async*` methods are the task path only). Chimod: `core_shm_cache_writethru/src/` (`_runtime.cc`
  lifecycle/tags/sync, `_data.cc` task-path data verbs, `_flush.cc` flusher
  and coherence).
- **Locking.** The blob table is 64 stripes; a key probes only inside its
  stripe, so one stripe lock covers it. Payload memcpys hold a pin, never a
  lock. Eviction takes other stripes with TryLock only. Lock order is
  stripe -> tag table -> allocator.
- **Discovery.** The chimod publishes a `Locator` in the runtime metadata
  segment under its pool id; the segment name is derived from the runtime
  pid and pool id. The CTE core's Create (and a get-or-create of an
  existing core pool, via `Container::FillExistingCreate`) returns
  `CreateParams::cache_keys_`: the keys of the cache whose chain of next
  pools ends at that core. `CLIO_CTE_CLIENT_INIT` binds the process to the
  cache's entry pool and calls `Client::ConnectCache(keys)`; a client bound
  with `Init()` calls `Client::ConnectCache()`, which gets the same keys
  with a lookup-only Create of the canonical core pool.
- **What the cache absorbs.** Client side: `Client::PutBlob` with no
  explicit score and no context, flags limited to `kCtePutDroppable`
  (`View::TryPut` refuses any other flag). Task side: puts with auto score,
  a plain `Context` (`Runtime::PlainContext`; a `preallocate_` sizing hint
  still counts as plain and is dropped at flush) and the same flags. Droppability is a
  per-blob property, so it is stored on the entry and carried to the chain
  in the new `MultiPutDesc::flags_`. Replace, append, explicit scores and
  any codec/replica/persistence context take the task path, which waits for
  the blob's pending ranges, drops the local copy and forwards.
- **Routing.** Data verbs and tag create/delete route submitter-local (the
  segment is per node). Other core methods keep the core's owner routing,
  drain that node's log, and forward.
- **Homes are containers.** A blob's home is
  `BlobHash(tag, blob) % containers` of this pool (one container per node).
  Flush groups per home: the local group goes straight to `next_pool_id` as
  a `MultiPutBlob`; a remote group goes to the home as `kHomeFlush` (the
  same task struct), which forwards it down and invalidates before acking.
- **Who gets invalidated.** Every container in the blob's sharer set except
  the writer, and the writer too when another container had also written
  the blob since (its buffer may lack those bytes). N-1 disjoint writers
  therefore exchange invalidation messages per flush; their writes are not
  blocked (buffers with pending ranges keep their bytes).
- **Known gaps.**
  - **Placement is assumed, not asked (to revisit).** The cache computes a
    blob's home as `BlobHash(tag, blob) % containers` (`Runtime::HomeOf`) and
    addresses it with `PoolQuery::DirectId`: flush groups (`HomeFlush`),
    sharer registration, invalidation fan-out. That duplicates the core's
    placement instead of letting `PoolQuery::Dynamic` decide, so it is only
    right while placement is plain hash-mod-N (no successor failover,
    migration or a different placement policy). Chimods should not assume a
    hashing layout; the intended fix routes these tasks with
    `PoolQuery::Dynamic` (per blob, or batched by asking the chain's routing).
    Tracked in iowarp/clio-core#1372.
  - A process that cannot map the segment (TCP client, remote node) stays
    bound to the core pool unless `CLIO_CTE_POOL=568.0`; writes through the
    core directly bypass invalidation.
  - A `MultiPutBlob` record whose flags conflict with the cached blob's is
    reported as failed rather than forwarded.
  - The HDF5 VFD reaches storage through clio-fs, so it benefits from the
    clio-fs cache (`filesystem/SHM_CACHE_DESIGN.md`), not this one. The VOL
    uses this cache directly.
