# clio-fs shared-memory filesystem — design

Status: accepted requirement; phase 1 in progress (branch `fs-clone-bench-dev`).
Companion to `context-transfer-engine/core_shm_cache_writethru/DESIGN.md`
(the CTE blob cache), whose segment mechanics this reuses.

## Requirement

clio-fs is an **entirely shared-memory filesystem**. No access submits a
task. That covers every operation:

| class | operations |
|---|---|
| data | read, write, append, truncate |
| attributes | stat/getattr, chmod, chown, utimens, xattrs, fsync's view of size |
| namespace | lookup, readdir, create, mkdir, unlink, rmdir, rename, link, symlink |

Each completes in the calling process against the node's clio-fs segment.
Updates propagate **asynchronously, in batches**, and a **strong coherence
protocol** keeps nodes consistent. The point is to batch requests and remove
the per-operation task cost that dominates complex workloads (a kernel
checkout: ~100k creates at ~540 us and ~100k page writes at ~310 us, each a
synchronous task today).

Where the logic lives:

- **clio-fs client** (`filesystem_client`): all filesystem semantics — path
  resolution, permission checks, logical size, appends, page split, holes,
  EOF clamp, rename rules, readdir. ~99% of the code.
- **FUSE adapter** (`fuse_cte.cc`): translation only (FUSE arguments, handles,
  errno, kernel cache options). It calls the client; it duplicates nothing.
- **fs_shm_cache module (server)**: three jobs only —
  1. drain the op log and propagate it in batches (data below the CTE cache,
     metadata to each inode's/directory's home);
  2. capacity: evict clean entries and pages;
  3. coherence: grant/recall ownership and apply remote changes into the
     segment.

The filesystem chimod remains the **home and persistence layer**: it owns the
durable inode records and inode-table blocks, and receives the batches. It is
never on a client's access path.

## Segment

One named segment per node per clio-fs pool, a multi-process allocator,
offsets only, POD records (same mechanics as the CTE cache: create/attach,
`MetadataDirectory` registration, seqlocked open-addressed tables).

```
FsShmRoot      version, ready, node_id, limits, stats, log seqs,
               id-range allocator (see Ids)
InodeTable     key = inode id
               value = kind, logical size, mode, uid, gid, atime/mtime/ctime,
                       nlink, symlink target ref, xattr ref, flags,
                       lease (none/shared/exclusive), lease_epoch,
                       sticky_rc, seqlock, frecency, dirty
DentryTable    key = (parent id, canonical-name hash) + name
               value = child id | NEGATIVE, type, spelling ref, state,
                       seqlock, dirty
DirTable       key = directory id
               value = kComplete (every entry present), entry count,
                       lease, lease_epoch, version
PageTable      key = (inode id, page index); value = 1 MiB buffer,
               pending ranges, write_gen, kComplete, frecency
OpLog          mpsc ring of FsOp (data ranges, attribute changes,
               namespace changes), each stamped with a log seq
```

Locks are per entry (seqlock for readers, a spin bit for writers);
multi-entry operations (rename, link) take entry locks in key order.

## Ids

Inode ids are minted in the client with no task. The segment root holds an
id range leased to this node by the filesystem's id authority; clients take
ids from it with an atomic add. The background module refills the range when
it falls under a low-water mark, so a client never waits for one except when
the range is exhausted faster than the refill (then it waits on a flag in the
segment, not on a task). Ids carry the minting node in the high bits
(the existing `major_` ranges), so they are unique cluster-wide.

## Operations (client)

All of these run in the client against the segment, then push an `FsOp`.

- **lookup/stat/getattr**: DentryTable → InodeTable. A miss under a
  `kComplete` directory is an authoritative ENOENT.
- **readdir**: DirTable `kComplete` → iterate the DentryTable for that parent.
- **read**: clamp to the logical size; per page: `kComplete` page → memcpy;
  page inside the size never written → zeros; otherwise a **fill** (below).
- **write**: per page: find-or-insert, copy, mark the range pending, push
  `kData`. If `off+len > size`, raise the size in the InodeTable and push
  `kSize`.
- **append**: under the inode's lock, `off = size; size += len`, then write.
  Every process on the node sees the same size, so appends never overlap;
  across nodes the exclusive lease on the inode makes it the only appender.
- **truncate**: set the size, drop pages past the new end, push `kTruncate`
  (ordered after earlier `kData` for that inode by log seq).
- **chmod/chown/utimens/setxattr**: update the InodeTable, push `kAttr`.
- **create/mkdir/symlink**: check the dentry (negative or absent under a
  `kComplete` directory), mint an id, insert the inode and the dentry, push
  `kCreate`. A new directory is `kComplete` from birth.
- **unlink/rmdir**: mark the dentry NEGATIVE, drop nlink (free the inode's
  pages when it reaches 0 and no fd holds it), push `kUnlink`. rmdir
  requires the directory `kComplete` and empty.
- **rename/link**: lock both dentries (and both directories) in key order,
  move/insert, push one `kRename`/`kLink` op carrying both sides so it
  propagates atomically.
- **fsync**: wait until the log has flushed every op for that inode (and its
  dentry, for a new file) — a seq comparison in the segment — then return
  the inode's sticky rc. close returns the sticky rc too.

A **fill** (cold miss: data or metadata never seen on this node) is the one
place the client cannot answer alone. It posts a fill request into the
segment and waits on the entry's state flag; the module performs it. No task
is submitted by the client. Fills are not on the steady-state path: a
directory listed once is `kComplete`, files created here are complete from
birth.

## Appends

Visibility rule: a node's view of a file may **lag** (it sees a shorter
file), but nothing it sees is ever provisional. Every byte inside the size
a node reports is final; no offset a reader or appender observes ever moves.

**One node** (the namespace segment is on and this node holds the file):
an append takes the file's append lock in the segment, writes its pages at
the inode's current size (shared-memory page puts), raises the size, and
releases the lock. Exact and immediate; no merge.

**Asynchronous merge** (other nodes may append to the same file; forced on
one node with `CLIO_CFS_APPEND_MERGE=1`):

1. *Stage.* The client reserves `[off, off+len)` in the file's current
   **append generation** in the segment (per-file `{gen, bytes[2],
   inflight[2]}`) and copies the bytes into the blob `app.<node>.<gen>` of
   the file's tag with a plain `PutBlob` (the CTE shm cache: no task). The
   offset comes from the segment, not from the cache entry, which can be
   dropped under capacity pressure. It then pushes `kOpAppend` onto the op
   log. `write(O_APPEND)` returns here.
2. *Seal.* The module's drain, on `kOpAppend`, switches the file to
   generation `gen+1` (new appends go there), waits for `inflight[gen]`
   to reach zero, and takes `L = bytes[gen]`.
3. *IdentifyRange.* One `IdentifyRange(file, node, gen, L)` per sealed
   generation, sent with `PoolQuery::ManyToOne(owner, batch_key = file id)`
   so every node's request for the file in a window is one collective. At
   the neighborhood leader `AggregateIn` appends `(node, gen, L)` to the
   aggregate's member list in arrival order. The aggregate runs on the
   file's owner, which reserves `[end, end + total)` **without publishing a
   size** and returns the map `node -> offset` (base + prefix); it is
   broadcast to every member and each origin takes its own entry. A node
   has at most one generation of a file in flight, so (node, gen) is unique.
4. *Write.* The origin reads `app.<node>.<gen>` and writes it to the file's
   page blobs at its offset (bytes move once, origin to page homes).
5. *CommitRange.* The origin tells the owner the range is written. The
   owner publishes the size as the end of the committed prefix: a range
   reserved but not committed blocks the size, so readers never see bytes
   that are not written. The origin deletes `app.<node>.<gen>`.
6. *Dead origin.* A reserved range not committed within
   `append_commit_timeout` (or whose origin the cluster declares dead) is
   committed as a hole (zeros): its appends were never fsynced.

fsync, truncate and unlink of a file with staged appends first wait for its
generations to merge. A node's appends keep their order (one generation in
flight per file); across nodes the owner orders batches by arrival.

The stream module's deferred-append pipeline (staging tag, Sequence/Plan)
is replaced by this; the module's remaining job (file sizes) moves to the
filesystem module's inode records in a later step.

## Background propagation (module)

A single flusher drains the OpLog continuously:

1. Pop a batch, group by destination home.
2. Coalesce: data ranges per page (exact ranges, merged); attributes per
   inode (last value wins; size = max of growths unless a later truncate);
   namespace ops kept in log order per directory (a create followed by its
   unlink inside one batch cancels).
3. Send one batch task per home: data as `MultiPutBlob` to the pool below
   the CTE cache; metadata as one filesystem `ApplyBatch` (inode records and
   inode-table rows) per home.
4. On ack, advance `log_flushed_seq`, clear `dirty` bits, record a failure
   as the inode's `sticky_rc`.

Nothing is flushed per operation. Writers block only on a full log or a
segment with nothing clean to evict.

## Capacity (module)

Evict clean entries by frecency when the segment passes its high-water mark:
pages first, then inodes and dentries of closed files, then whole
`kComplete` directories (which clears `kComplete`). Dirty entries are never
evicted; the flusher makes them clean. Entries holding an exclusive lease
are evicted only after they are clean, and the lease is released with them.

## Coherence (strong)

Single-writer/multi-reader per object, by **lease**, on the existing home
mapping (inode records and inode-table blocks each have a home container):

- **Shared lease**: the node may serve reads of the object from its segment.
- **Exclusive lease**: the node may also modify it locally.
- A node's first touch of an object it has no lease for is a fill, which
  acquires the lease as part of the fill. Objects created locally are born
  with an exclusive lease (the dentry insert takes the directory block's
  exclusive lease).
- **Recall**: when another node needs a conflicting lease, the home asks the
  holder to flush that object's pending ops and downgrade. The holder's
  module performs the flush and applies the downgrade into the segment;
  clients on the holder see the lease bit change and retry their next
  modification as a fill. Readers on other nodes never see a stale value:
  the home grants a new lease only after the old holder's flush landed.
- **Directory leases** cover a directory's dentries; namespace ops across
  two directories (rename) need both, taken in id order by the module on the
  client's behalf.
- **Single node**: this node holds every lease from birth; nothing is ever
  recalled, so no operation ever waits.
- **Failure**: a lease held by a node the runtime reports dead is revoked by
  the home; that node's unflushed ops are lost exactly as a crashed local
  page cache's would be (fsync is the durability point).

## Crash recovery

The segment belongs to the module and outlives every client. A client may be
SIGKILLed at any instruction, including while it holds the segment's lock.
Nothing may hang because of that, no other client's change may be lost, and
clients never repair anything: the module does, from NsFastTick
(`View::RepairDead`, every 500 ms), or inline when one of its own threads is
stuck on a dead holder.

- **Lock (`NsLock`).** The writer word carries the writer's pid. Readers
  hold one bit each, the bit of their registered thread (`NsThread`), so a
  dead reader's hold is exactly its own bit; a reader *count* could not be
  attributed. A dead reader's bit is cleared and its registration freed.
- **Dead writer.** Its table edits may be half-done, and they were never
  logged (a record is pushed last, and only a pushed record is visible to
  the drain). The module freezes the lock out of its hands (`kFrozen`):
  clients wait; the module's own calls skip the tables (`ModuleGuard`).
  The drain keeps applying what WAS logged, so no acknowledged change is
  lost. Once the log is empty the module rebuilds the tables empty (root
  incomplete), so every name refills from the module, and unfreezes.
  Appends, the id block and the log are kept.
- **Appends.** A file's append lock holds its holder's pid (freed when it
  dies). An append in flight is recorded in the appending thread's
  `NsThread` (per-generation count), not in a shared count, so a dead
  appender's record is dropped exactly and its generation can merge.
- **Size table.** Each stripe lock holds its holder's pid. A dead holder's
  stripe is released after its dirty count is recounted (the held section
  edits one slot, which stands as written).

## Relationship to the CTE cache

clio-fs data lives only in this segment's PageTable. The module's flusher
and fills address the pool **below** the CTE blob cache, so a clio-fs byte
is never held twice by clio-fs's own traffic. A non-clio-fs reader of a
clio-fs page through the CTE cache may hold a second read copy; the page's
home invalidates it on the next landed write.

## Replaces

| Today | After |
|---|---|
| `shm_fs_cache.h` read-only mirror | InodeTable/DentryTable/DirTable (read-write) |
| per-op `Open`/`Create`/`Mkdir`/`Unlink`/`Rename` tasks | client ops + `kCreate`/`kUnlink`/`kRename` log entries |
| native `Write` task; FUSE `WriteDirect` via the CTE cache | PageTable writes |
| FUSE hiwater map / `AdvanceSize` / append reservation | InodeTable logical size |
| FUSE sieve-create and negative-lookup caches | DentryTable + `kComplete` |
| fsync error latching in the adapter | per-inode sticky rc |

## Phases

1. **Single node.** Segment, tables, OpLog, id ranges, every client
   operation, flusher (`ApplyBatch` to the filesystem chimod, `MultiPutBlob`
   below the CTE cache), capacity. Native API and FUSE adapter both on the
   client; the adapter's own caches deleted. Gate: clio-fs unit suites,
   persistence test, kernel checkout vs NTFS (target 1.5–2x).
2. **Coherence.** Leases, recall, remote apply into the segment, fills
   across nodes. Gate: multi-node Docker reproducer for concurrent create,
   rename, append and read-after-fsync.
3. **Removal.** Delete the mirror and every per-op task path from the client
   and adapter; benchmarks (checkout, small-write, large-write).
