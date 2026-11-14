# clio-fs shared-memory cache — design

Status: proposal (branch `cte-shm-cache`). Companion to
`context-transfer-engine/shm_writethru_cache/DESIGN.md` (the CTE cache);
read that first. This doc only covers what differs for a filesystem.

## Why clio-fs needs its own

The CTE cache speaks blobs. A filesystem needs things a blob cache cannot
answer:

- **Logical size.** A file's size is not its tag's physical size (truncate
  grow, sparse writes, appends). Short reads at EOF and zeros in holes
  depend on it.
- **Namespace.** Path lookup, negative lookup (ENOENT), readdir, attributes
  (mode, uid, gid, times, nlink, symlink targets, xattrs).
- **File-range writes.** POSIX writes are `(inode, offset, len)`; clio-fs
  splits them into 1 MiB page blobs. Doing that split in the client per
  write is cost the background flusher can absorb.

Today clio-fs covers this with three separate mechanisms:
1. `shm_fs_cache.h`: a read-only mirror of per-path records, published by
   the filesystem chimod into the metadata segment;
2. a zero-IPC page read built on the CTE client's `TryReadBlobShm` (being
   deleted, see the CTE doc);
3. the client deferred-put registry for writes (being deleted).

The fs cache replaces all three with one write-through SHM cache of the same
design as the CTE cache.

## Shape

A new chimod, `fs_shm_cache`, in front of the filesystem chimod; for data
it sits beside the CTE cache and flushes to the pool below it (see
Relationship to the CTE cache). One named SHM segment per node, a
multi-process allocator, offsets only, POD records — identical mechanics to
the CTE cache.

```
FsShmRoot      version, ready, limits, log seqs, locks
InodeTable     key = inode id; value = kind, logical size, mode, uid, gid,
               times, nlink, flags, seqlock, frecency, sticky_rc
DentryTable    key = (parent inode, name hash) + name; value = child inode
               or NEGATIVE; per-directory kComplete
PageTable      key = (inode, page index); value = 1 MiB contiguous buffer,
               pending, write_gen, inval_epoch, kComplete, frecency
OpLog          ctp::ipc::mpsc_ring_buffer<FsOp>   (data ranges AND attr ops)
```

## Writes (data)

```
Write(inode, off, len, src):
  for each 1 MiB page the range touches:
    page entry find-or-insert; buffer alloc (may evict clean pages)
    memcpy the slice; pending++; write_gen++
    log.Push(FsOp{kData, inode, page, page_off, slice_len})
  if off+len > logical size: size = off+len (in SHM); log kSize
```

Same rules as the CTE cache: exact ranges only, nothing flushed per page,
writers block on a full log or a cache with nothing clean.

## Writes (attributes)

`truncate`, `chmod`, `chown`, `utimens`, `setxattr`, and size growth from
writes update the InodeTable entry in SHM and push an `FsOp` onto the same
log. The flusher coalesces per inode (last value wins for each attribute;
size takes the max of growths, or the latest explicit truncate) and sends
them in one `ShardOp`/`AdvanceSize` batch to each inode's home.

Truncate-shrink also drops cached pages past the new EOF and logs the cut so
the flusher orders it after any earlier logged data ranges for that inode.

## Namespace mutations: not cached

`create`, `mkdir`, `unlink`, `rmdir`, `rename`, `link`, `symlink` stay tasks
to the filesystem chimod, which resolves and makes them durable as today.
Same reasoning as tag creation in the CTE cache: exclusivity and atomic
rename across nodes cannot be decided from one node's SHM. When the task
returns, the chimod writes the result into the DentryTable/InodeTable (so
the next lookup is local).

The existing O_EXCL sieve-create (fuse_cte.cc) is the one exception and
keeps its current rules (the kernel's negative lookup makes it exclusive on
that node); it inserts its minted inode and dentry into the SHM tables.

## Reads

- `lookup` / `stat` / `getattr`: DentryTable + InodeTable from SHM. A miss
  under a `kComplete` directory is an authoritative ENOENT (today's
  negative-lookup fast path). Otherwise a task, and the answer is cached.
- `readdir`: served from SHM for a `kComplete` directory, else task (and
  the listing fills the DentryTable).
- `read`: clamp against the SHM logical size (short read at EOF), then per
  page: `kComplete` page → memcpy; page known absent (inside size, never
  written, inode created here) → zeros; else task, fill the page.

## Coherence

clio-fs already has the mechanism: the home of each inode-table block keeps
the set of nodes caching it and pushes row deltas to them. The fs cache puts
the receiving side of that push into SHM: a delta updates (or invalidates)
the DentryTable/InodeTable entries directly, so clients see it with no task.

Pages use the CTE cache's protocol: the page's home (CTE per-page hash)
tracks sharers, and a landed write range invalidates the other nodes' copy.

Attribute changes (size, mode, times) ride the inode home's existing delta
push to the nodes caching that inode.

## Relationship to the CTE cache

The fs cache owns clio-fs file data (the pages above). clio-fs **skips the
CTE cache**: the fs cache's flusher, and every clio-fs task, address the
next pool BELOW the CTE cache in the chain (configured as the fs cache's
`next_pool_id`), not the CTE cache itself.

Consequences:
- No ordering between the two caches' logs is needed: clio-fs bytes only
  ever travel through the fs cache's log.
- A byte may be held in SHM twice in one corner (a non-clio-fs client
  reading a clio-fs page through the CTE cache fills a second copy there).
  Accepted; it is a read copy, invalidated by the same page-home protocol.
- clio-fs reads that miss go below the CTE cache too, so a page never sits
  in both caches because of clio-fs's own traffic.

## Shared code

Both caches need the same machinery. Factor it once, as a header library in
`context-transfer-engine/shm_cache_common/`, used by both chimods:

- segment create/attach + directory registration;
- open-addressed seqlocked SHM table template;
- the MPSC op log and flusher loop skeleton (pop, sort, coalesce, batch);
- buffer allocation with frecency eviction of clean entries;
- sharer tracking + invalidate protocol;
- barrier (pushed/flushed seqs) and per-key sticky rc.

Each chimod supplies only its key types, op types, coalescing rule and the
batch task it sends.

## Replaces

| Today | After |
|---|---|
| `shm_fs_cache.h` mirror (read-only, runtime-published) | InodeTable/DentryTable (read-write, same SHM) |
| zero-IPC page read via `TryReadBlobShm` | PageTable reads |
| deferred registry in `filesystem_client.h` / `fuse_cte.cc` | OpLog + flusher |
| client-side size bookkeeping (`DeferMaxPendingEnd`) | InodeTable logical size |
| fsync errors (`DeferTakeKeyError`) | per-inode sticky rc, returned by fsync/close |

## Phases (after CTE cache phases 1-2)

1. Extract `shm_cache_common` from the CTE cache.
2. fs_shm_cache chimod: segment, tables, OpLog, flusher for data and
   attributes; namespace results written in by the chimod.
3. clio-fs client and FUSE adapter on the cache path; delete the mirror,
   the zero-IPC read and the deferred-registry use.
4. Coherence: SHM receiving side of the block delta push; page invalidation.
5. Benchmarks against today's clio-fs (checkout, small-write, large-write).

## Open questions

- **Concurrent fs branches.** The inode-table work (`fs-inode-table`) and
  large-write work (`fs-large-write-perf`) touch the same files. This should
  start from dev after they merge, or rebase onto them.
- **Hard links and symlinks** today demote a directory's mirror record to
  no-fast-path. The DentryTable can model them (a dentry is just a name to
  inode mapping), but the first cut can keep that demotion.
