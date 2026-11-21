# Metadata Recovery After a Job — Design (draft for review)

Status: proposal, not implemented. Branch `persistence-hardening`.

## Problem

At an HPC site the CTE runtime lives only as long as the job. The data it
made durable survives on the persistent tier: bdev target files and metadata
logs on a shared parallel filesystem (Lustre/GPFS). After the job, nothing
can read it. A later job, possibly with a different node count, or a
single-node viewer on a login node, should be able to ingest the old job's
metadata. Users can then read their blobs and their CLIO-FS namespace without
re-running the simulation.

Requirements, as decided for this branch:

- The old job's state is on a shared filesystem.
- The ingesting runtime may have **any node count**, including 1.
- Data written by the cache tier must already be durable when the job ends.
  This is the job of the unplug/stop flush (Part 1 of this branch): a
  graceful `clio_run stop` flushes the write-back cache synchronously.

## What exists today (facts the design builds on)

| Piece | On disk | Keyed by |
|---|---|---|
| CTE core metadata | snapshot `<metadata_log_path>`, WAL shards `.blob.<w>` / `.tag.<w>` (format `WAL2`) | **nothing**: same name on every node |
| Bdev data | `<device.path>_node<N>` | old node index |
| Blob → bytes | `BlobBlock{bdev PoolId (512+dev, 1+node), PoolQuery bytes, offset, size}` | runtime-derived PoolId |
| Blob owner | `BlobHash(tag, name) % num_containers` | old container count |
| Tag home / name binding | `TagHomeHash(id)`, `NameHash(name)` | old container count |
| CLIO-FS inode | `~i` xattr on the inode's tag; inode id = tag id; home encoded in id (`major & 0xFFFF`) | old container index |
| CLIO-FS directories | `.dirs.<container>` journal (`FsMetaLog`) | old container id |
| CLIO-FS sizes | `<path>.stream.<container>` log | old container id |
| Replication shadows | stored as the receiver's own blobs; `shadow_` **not logged** | old container index |
| Pool registry | `<conf_dir>/wal/pools.<node>.bin` | node id |

Restart today means "same cluster, re-init in place." Restart does no
node-count check. Blobs are not re-homed; only tag names are
(`BindRestoredNames`). Nothing moves WAL or snapshot state between
containers. The core implements none of `Recover`, `Migrate` or `Expand`.

Offline-usable building blocks already exist:
- `TransactionLog::Load()`, header-only, with static deserializers.
- `RecordLog` / `FsMetaLog::Replay`.
- `DecodeInodeRecord`.
- The stream-log decoder.

The snapshot parser exists only inline in `Runtime::RestoreMetadataFromLog`.

## Defects found along the way (fix regardless of recovery)

1. **Metadata log names collide on shared storage.** Every core container
   writes `${CLIO_STORAGE_ROOT}/cte_metadata_log{,.blob.N,.tag.N}`. The
   default root is `~/.clio`, and on clusters `$HOME` is usually NFS, so
   nodes overwrite each other's WAL and snapshot. The FS, stream and
   replication logs already append the container id; the core must too.
2. **`shadow_` is not persisted.** After a restart a replication shadow copy
   looks like a primary blob. A merge across containers cannot tell copies
   from originals.
3. **Block references use runtime-derived bdev PoolIds** (`(512+dev, 1+node)`)
   with no record of which file each id meant. If the node count or device
   list changes, old references point at the wrong or a missing target, and
   restore keeps them silently (`ApplyWalExtendBlob` treats an unregistered id
   as a valid non-volatile target).

## Proposed design

The approach is offline re-partitioning into the existing restart format.
The new runtime keeps restoring the way it does today, from a snapshot and
WAL per container. The offline tool rewrites the old job's state into
exactly those files for the new container count.

```
old job (N containers)                         new job (M containers)
 manifest + snapshots + WALs  ──►  clio_run recover  ──►  per-container snapshots
 .dirs.* / .stream.* journals        (offline merge,       .dirs.* / .stream.*
 bdev files (untouched)               re-hash to M)         target attach list
                                                            → clio_run start (restart)
```

### Step 0 — Self-describing, collision-free on-disk state

- Core metadata files get the container id: `<metadata_log_path>.c<id>`
  (snapshot) and `.c<id>.blob.<w>` / `.c<id>.tag.<w>`. Restore of a
  single-container pool still accepts the old bare name, so existing
  single-node state keeps working.
- Each core container writes a **manifest**, `<metadata_log_path>.c<id>.manifest`,
  as YAML at create time and at each snapshot. It records:
  - format version, pool id/name, `num_containers`, this container id, node
    id, hostname;
  - the hash function versions (BlobHash, TagHomeHash, NameHash, FsPathHash);
  - **targets**: bdev PoolId → {path, bdev type, capacity, durability}. This
    one table turns block references into files;
  - the companion files: `.dirs.<c>`, `.stream.<c>`, the handoff log, and the
    `.alloc_log` of each file target;
  - a `sealed: true|false` flag plus the last WAL seq. It is set by a graceful
    stop after the final snapshot, so recovery knows whether the WAL tail must
    be replayed.
- Persist `shadow_` (new WAL record `kSetBlobShadow`, plus a snapshot flag
  bit).

### Step 1 — One offline reader library: `MetadataImage`

Move snapshot parsing out of `RestoreMetadataFromLog` into a
runtime-independent `MetadataImage`:
- `Load(snapshot, wal_shards)` returns in-memory maps of tags, blobs (blocks,
  replicas, shadow flag, times), xattrs and name binds.
- `Save(snapshot_path)` writes them back.

The runtime restore path then becomes `MetadataImage::Load` followed by
installing the maps, so one parser serves both. This also gives a debug
tool: `clio_run meta dump <manifest>`.

### Step 2 — `clio_run recover`

```
clio_run recover --from <old manifests glob> --containers M \
                 --to <new metadata_log_path> [--read-only] [--fs]
```

1. **Load** every old container's image: snapshot plus WAL replay up to the
   last seq. An unsealed container (crashed job) replays its full WAL tail.
   This is the same rule as today's restart.
2. **Merge** into one global image. Primaries win over shadows. For a blob
   that appears twice (an owner and a stand-in after a failover), the higher
   WAL seq or `wall_ns` wins.
3. **Re-map targets.** Each old bdev PoolId resolves through the manifests to
   a file. The new job gets a target list for those files with new PoolIds.
   Every block reference is rewritten from old to new PoolId. The core
   already has an attach mode (`existing_pool_id_`).
4. **Re-partition** by the new count M:
   - blob owner = `BlobHash % M`;
   - tag home = `TagHomeHash`;
   - name binding = `NameHash % M`.

   Write one snapshot per new container (`.c<k>`). The new job starts with
   `is_restart_` and restores as today; no runtime change is needed beyond
   Step 0's naming.
5. **Imported targets are read-only by default.** New writes go to the new
   job's own targets, and changing an imported blob re-allocates it there.
   This avoids the dangerous part: reserving the old allocations inside
   reused bdev files. `--writable` instead replays each file target's
   `.alloc_log` (kFile already rebuilds from it) and reserves restored block
   ranges (`ReserveRestoredBlockSpace`).

### Step 3 — CLIO-FS namespace

The hard part is that ownership is baked into ids and journal names:
- inode homes are encoded in inode ids (`major & 0xFFFF`);
- directory blocks live in their old home's `.dirs.<c>` journal;
- sizes live in `.stream.<c>`.

The recovery side:
- **Inode home lookup becomes `encoded_home % M`** (one change in
  `FsIdHome`). Ids stay stable, so tags, blobs and the `~i` xattrs need no
  rewriting.
- **Directory journals are re-partitioned offline.** Replay all old
  `.dirs.*` into a block map, recompute each block's home
  (`BlobHash(dir, "d.<k>") % M`), and write new `.dirs.<k>` journals. Each
  block becomes one `kBlockImage` record.
- **Stream size logs** are re-keyed by the inode's new home into new
  `.stream.<k>` files. A file whose stream record is missing takes its size
  from its `~i` record.
- **Verification pass:** walk from the root (`0x3FFFFFFF.1`) and check that
  every row's child inode resolves and every page blob of a file below its
  size exists. Report orphans instead of failing.

### Single-node viewer

M = 1 is a special case of the same tool: every record lands in container 0.
On a login node, `clio_run recover --containers 1` followed by `clio_run
start` and a CLIO-FS FUSE mount gives a browsable, read-only view of the old
job.

## Phasing

| Phase | Deliverable | Risk |
|---|---|---|
| P0 | Container-id file names + manifest + persisted `shadow_` (fixes the collision bug for same-count restarts too) | low; format change, keep reading the old name |
| P1 | `MetadataImage` library; restore switched to it; `clio_run meta dump` | medium; touches the restore path |
| P2 | `clio_run recover` for the CTE core, same and different M, read-only import | medium |
| P3 | CLIO-FS re-homing (dirs, streams, inode home `% M`) + verification walk | high; namespace invariants |
| P4 | `--writable` import (alloc-log reservation) | medium |

## Tests

- **Unit:** `MetadataImage` round-trip (save → load equal); merge rules
  (shadow vs primary, duplicate seq); re-partition (every blob in exactly one
  new container, owner = `BlobHash % M`).
- **Integration** (real daemons; multi-container on one host via separate
  ports and storage roots):
  - job A with N=2 writes blobs and a CLIO-FS tree, then `clio_run stop`
    (cache unplug flush);
  - `recover --containers 1` and `--containers 3`;
  - start, read every blob and file, and compare checksums;
  - crash case: kill -9 job A, recover from an unsealed WAL, and confirm
    every acknowledged fsync'd write survives.
- **Regression:** two nodes sharing one `CLIO_STORAGE_ROOT` restart without
  cross-contaminated metadata (the P0 bug).

## Open questions for review

1. Is read-only import the right default? It keeps recovery from ever
   corrupting the old job's files, at the cost of rewriting blobs a user
   modifies after recovery.
2. For a crashed (unsealed) job, should recovery replay the WAL tail
   (consistent with restart), or refuse without `--allow-unsealed`?
3. Offline merge memory: the tool holds the whole image. Very large
   namespaces (1e8+ blobs) may need a streaming, per-output-container pass.
   Is that scale in scope for the first cut?
