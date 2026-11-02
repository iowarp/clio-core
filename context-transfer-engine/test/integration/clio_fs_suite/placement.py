"""Where the CTE puts a clio-fs page: the Python twin of BlobHash
(core/include/clio_cte/core/blob_placement.h) and PageName
(filesystem/include/clio_cte/filesystem/filesystem_tasks.h).

Fault tests use it to aim a fault at the node that owns a page, or at the
node holding the page's remote copy (the replication chimod mirrors a blob to
the next `remote_copies` containers after its owner), instead of hoping a
random kill hits something interesting.

The CTE side hashes with libstdc++'s std::hash: identity for uint32 and
_Hash_bytes (64-bit MurmurHash2 with seed 0xc70f6907) for std::string.
.deletelater/blobhash_check.cc checks this port against the real thing.
"""

MiB = 1 << 20
M64 = (1 << 64) - 1
M32 = (1 << 32) - 1


def _hash_bytes(data, seed=0xc70f6907):
  """libstdc++ _Hash_bytes on a 64-bit target (MurmurHash64A)."""
  m = 0xc6a4a7935bd1e995
  r = 47
  ln = len(data)
  h = (seed ^ ((ln * m) & M64)) & M64
  nblocks = ln // 8
  for k in range(nblocks):
    w = int.from_bytes(data[8 * k:8 * k + 8], 'little')
    w = (w * m) & M64
    w ^= w >> r
    w = (w * m) & M64
    h ^= w
    h = (h * m) & M64
  tail = data[8 * nblocks:]
  if tail:
    h ^= int.from_bytes(tail, 'little')
    h = (h * m) & M64
  h ^= h >> r
  h = (h * m) & M64
  h ^= h >> r
  return h


def blob_hash(major, minor, name):
  """BlobHash(TagId{major, minor}, name) as a uint32."""
  h = major & M32
  h ^= ((minor & M32) + 0x9e3779b9 + ((h << 6) & M32) + (h >> 2)) & M32
  h &= M32
  s = _hash_bytes(name.encode()) & M32
  h ^= (s + 0x9e3779b9 + ((h << 6) & M32) + (h >> 2)) & M32
  return h & M32


def tag_of_ino(ino):
  """clio-fs reports a file's tag id as its inode: major << 32 | minor."""
  return ino >> 32, ino & M32


def page_owner(ino, page, n):
  """Container (node index) that owns page `page` of the file with inode
  `ino` in an n-node deployment."""
  ma, mi = tag_of_ino(ino)
  return blob_hash(ma, mi, str(page)) % n


def copy_holders(ino, page, n, remote_copies=1):
  """Nodes holding the page's remote copies, in failover order."""
  o = page_owner(ino, page, n)
  return [(o + i) % n for i in range(1, remote_copies + 1) if i < n]


def pages_by_owner(ino, size, n):
  """{owner node: [page indexes]} for a file of `size` bytes."""
  out = {}
  for p in range((size + MiB - 1) // MiB):
    out.setdefault(page_owner(ino, p, n), []).append(p)
  return out
