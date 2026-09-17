#!/usr/bin/env python3
"""Where a CTE core placed each blob, from its metadata write-ahead log.

  decode_wal.py <metadata_log_path>            every record, in file order
  decode_wal.py --summary <metadata_log_path>  blobs per storage pool

Reads <path>.blob.0, .blob.1, ... (one shard per worker; transaction_log.h).
"""
import os
import struct
import sys
from collections import defaultdict

NAMES = {0: 'CreateNewBlob', 1: 'ExtendBlob', 2: 'ClearBlob', 3: 'DelBlob',
         4: 'CreateTag', 5: 'DelTag', 6: 'SetBlobTransform', 7: 'ExtendReplica'}


def records(base):
    i = 0
    while os.path.exists(f'{base}.blob.{i}'):
        data = open(f'{base}.blob.{i}', 'rb').read()
        p = 0
        while p + 5 <= len(data):
            t, n = struct.unpack_from('<BI', data, p)
            p += 5
            pay = data[p:p + n]
            p += n
            if len(pay) < 12:
                break
            _, _, sl = struct.unpack_from('<III', pay, 0)
            name = pay[12:12 + sl].decode(errors='replace')
            q = 12 + sl
            blocks = []
            if t == 1:
                (nb,) = struct.unpack_from('<I', pay, q)
                q += 4
                # u32 major, u32 minor, PoolQuery, u64 offset, u64 size
                size = (len(pay) - q) // nb if nb else 0
                for k in range(nb):
                    base_k = q + k * size
                    major, minor = struct.unpack_from('<II', pay, base_k)
                    off, sz = struct.unpack_from('<QQ', pay, base_k + size - 16)
                    blocks.append((f'{major}.{minor}', off, sz))
            yield i, NAMES.get(t, str(t)), name, blocks
        i += 1


def main():
    args = sys.argv[1:]
    summary = bool(args) and args[0] == '--summary'
    if summary:
        args = args[1:]
    if len(args) != 1:
        sys.exit(__doc__)
    base = args[0]
    if not summary:
        for shard, kind, name, blocks in records(base):
            bl = ' '.join(f'{p}@{o // 1048576}M+{s // 1048576}M' for p, o, s in blocks)
            print(f'shard {shard}  {kind:14s} {name:32s} {bl}')
        return
    blobs = set()
    per_pool = defaultdict(set)
    for _, kind, name, blocks in records(base):
        if kind == 'CreateNewBlob':
            blobs.add(name)
        for pool, _, _ in blocks:
            per_pool[pool].add(name)
    pools = ', '.join(f'{p}: {len(v)}' for p, v in sorted(per_pool.items()))
    print(f'{len(blobs)} blobs created; blobs placed per pool: {pools or "none"}')


if __name__ == '__main__':
    main()
