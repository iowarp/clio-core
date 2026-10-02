#!/usr/bin/env python3
"""Copy a large random file into a clio-fs mount N times and check every
copy byte for byte (issue #1116: cp exits 0, size is right, content wrong).

Run on a node (the suite's `sh` op). Prints one JSON line:
  {"copies": N, "bad": [{"copy": i, "pages": [...]}...]}
where each bad page reports which 128 KiB sub-blocks differ and, for each,
the offset in the SOURCE whose bytes it actually holds (or null).
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess

MiB = 1 << 20
SUB = 128 * 1024  # cp's write size, and the CTE client's batch chunk


def make_source(path, size_mib):
  """Write size_mib MiB of random bytes to path (once)."""
  if os.path.exists(path) and os.path.getsize(path) == size_mib * MiB:
    return
  with open(path, 'wb') as f:
    for _ in range(size_mib):
      f.write(os.urandom(MiB))


def locate(src_index, piece):
  """Offset in the source whose 128 KiB sub-block equals piece, or None."""
  return src_index.get(hashlib.sha1(piece).digest())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--src', required=True)
  ap.add_argument('--dst-dir', required=True)
  ap.add_argument('--size-mib', type=int, default=1024)
  ap.add_argument('--copies', type=int, default=10)
  ap.add_argument('--tool', default='cp', choices=['cp', 'python'])
  ap.add_argument('--keep', action='store_true',
                  help='keep every copy (fills the store with copies x size)')
  args = ap.parse_args()
  make_source(args.src, args.size_mib)
  want = hashlib.sha256()
  src_pages = []
  src_index = {}
  with open(args.src, 'rb') as f:
    off = 0
    while True:
      page = f.read(MiB)
      if not page:
        break
      want.update(page)
      src_pages.append(hashlib.sha1(page).digest())
      for s in range(0, len(page), SUB):
        src_index[hashlib.sha1(page[s:s + SUB]).digest()] = off + s
      off += len(page)
  bad = []
  for i in range(args.copies):
    dst = os.path.join(args.dst_dir, f'r{i}.bin')
    if args.tool == 'cp':
      rc = subprocess.call(['cp', args.src, dst])
    else:
      shutil.copyfile(args.src, dst)
      rc = 0
    pages = []
    size = os.path.getsize(dst)
    with open(dst, 'rb') as f, open(args.src, 'rb') as s:
      for p in range(len(src_pages)):
        page = f.read(MiB)
        if hashlib.sha1(page).digest() == src_pages[p]:
          continue
        s.seek(p * MiB)
        ref = s.read(MiB)
        subs = []
        for o in range(0, MiB, SUB):
          got, exp = page[o:o + SUB], ref[o:o + SUB]
          if got != exp:
            zero = got.count(0) == len(got)
            subs.append({'sub_off': o, 'zero': zero,
                         'holds_src_off': locate(src_index, got)})
        pages.append({'page': p, 'zero_subs': sum(1 for x in subs if x['zero']),
                      'subs': subs[:2]})
    if pages or rc != 0 or size != len(src_pages) * MiB:
      bad.append({'copy': i, 'rc': rc, 'size': size,
                  'bad_pages': len(pages),
                  'all_zero_pages': sum(1 for x in pages
                                        if x['zero_subs'] == MiB // SUB),
                  'first_bad': pages[0]['page'] if pages else None,
                  'last_bad': pages[-1]['page'] if pages else None,
                  'sample': pages[:2]})
    # Free the copy before the next one: the test checks copies, not how
    # many fit (the store is ~10 GB on the safe profile).
    if not args.keep:
      os.unlink(dst)
  print(json.dumps({'copies': args.copies, 'bad': bad}))


if __name__ == '__main__':
  main()
