"""Print the live bytes of each file-tier bdev from its allocation log.

Usage: python3 alloc_log_usage.py <data dir>
Replays every *.alloc_log (40-byte AllocLogRecord: u32 type, u32 group,
u32 block_type, u32 reserved, u64 offset, u64 size, u64 row; type 1 =
alloc, 2 = free) and prints "<tier file> <live MiB> <allocs> <frees>".
"""
import glob
import os
import struct
import sys

REC = struct.Struct('<IIIIQQQ')


def live_bytes(path, group=None):
  """Replay one allocation log; return (live bytes, allocs, frees).

  Args:
    path: the log.
    group: count only this allocator group (None = all).
  """
  live, na, nf = {}, 0, 0
  with open(path, 'rb') as f:
    data = f.read()
  for off in range(0, len(data) - len(data) % REC.size, REC.size):
    typ, grp, _, _, boff, size, _ = REC.unpack_from(data, off)
    if group is not None and grp != group:
      continue
    if typ == 1:
      live[(grp, boff)] = size
      na += 1
    elif typ == 2:
      live.pop((grp, boff), None)
      nf += 1
  return sum(live.values()), na, nf


for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.alloc_log'))):
  b, na, nf = live_bytes(p)
  print(f'{os.path.basename(p)[:-10]} {b / (1 << 20):.0f} {na} {nf}')
# safe_bdev arrays: group 0 is the slot allocator (group 1 holds stripe
# intents, not space).
for p in sorted(glob.glob(os.path.join(sys.argv[1], '*.alog'))):
  b, na, nf = live_bytes(p, group=0)
  if na:
    print(f'{os.path.basename(p)[:-5]} {b / (1 << 20):.0f} {na} {nf}')
