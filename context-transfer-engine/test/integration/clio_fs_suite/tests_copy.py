"""Large sequential copies into clio-fs, checked byte for byte (#1116).

`cp` of a 1 GiB random file must produce identical bytes every time. The
copy runs on the node itself (cp's 128 KiB writes through FUSE), with the
source on node-local disk outside the mount.
"""

import json
import os

from cluster import sh
from suite import test

HERE = os.path.dirname(os.path.abspath(__file__))


@test('cp_1g_byte_exact', 'copy', min_nodes=1, timeout=7200)
def t_cp_1g(ctx):
  """cp a 1 GiB random file into the mount 10 times; every copy must match
  the source exactly (reports the bad 128 KiB sub-blocks and whose bytes
  they hold)."""
  cl = ctx.cl
  host = ctx.hosts[0]
  base = ctx.p('cp')
  ctx.ok(0, 'mkdir', path=base)
  src = f'{cl.local_root}/cp_src_1g.bin'
  rc, out = sh(host, f'python3 {HERE}/copy_check.py --src {src} '
                     f'--dst-dir {base} --size-mib 1024 --copies 10',
               timeout=7000)
  line = (out.strip().splitlines() or [''])[-1]
  try:
    res = json.loads(line)
  except ValueError:
    ctx.check(False, f'copy_check rc={rc}: {out[-2000:]}')
    return
  ctx.metrics['bad_copies'] = len(res['bad'])
  ctx.check(not res['bad'], f'cp produced wrong bytes: {res["bad"][:3]}')
