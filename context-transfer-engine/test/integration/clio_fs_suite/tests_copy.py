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
                     f'--dst-dir {base} --size-mib 1024 --copies 10 --keep',
               timeout=7000)
  # The JSON summary is copy_check's last stdout line; cp's own errors
  # (stderr) are appended after it by sh().
  line = next((ln for ln in reversed(out.splitlines())
               if ln.startswith('{"copies"')), '')
  try:
    res = json.loads(line)
  except ValueError:
    ctx.check(False, f'copy_check rc={rc}: {out[-2000:]}')
    return
  # A copy cp reported as failed (ENOSPC on a full store) is an honest
  # failure; a copy cp reported as fine must be byte-exact.
  silent = [b for b in res['bad'] if b['rc'] == 0]
  reported = [b for b in res['bad'] if b['rc'] != 0]
  brief = lambda bs: [{k: b[k] for k in ('copy', 'rc', 'size', 'bad_pages',
                                          'all_zero_pages', 'first_bad',
                                          'last_bad')} for b in bs]
  if reported:
    ctx.note(f'copies cp reported as failed: {brief(reported)}')
  ctx.metrics['silent_bad_copies'] = len(silent)
  ctx.metrics['reported_failed_copies'] = len(reported)
  ctx.check(not silent, f'cp returned 0 but wrote wrong bytes: '
                        f'{brief(silent)[:4]} sample {silent[0]["sample"][:1] if silent else None}')
