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
  # CLIO_SUITE_CP_COPIES: more copies = more prior traffic for the later
  # ones (#1116 failed more often on a store that had seen traffic).
  copies = int(os.environ.get('CLIO_SUITE_CP_COPIES', '10'))
  rc, out = sh(host, f'python3 {HERE}/copy_check.py --src {src} '
                     f'--dst-dir {base} --size-mib 1024 --copies {copies} '
                     f'--keep',
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


FILL_SCRIPT = r'''
import errno, json, os, sys
d, mib = sys.argv[1], int(sys.argv[2])
buf = os.urandom(1 << 20)
res = {"files": 0, "fail": None}
for i in range(10000):
    p = os.path.join(d, "f%d" % i)
    step, done = "open", 0
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            step = "write"
            for _ in range(mib):
                os.write(fd, buf)
                done += 1
            step = "fsync"
            os.fsync(fd)
            step = "close"
        finally:
            os.close(fd)
        res["files"] += 1
    except OSError as e:
        st = os.stat(p) if os.path.exists(p) else None
        res["fail"] = {"file": p, "errno": errno.errorcode.get(e.errno, e.errno),
                       "step": step, "mib_written_before": done,
                       "size_after": st.st_size if st else None}
        break
print(json.dumps(res))
'''


@test('enospc_reported_honestly', 'copy', min_nodes=1, timeout=3600)
def t_enospc(ctx):
  """#1129: fill the store with 256 MiB files until a write fails. The
  failure must be ENOSPC (not EIO), and afterwards opening the failed file
  and an earlier good one must work (a failed write's error belongs to
  write/fsync/close, not to every later open), with the good file intact."""
  base = ctx.p('fill')
  ctx.ok(0, 'mkdir', path=base)
  script = f'{ctx.cl.local_root}/fill.py'
  rc, _ = sh(ctx.hosts[0], f"cat > {script} <<'PYEOF'\n{FILL_SCRIPT}\nPYEOF")
  rc, out = sh(ctx.hosts[0], f'python3 {script} {base} 256', timeout=3400)
  line = next((ln for ln in reversed(out.splitlines()) if ln.startswith('{')),
              '')
  res = json.loads(line) if line else {}
  ctx.metrics.update({'files_written': res.get('files'),
                      'failure': res.get('fail')})
  fail = res.get('fail')
  ctx.check(fail is not None, f'the store never filled: {out[-500:]}')
  ctx.check(fail['errno'] == 'ENOSPC',
            f'a full store failed with {fail["errno"]}, not ENOSPC')
  for p in (fail['file'], f'{base}/f0'):
    r = ctx.call(0, 'open', path=p, flags='r')
    ctx.check(r.get('ok'), f'open({p}) after the store filled: {r.get("err")}')
    ctx.ok(0, 'close', h=r['ret'])
  r = ctx.call(0, 'sha256', path=f'{base}/f0', timeout=600)
  ctx.check(r.get('ok') and r['ret']['size'] == 256 << 20,
            f'the first (fsynced) file after the fill: {r}')
