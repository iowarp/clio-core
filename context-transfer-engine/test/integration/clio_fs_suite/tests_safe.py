"""safe_bdev under clio-fs: disks and nodes failing while the system runs.

Profile 'safe' (suite.py --profile safe): every node's slow tier is a
safe_bdev array of cluster.SAFE_MEMBERS file bdevs, the last SAFE_PARITY of
them parity (max_failures = SAFE_PARITY), under a RAM and a fast file tier
with the organizer migrating between them. A disk is "killed" by the file
bdev's fault injection (a `<member file>.fail` marker: every I/O on that
member fails from then on, as on a dead drive).

Every test writes self-verifying records (stress_records) and logs each
version a writer saw fsynced; afterwards every fsynced version must read back
intact -- nothing CORRUPT, FOREIGN or lost -- as long as no array lost more
than max_failures members. Beyond that, reads may FAIL (EIO) but must never
return wrong bytes.
"""

import threading
import time

from cluster import SAFE_MEMBERS, SAFE_PARITY
from suite import test
from tests_fault import restart_cluster
from tests_stress import (CORRUPT, FILE_BLOCKS, FOREIGN, ZERO,
                          _check_filesets)


def _writers(ctx, base, secs, tag):
  """Start one fsynced record-file writer per node (retrying failed rounds).

  Args:
    ctx: test context.
    base: directory for the files.
    secs: how long each writer runs.
    tag: name for the per-node durable logs.
  Returns:
    (thread, replies, logs, nfiles); join the thread before checking.
  """
  n = len(ctx.hosts)
  cl = ctx.cl
  nfiles = max(4, (2 * cl.ram_mb) // 64)
  logs = [f'{cl.run_dir}/{tag}_writer_{i}.log' for i in range(n)]
  for lp in logs:  # a rerun into the same --out must not inherit old lines
    open(lp, 'w').close()
  replies = {}

  def writer(i):
    replies[i] = ctx.a(i).call('rec_fileset', timeout=secs + 900,
                               dirpath=base, writer=i + 1, nfiles=nfiles,
                               blocks=FILE_BLOCKS, secs=secs, seed=i,
                               log_path=logs[i], retry=True)
  th = threading.Thread(target=lambda: ctx.each(writer))
  th.start()
  return th, replies, logs, nfiles


def _writer_errors(replies, n, skip=()):
  """Write errors the writers saw (failed rounds they had to retry)."""
  return {f'node{i}': ((replies.get(i) or {}).get('ret') or {}).get(
      'nerrors', 0) for i in range(n) if i not in skip}


def _bounce(ctx, host, crash, down_s):
  """Take one node down (SIGKILL or graceful stop) and bring it back."""
  cl = ctx.cl
  if crash:
    cl.kill_fuse(host)
    cl.kill_runtime(host)
  else:
    cl.unmount(host)
    cl.stop_runtime(host)
  time.sleep(down_s)
  cl.start_runtime(host)
  ctx.check(cl.runtime_up(host), f'{host} runtime did not restart')
  time.sleep(3)
  ctx.check(cl.mount(host), f'{host} remount failed')
  cl.agents.pop(host, None)


@test('safe_disks_die_during_writes', 'safe', min_nodes=2,
      redeploy_after=True, timeout=5400)
def t_disks_die(ctx):
  """Every node rewrites fsynced record files under tier pressure. After
  30 s one data disk dies in EVERY node's array, after 60 s a second one
  (a parity disk): each array now runs with max_failures members gone.
  Writers must keep going without errors, and every fsynced version must
  read back intact, before and after a full restart with the disks still
  dead."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('dd')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 120, 'diskdie')
  time.sleep(30)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  time.sleep(30)
  for h in cl.hosts:
    cl.kill_disk(h, SAFE_MEMBERS - 1)  # a parity member
  th.join(timeout=120 + 1000)
  errs = _writer_errors(replies, n)
  ctx.metrics['writer_errors'] = errs
  ctx.check(not any(errs.values()),
            f'writes failed with {SAFE_PARITY} disks per array dead '
            f'(within max_failures): {errs}')
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  f'with {SAFE_PARITY} disks dead per array')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart with the disks still dead')


@test('safe_disk_dies_then_crash', 'safe', min_nodes=1,
      redeploy_after=True, timeout=5400)
def t_disk_then_crash(ctx):
  """A disk dies in node0's array while every node writes; 30 s later
  every daemon is SIGKILLed mid-write and the cluster restarts. All
  fsynced versions must survive (the array recovers its membership with
  the dead member still faulty), and the filesystem must take new writes."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('dc')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 90, 'diskcrash')
  time.sleep(20)
  cl.kill_disk(cl.hosts[0], 2)
  time.sleep(30)
  from cluster import parallel
  parallel(cl.kill_fuse, cl.hosts)
  parallel(cl.kill_runtime, cl.hosts)
  th.join(timeout=90 + 900)
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a disk death and a crash')
  p = ctx.p('after_restart')
  ctx.ok(0, 'write_file', path=p, size=8 << 20, seed=11, fsync=True)
  v = ctx.ok(n - 1, 'verify_file', path=p, size=8 << 20, seed=11)
  ctx.check(v['ok'], f'write after restart with a dead disk: {v}')


@test('safe_nodes_stop_and_restart', 'safe', min_nodes=3,
      redeploy_after=True, timeout=5400)
def t_nodes_bounce(ctx):
  """While every node writes: one node is stopped gracefully, another is
  SIGKILLed, each comes back 30 s later, and a disk dies in a third node's
  array meanwhile. Every fsynced version from every writer survives."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('nb')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 180, 'bounce')
  time.sleep(20)
  _bounce(ctx, cl.hosts[n - 1], crash=False, down_s=30)
  cl.kill_disk(cl.hosts[0], 1)
  _bounce(ctx, cl.hosts[n - 2], crash=True, down_s=30)
  th.join(timeout=180 + 1200)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after nodes stopped, crashed and restarted')


@test('safe_beyond_tolerance_never_lies', 'safe', min_nodes=1,
      redeploy_after=True, timeout=3600)
def t_beyond(ctx):
  """node0's array loses max_failures + 1 disks. Reads of data that lived
  there may now fail -- but a read that succeeds must return the right
  bytes: never another file's, never torn, never zeros for written data."""
  from stress_records import file_id_of  # noqa: F401  (record format)
  cl = ctx.cl
  base = ctx.p('bt')
  ctx.ok(0, 'mkdir', path=base)
  names = [f'b{k}' for k in range(8)]
  for nm in names:
    ctx.ok(0, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
           runs=[[0, FILE_BLOCKS]], writer=1, gen=1, fsync=True)
  time.sleep(5)  # let the organizer push data down to the array
  for k in range(SAFE_PARITY + 1):
    cl.kill_disk(cl.hosts[0], k)
  lies, failed, ok = [], 0, 0
  for nm in names:
    r = ctx.call(0, 'rec_scan', timeout=900, path=f'{base}/{nm}', name=nm,
                 nblocks=FILE_BLOCKS)
    if not r['ok']:
      failed += 1  # an I/O error is an honest answer
      continue
    ok += 1
    bad = False
    for start, count, w, g in r['ret']['runs']:
      if w in (CORRUPT, FOREIGN, ZERO) or (w, g) != (1, 1):
        lies.append((nm, start, count, w, g))
        bad = True
    if bad:
      # What the wrong bytes are (zeros, another record, garbage) and
      # whether a second read agrees: a lie that re-reads right is a read
      # path bug, one that persists is stored garbage.
      ctx.note(f'{nm} wrong blocks: {r["ret"].get("corrupt", [])[:2]} '
               f'foreign {dict(list((r["ret"].get("foreign") or {}).items())[:2])}')
      r2 = ctx.call(0, 'rec_scan', timeout=900, path=f'{base}/{nm}', name=nm,
                    nblocks=FILE_BLOCKS)
      ctx.note(f'{nm} second read: ok={r2.get("ok")} '
               f'runs={(r2.get("ret") or {}).get("runs", r2.get("err"))}')
  ctx.metrics.update({'files_readable': ok, 'files_failed': failed})
  ctx.check(not lies, f'reads beyond max_failures returned wrong bytes: '
                      f'{lies[:6]}')


@test('safe_disk_comes_back', 'safe', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_disk_back(ctx):
  """A disk dies under writes and later answers again (marker removed),
  while writers continue; then the cluster restarts. Every fsynced version
  survives -- the returning device's stale contents must never be served
  in place of what was written while it was dead."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('db')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 120, 'diskback')
  time.sleep(20)
  cl.kill_disk(cl.hosts[0], 0)
  time.sleep(40)
  cl.revive_disk(cl.hosts[0], 0)
  th.join(timeout=120 + 1000)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a disk died and came back')
  restart_cluster(ctx, crash=False)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a disk came back and the cluster restarted')


@test('safe_fsync_reports_lost_node', 'safe', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_fsync_lost_node(ctx):
  """Issue #1133: node0 writes 32 MiB (pages hash over every node) and does
  NOT fsync; the last node is SIGKILLed and restarted. Its share of those
  unsynced bytes may be gone, so node0's fsync must fail with EIO rather
  than report success. Bytes written after the node rejoined must fsync
  cleanly, and a file fsynced before the crash must read back intact."""
  n = len(ctx.hosts)
  cl = ctx.cl
  size = 32 << 20
  safe = ctx.p('synced_before')
  ctx.ok(0, 'write_file', path=safe, size=8 << 20, seed=3, fsync=True)
  h = ctx.ok(0, 'open', path=ctx.p('dirty'), flags='wc')
  ctx.ok(0, 'fpwrite', h=h, off=0, length=size, seed=5, timeout=600)
  time.sleep(3)  # let the write-behind ship the full pages to their owners
  _bounce(ctx, cl.hosts[n - 1], crash=True, down_s=30)
  time.sleep(10)  # every node has seen the rejoin
  ctx.err(0, 'fsync', ['EIO'], h=h, timeout=600)
  # Reported once; a write window opened after the rejoin syncs cleanly.
  ctx.ok(0, 'fpwrite', h=h, off=0, length=1 << 20, seed=6, timeout=600)
  ctx.ok(0, 'fsync', h=h, timeout=600)
  ctx.ok(0, 'close', h=h)
  v = ctx.ok(n - 1, 'verify_file', path=safe, size=8 << 20, seed=3,
             timeout=600)
  ctx.check(v['ok'], f'file fsynced before the crash: {v}')


@test('safe_cache_coherent_after_crash', 'safe', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_cache_coherent_after_crash(ctx):
  """#1136: node0 writes files WITHOUT fsync; node1 reads them (populating
  its node-local cache copies); every daemon is SIGKILLed and the cluster
  restarts. Whatever survived of unsynced data is up to the crash -- but
  every node must then read the SAME bytes for each file (or all agree it
  is gone): a node serving a cached copy the owner's recovery no longer
  agrees with is two versions of one file."""
  n = len(ctx.hosts)
  base = ctx.p('cc')
  ctx.ok(0, 'mkdir', path=base)
  names = [f'f{k}' for k in range(8)]
  for k, nm in enumerate(names):
    ctx.ok(0, 'write_file', path=f'{base}/{nm}', size=(4 << 20) + 1000 * k,
           seed=40 + k, fsync=False, timeout=600)
  ctx.ok(0, 'sh', cmd=f'python3 -c "import os; os.fsync(os.open({base!r}, '
                      f'os.O_RDONLY))"', timeout=120)  # the NAMES survive
  time.sleep(3)  # let the write-behind ship the pages to their owners
  for nm in names:
    ctx.ok(1, 'sha256', path=f'{base}/{nm}', timeout=600)
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  split = []
  for nm in names:
    seen = {}
    for i in range(n):
      r = ctx.call(i, 'sha256', path=f'{base}/{nm}', timeout=600)
      seen[f'node{i}'] = ((r.get('ret') or {}).get('sha256', '')[:12],
                          (r.get('ret') or {}).get('size')) if r.get('ok') \
          else f'err {r.get("errno")}'
    if len(set(map(str, seen.values()))) > 1:
      split.append((nm, seen))
  ctx.metrics['files_split'] = len(split)
  ctx.check(not split, f'nodes disagree about unsynced files after a crash '
                       f'restart: {split[:3]}')
