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

import os
import random
import threading
import time

from cluster import SAFE_MEMBERS, SAFE_PARITY, AgentConn, parallel, sh
from suite import TestFailure, test
from tests_fault import restart_cluster
from tests_stress import (CORRUPT, FILE_BLOCKS, FOREIGN, ZERO, _apply,
                          _check_filesets, _compare, _corrupt_blocks,
                          _tier_mb, _tier_usage, _to_runs)


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


@test('safe_cache_coherent_after_partial_crash', 'safe', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_cache_coherent_partial_crash(ctx):
  """#1136 (restart-invalidation gap): node1 caches unsynced files written on
  node0; then every node EXCEPT node1 is SIGKILLed and restarted. The
  restarted owners' recovery may shorten pages (unsynced volatile tails are
  dropped) while node1 -- never restarted, so its cached copies were never
  dropped -- still holds the pre-crash bytes. Every node must still read the
  same bytes for each file."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('pc')
  ctx.ok(0, 'mkdir', path=base)
  names = [f'g{k}' for k in range(8)]
  for k, nm in enumerate(names):
    ctx.ok(0, 'write_file', path=f'{base}/{nm}', size=(4 << 20) + 777 * k,
           seed=60 + k, fsync=False, timeout=600)
  ctx.ok(0, 'sh', cmd=f'python3 -c "import os; os.fsync(os.open({base!r}, '
                      f'os.O_RDONLY))"', timeout=120)
  time.sleep(3)
  for nm in names:
    ctx.ok(1, 'sha256', path=f'{base}/{nm}', timeout=600)
  others = [h for i, h in enumerate(cl.hosts) if i != 1]
  from cluster import parallel
  parallel(cl.kill_fuse, others)
  parallel(cl.kill_runtime, others)
  time.sleep(5)
  for h in others:
    cl.start_runtime(h)
  for h in others:
    ctx.check(cl.runtime_up(h), f'{h} runtime did not restart')
  time.sleep(5)
  for h in others:
    ctx.check(cl.mount(h), f'{h} remount failed')
    cl.agents.pop(h, None)
  time.sleep(10)  # recovery (restart pulls, peers see the rejoin)
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
  ctx.check(not split, f'nodes disagree after a crash of every node but the '
                       f'one holding cached copies: {split[:3]}')


@test('safe_disk_replaced_and_rebuilt', 'safe', min_nodes=1,
      redeploy_after=True, timeout=5400)
def t_disk_replaced(ctx):
  """The operator's repair path under load: a data disk dies in every
  node's array while writers run, is swapped for a fresh disk and rebuilt
  onto it (RecoverBdev) with the writers still going. Redundancy must be
  back afterwards: two MORE members per array then die (max_failures again,
  three originals dead in all) and every fsynced version must still read
  back intact -- before and after a crash restart, which must bring the
  replacement disk back as a member rather than the dead original."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('dr')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 150, 'diskrepl')
  time.sleep(20)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  time.sleep(20)
  t0 = time.time()
  res = parallel(lambda h: cl.replace_disk(h, 0), cl.hosts)
  ctx.metrics['rebuild_s'] = round(time.time() - t0, 1)
  bad = {h: r for h, r in zip(cl.hosts, res)
         if isinstance(r, Exception) or r[0] != 0}
  ctx.check(not bad, f'rebuild onto the replacement disk failed: '
                     f'{ {h: str(r)[-400:] for h, r in bad.items()} }')
  th.join(timeout=150 + 1000)
  errs = _writer_errors(replies, n)
  ctx.metrics['writer_errors'] = errs
  ctx.check(not any(errs.values()),
            f'writes failed across a disk death and its rebuild: {errs}')
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a dead disk was replaced and rebuilt')
  for h in cl.hosts:  # max_failures more: only parity can cover them now
    cl.kill_disk(h, 1)
    cl.kill_disk(h, SAFE_MEMBERS - 1)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  f'after {SAFE_PARITY} more disks died behind the rebuild')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart with the replacement disk seated')


def _log_has(cl, host, needle, wait_s=60):
  """True if `host`'s runtime log (on the shared run dir) contains needle.

  The daemon appends over NFS and the driver's view of the file lags its
  writes by the attribute-cache time, so the file is re-read for up to
  `wait_s` seconds before giving up.
  """
  deadline = time.time() + wait_s
  while True:
    try:
      with open(cl.log_path(host, 'runtime'), errors='replace') as f:
        if needle in f.read():
          return True
    except OSError:
      pass
    if time.time() >= deadline:
      return False
    time.sleep(2)


@test('safe_rebuild_interrupted_then_crash', 'safe', min_nodes=2,
      redeploy_after=True, timeout=5400)
def t_rebuild_interrupted(ctx):
  """A rebuild that does not get to finish. A data disk dies in every
  node's array under writers and is replaced; on node1 the rebuild onto the
  replacement stops part-way (the RECOVER_MAX_ROWS hook: the member stays
  'recovering', as a crash mid-rebuild leaves it). node1 then loses a parity
  disk too (max_failures down, one of them the half-built replacement) and is
  SIGKILLed mid-write. Its restart must resume and finish the rebuild from
  exactly k survivors while every other node keeps writing, and every
  fsynced version must read back intact -- then with one more disk dead
  per array, and after a crash restart of the whole cluster."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('ri')
  ctx.ok(0, 'mkdir', path=base)
  victim = cl.hosts[1 % n]
  th, replies, logs, nfiles = _writers(ctx, base, 240, 'rebuildint')
  time.sleep(15)
  # node1's daemon restarts with the hook: its next rebuild stops after 8
  # rows and leaves the member recovering.
  cl.extra_env['CLIO_SAFE_BDEV_RECOVER_MAX_ROWS'] = '8'
  try:
    _bounce(ctx, victim, crash=False, down_s=5)
  finally:
    cl.extra_env.pop('CLIO_SAFE_BDEV_RECOVER_MAX_ROWS', None)
  time.sleep(15)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  time.sleep(15)
  t0 = time.time()
  res = parallel(lambda h: cl.replace_disk(h, 0), cl.hosts)
  ctx.metrics['rebuild_s'] = round(time.time() - t0, 1)
  bad = {h: r for h, r in zip(cl.hosts, res)
         if isinstance(r, Exception) or r[0] != 0}
  ctx.check(not bad, f'rebuild onto the replacement disk failed: '
                     f'{ {h: str(r)[-400:] for h, r in bad.items()} }')
  ctx.check(_log_has(cl, victim, 'rebuild interrupted (test hook)'),
            f'{victim}: the rebuild was not interrupted by the hook')
  # max_failures on node1: the half-built replacement and a parity disk.
  cl.kill_disk(victim, SAFE_MEMBERS - 1)
  time.sleep(10)
  _bounce(ctx, victim, crash=True, down_s=20)
  ctx.check(_log_has(cl, victim, 'resuming interrupted recovery'),
            f'{victim}: restart did not resume the interrupted rebuild')
  ctx.check(_log_has(cl, victim, 'completed on restart'),
            f'{victim}: the resumed rebuild did not complete')
  th.join(timeout=240 + 1200)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after an interrupted rebuild, a parity death and a crash')
  for h in cl.hosts:  # one more per array: node1 is at max_failures again
    cl.kill_disk(h, 1)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after one more disk died behind the resumed rebuild')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart with the resumed replacement seated')


@test('safe_two_nodes_down_never_lies', 'safe', min_nodes=4,
      redeploy_after=True, timeout=5400)
def t_two_nodes_down(ctx):
  """Two adjacent nodes -- a primary and the node that holds its remote
  copies (container id + 1) -- are SIGKILLed at once while every node writes
  fsynced record files. Data homed on the pair may be unreachable until they
  return: a read from a survivor may fail (EIO) but must never return wrong
  bytes, and writers elsewhere must keep going. Both come back 30 s later;
  every fsynced version from every writer must then read back intact, and
  again after a crash restart of the whole cluster."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('tn')
  ctx.ok(0, 'mkdir', path=base)
  # Fsynced files written before the outage, probed from a survivor during it.
  names = [f'p{k}' for k in range(8)]
  for nm in names:
    ctx.ok(0, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
           runs=[[0, FILE_BLOCKS]], writer=1, gen=1, fsync=True)
  # Long enough to outlive the outage: the probes below take ~20 s per
  # unreadable file, so the pair is back only ~3 min in.
  th, replies, logs, nfiles = _writers(ctx, base, 300, 'twodown')
  time.sleep(20)
  down = [cl.hosts[n - 2], cl.hosts[n - 1]]
  t_kill = time.time()
  parallel(cl.kill_fuse, down)
  parallel(cl.kill_runtime, down)
  time.sleep(5)
  lies, failed, ok = [], 0, 0
  # Probe through a connection of its own: a node's agent serializes its
  # calls, so a probe on node0's shared agent waited behind node0's 300 s
  # writer call and that wait read as a 274 s stat (#1169). Each step is
  # timed -- the stat, the first 4 KiB read, then the record scan -- so a
  # slow one is named.
  probe = AgentConn(cl.hosts[0], cl.agent_py, cl.env_prefix())
  step_s = []
  for nm in names[:3]:
    path = f'{base}/{nm}'
    t = time.time()
    st = probe.call('stat', timeout=900, path=path)
    t_stat = round(time.time() - t, 1)
    t = time.time()
    rd = probe.call('read_hex', timeout=900, path=path, off=0, length=4096)
    t_read = round(time.time() - t, 1)
    t = time.time()
    r = probe.call('rec_scan', timeout=900, path=path, name=nm,
                   nblocks=FILE_BLOCKS)
    t_scan = round(time.time() - t, 1)
    step_s.append([nm, t_stat, st.get('ok'), t_read, rd.get('ok'), t_scan,
                   r.get('ok')])
    if not r['ok']:
      failed += 1  # an I/O error is an honest answer
      continue
    ok += 1
    for start, count, w, g in r['ret']['runs']:
      if w in (CORRUPT, FOREIGN, ZERO) or (w, g) != (1, 1):
        lies.append((nm, start, count, w, g))
  probe.close()
  ctx.metrics.update({'files_readable_during_outage': ok,
                      'files_failed_during_outage': failed,
                      'probe_steps': step_s})
  ctx.note(f'outage probes [file, stat s, ok, 4K read s, ok, scan s, ok]: '
           f'{step_s}')
  ctx.check(not lies, f'reads with two adjacent nodes down returned wrong '
                      f'bytes: {lies[:6]}')
  ctx.metrics['probe_s'] = round(time.time() - t_kill, 1)
  time.sleep(5)
  parallel(lambda h: cl.start_runtime(h), down)
  ups = parallel(cl.runtime_up, down)
  ctx.check(all(u is True for u in ups), f'runtime restart failed: {ups}')
  time.sleep(3)
  ms = parallel(cl.mount, down)
  ctx.check(all(m is True for m in ms), f'remount failed: {ms}')
  for h in down:
    cl.agents.pop(h, None)
  back_at = time.time()
  ctx.metrics['outage_s'] = round(back_at - t_kill, 1)
  # The filesystem must take writes again once the pair is back: one file
  # checked from another node, and a second round of record writers whose
  # every fsynced version is verified like the first round's.
  p = ctx.p('after_return')
  ctx.ok(0, 'write_file', path=p, size=8 << 20, seed=17, fsync=True)
  v = ctx.ok(n - 1, 'verify_file', path=p, size=8 << 20, seed=17)
  ctx.check(v['ok'], f'write after the pair returned: {v}')
  base2 = ctx.p('tn_after')
  ctx.ok(0, 'mkdir', path=base2)
  th2, replies2, logs2, nfiles2 = _writers(ctx, base2, 60, 'twodown_after')
  th.join(timeout=300 + 1200)
  th2.join(timeout=60 + 600)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  ctx.metrics['writer_errors_after_return'] = _writer_errors(replies2, n)
  ctx.metrics['writer_secs_after_return'] = round(time.time() - back_at, 1)
  for i in range(n):  # what the failed rounds saw (first few per node)
    errs = ((replies.get(i) or {}).get('ret') or {}).get('errors') or []
    if errs:
      ctx.note(f'node{i} first writer errors: {errs[:2]}')
    errs2 = ((replies2.get(i) or {}).get('ret') or {}).get('errors') or []
    if errs2:
      ctx.note(f'node{i} writer errors after the return: {errs2[:2]}')
  ctx.check(not any(ctx.metrics['writer_errors_after_return'].values()),
            'writes failed after both nodes were back: '
            f'{ctx.metrics["writer_errors_after_return"]}')
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after two adjacent nodes crashed and returned')
  _check_filesets(ctx, base2, n, nfiles2, logs2, replies2,
                  'for the writers started after the pair returned')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart following the double node loss')
  _check_filesets(ctx, base2, n, nfiles2, logs2, replies2,
                  'after a crash restart, for the writers started after the '
                  'return')


@test('safe_chaos_disks_and_nodes', 'safe', min_nodes=3,
      redeploy_after=True, timeout=5400)
def t_chaos_disks_and_nodes(ctx):
  """Random faults in a loop under fsynced record writers: a disk dies in
  some node's array (never past max_failures per array), a dead disk comes
  back or is replaced and rebuilt, a node is SIGKILLed or stopped and
  brought back -- six rounds, in an order drawn from CLIO_SUITE_CHAOS_SEED
  (default 7), so a failing run can be repeated. No array ever loses more
  than it can cover, so every fsynced version must read back intact at the
  end and after a crash restart of the whole cluster."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('ch')
  ctx.ok(0, 'mkdir', path=base)
  rng = random.Random(int(os.environ.get('CLIO_SUITE_CHAOS_SEED', '7')))
  down = {h: set() for h in cl.hosts}      # members the array cannot use
  replaced = {h: set() for h in cl.hosts}  # members rebuilt onto new disks
  th, replies, logs, nfiles = _writers(ctx, base, 600, 'chaos')
  time.sleep(20)
  events = []
  for rnd in range(6):
    h = rng.choice(cl.hosts)
    roll = rng.random()
    if roll < 0.45 and len(down[h]) < SAFE_PARITY:
      k = rng.choice([k for k in range(SAFE_MEMBERS)
                      if k not in down[h] and k not in replaced[h]])
      cl.kill_disk(h, k)
      down[h].add(k)
      events.append([rnd, h, 'kill_disk', k])
    elif roll < 0.65 and down[h]:
      k = rng.choice(sorted(down[h]))
      if rng.random() < 0.5:
        # The device answers again; the array keeps it faulty (its contents
        # are stale), so it still counts against the budget.
        cl.revive_disk(h, k)
        events.append([rnd, h, 'revive_disk', k])
      else:
        rc, out = cl.replace_disk(h, k, gen=1 + len(replaced[h]))
        ctx.check(rc == 0, f'round {rnd}: rebuild of member {k} on {h} '
                           f'failed: {str(out)[-300:]}')
        down[h].discard(k)
        replaced[h].add(k)
        events.append([rnd, h, 'replace_disk', k])
    else:
      crash = rng.random() < 0.5
      _bounce(ctx, h, crash=crash, down_s=15)
      events.append([rnd, h, 'crash' if crash else 'stop'])
    ctx.note(f'round {rnd}: {events[-1]}')
    time.sleep(45)
  ctx.metrics['events'] = events
  th.join(timeout=600 + 1200)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after six rounds of random disk and node faults')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart following the chaos rounds')


@test('safe_rolling_restart_degraded', 'safe', min_nodes=2,
      redeploy_after=True, timeout=7200)
def t_rolling_restart_degraded(ctx):
  """A rolling restart of the whole cluster while every array runs
  degraded: a data disk is dead on every node from the start, and every
  node in turn is taken down (alternately SIGKILLed and stopped) and
  brought back while all nodes keep writing fsynced record files. Halfway
  through, a parity disk also dies on node0 (max_failures there). Crashes
  land in the middle of degraded writes, the case the degraded-write
  journal exists for (#1137): every fsynced version must survive, and the
  writers on nodes that stayed up must see no errors once their peer is
  back."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('rr')
  ctx.ok(0, 'mkdir', path=base)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  secs = 60 + 75 * n
  th, replies, logs, nfiles = _writers(ctx, base, secs, 'rolling')
  time.sleep(30)
  for i, h in enumerate(cl.hosts):
    _bounce(ctx, h, crash=(i % 2 == 0), down_s=20)
    if i == n // 2:
      cl.kill_disk(cl.hosts[0], SAFE_MEMBERS - 1)
    time.sleep(30)  # degraded writes resume on the rejoined node
  th.join(timeout=secs + 1800)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a rolling restart with every array degraded')
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after the rolling restart and a full crash restart')


# ---------------------------------------------------------------------------
# Tiering under faults: overflow onto degraded arrays, crash, rebuild
# ---------------------------------------------------------------------------
def _ovf_nfiles(ctx):
  """Number of 64 MiB record files per node that push the data through the
  RAM and fast tiers onto the safe arrays. Same sizing as
  stress_tier_overflow: every fsynced byte lands twice on the arrays (the
  primary and its remote copy), so stay near 70% of one disk_gb array."""
  per_node_mb = int(_tier_mb(ctx) * 1.5)
  if len(ctx.hosts) > 1:
    per_node_mb = min(per_node_mb, int(ctx.cl.disk_gb * 1024 * 0.7 / 2))
  return max(2, per_node_mb // 64)


def _ovf_verify(ctx, base, models, owned, reader_of, tag, tally, bad):
  """Scan every node's files from reader_of(owner) and compare each with
  its model. Differing and CORRUPT/FOREIGN block counts accumulate in
  `tally`, the first differing blocks in `bad`; metrics get the verify
  time and running count under `tag`. A file that cannot be read at all
  fails the test: no array ever loses more than max_failures members here."""
  lock = threading.Lock()

  def one(i):
    r = reader_of(i)
    for nm in owned[i]:
      got = ctx.ok(r, 'rec_scan', timeout=900, path=f'{base}/{nm}',
                   name=nm, nblocks=FILE_BLOCKS)
      ncorrupt = _corrupt_blocks(got['runs'])
      nbad = _compare(_to_runs(models[nm]), got, FILE_BLOCKS, nm, bad)
      with lock:
        tally['corrupt'] += ncorrupt
        tally['mismatch'] += nbad
  t0 = time.time()
  ctx.each(one)
  ctx.metrics[f'verify_{tag}_s'] = round(time.time() - t0, 1)
  ctx.metrics[f'bad_after_{tag}'] = tally['mismatch']


def _ovf_overwrite(ctx, base, owned, models):
  """From a third node, overwrite six random ranges of every file (gen 2)
  with fsync and record them in the models."""
  n = len(ctx.hosts)
  rng = random.Random(4321)
  plan = {}
  for i in range(n):
    for nm in owned[i]:
      plan[nm] = []
      for _ in range(6):
        s = rng.randrange(FILE_BLOCKS)
        plan[nm].append([s, min(rng.randrange(1, 512), FILE_BLOCKS - s)])

  def overwrite(i):
    ow = (i + 2) % n
    for nm in owned[i]:
      ctx.ok(ow, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=plan[nm], writer=100 + ow, gen=2, fsync=True)
  ctx.each(overwrite)
  for i in range(n):
    for nm in owned[i]:
      _apply(models[nm], plan[nm], 100 + (i + 2) % n, 2)


@test('safe_tier_overflow_degraded', 'safe', min_nodes=2,
      redeploy_after=True, timeout=5400)
def t_tier_overflow_degraded(ctx):
  """Tiering with the arrays already hurt. A data disk dies in every node's
  array first; then every node writes 1.5x its RAM + fast tiers in fsynced
  64 MiB record files, so the organizer pushes the data down onto the
  degraded arrays (every stripe is written with a member missing). Every
  block must read back from another node; then after a crash restart of
  the whole cluster with the disks still dead; then, once the dead disks
  are replaced and rebuilt, with max_failures OTHER members killed (the
  rebuilt member is now the only holder of its column); and finally after
  random ranges are overwritten in that state."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('tod')
  ctx.ok(0, 'mkdir', path=base)
  nfiles = _ovf_nfiles(ctx)
  ctx.metrics['bytes_per_node_mib'] = nfiles * 64
  for h in cl.hosts:
    cl.kill_disk(h, 1)  # a data member of every array
  owned = {i: [f'n{i}_f{k}' for k in range(nfiles)] for i in range(n)}
  models = {nm: [(i, 1)] * FILE_BLOCKS for i in range(n) for nm in owned[i]}

  def write_all(i):
    for nm in owned[i]:
      ctx.ok(i, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=[[0, FILE_BLOCKS]], writer=i, gen=1, fsync=True)
  t0 = time.time()
  try:
    ctx.each(write_all)
  except TestFailure:
    ctx.note(f'tier usage when a write failed: {_tier_usage(ctx)}')
    raise
  ctx.metrics['write_MiB_per_s'] = round(
      n * nfiles * 64 / max(0.001, time.time() - t0))
  ctx.note(f'tiers after the writes onto degraded arrays: {_tier_usage(ctx)}')
  tally = {'mismatch': 0, 'corrupt': 0}
  bad = []
  _ovf_verify(ctx, base, models, owned, lambda i: (i + 1) % n, 'degraded',
              tally, bad)
  restart_cluster(ctx, crash=True)
  cl.agents.clear()
  _ovf_verify(ctx, base, models, owned, lambda i: (i + 2) % n, 'crash',
              tally, bad)
  t0 = time.time()
  res = parallel(lambda h: cl.replace_disk(h, 1), cl.hosts)
  ctx.metrics['rebuild_s'] = round(time.time() - t0, 1)
  failed = {h: str(r)[-400:] for h, r in zip(cl.hosts, res)
            if isinstance(r, Exception) or r[0] != 0}
  ctx.check(not failed, f'rebuild onto the replacement disks failed: {failed}')
  for h in cl.hosts:  # max_failures others: the rebuilt member must serve
    cl.kill_disk(h, 0)
    cl.kill_disk(h, SAFE_MEMBERS - 1)
  _ovf_verify(ctx, base, models, owned, lambda i: (i + 3) % n, 'rebuilt',
              tally, bad)
  _ovf_overwrite(ctx, base, owned, models)
  _ovf_verify(ctx, base, models, owned, lambda i: i, 'overwrite', tally, bad)
  ctx.metrics['corrupt_or_foreign_blocks'] = tally['corrupt']
  ctx.metrics['bad_blocks_total'] = tally['mismatch']
  ctx.check(tally['mismatch'] == 0 and tally['corrupt'] == 0 and not bad,
            f'{tally["mismatch"]} blocks differ from the model '
            f'({tally["corrupt"]} CORRUPT/FOREIGN), e.g. {bad[:8]}')


# ---------------------------------------------------------------------------
# A dead disk swapped for a blank one while its node is down
# ---------------------------------------------------------------------------
def _log_tail_has(cl, host, needle, offset, wait_s=30):
  """True if `host`'s runtime log contains needle AFTER byte `offset` (the
  part written since the node was restarted); re-read for up to wait_s
  seconds because the driver's NFS view lags the daemon's appends."""
  deadline = time.time() + wait_s
  while True:
    try:
      with open(cl.log_path(host, 'runtime'), errors='replace') as f:
        f.seek(offset)
        if needle in f.read():
          return True
    except OSError:
      pass
    if time.time() >= deadline:
      return False
    time.sleep(2)


def _swap_member_blank(cl, host, k):
  """Simulate the operator swapping failed member k of `host`'s array for a
  blank drive while the node is down: the member's backing file, its
  allocation log and the fault marker go away, so the file bdev recreates
  an empty member with no superblock at the next start."""
  m = cl.safe_member_path(k)
  alog = m[:-len('.dat')] + '.alog'
  rc, out = sh(host, f'rm -f {m} {m}.fail {alog} && ls {os.path.dirname(m)}',
               timeout=60)
  return rc, out


@test('safe_disk_swapped_while_down', 'safe', min_nodes=2,
      redeploy_after=True, timeout=5400)
def t_disk_swapped_while_down(ctx):
  """The operator's offline repair path. A data disk dies in node1's array
  while every node writes fsynced record files; node1 is stopped
  gracefully, the dead disk is swapped for a BLANK one while the node is
  down (backing file, allocation log and fault marker gone, so the member
  comes back empty and without a superblock), and node1 restarts. The
  array must not seat the blank member as if it still held its column:
  every fsynced version must read back intact (reconstructed from parity,
  or after a rebuild onto the blank member), never zeros or corrupt bytes,
  and the filesystem must take new writes. How the member was seated is
  read from node1's log and recorded. A crash restart of the whole cluster
  must then keep the data intact too."""
  n = len(ctx.hosts)
  cl = ctx.cl
  vi = 1 % n
  victim = cl.hosts[vi]
  base = ctx.p('sw')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 150, 'swap')
  time.sleep(20)
  cl.kill_disk(victim, 2)
  time.sleep(20)
  cl.unmount(victim)
  cl.stop_runtime(victim)
  log_off = os.path.getsize(cl.log_path(victim, 'runtime'))
  rc, out = _swap_member_blank(cl, victim, 2)
  ctx.check(rc == 0, f'blank swap of member 2 on {victim} failed: {out[-300:]}')
  cl.start_runtime(victim)
  ctx.check(cl.runtime_up(victim),
            f'{victim} did not come back with the blank member seated')
  time.sleep(3)
  ctx.check(cl.mount(victim), f'{victim} remount failed after the swap')
  cl.agents.pop(victim, None)
  # How the array seated column 2 at this start (the member manifest is
  # expected to keep it down until an explicit rebuild).
  mp = cl.safe_member_path(2)
  seated = 'unknown'
  for needle, label in ((f"data column 2 ('{mp}') restored as down",
                         'restored-down'),
                        (f"initialized fresh member '{mp}'", 'fresh-active'),
                        (f"re-attached member '{mp}'", 'reattached'),
                        (f"REFUSING member '{mp}'", 'refused')):
    if _log_tail_has(cl, victim, needle, log_off, wait_s=20):
      seated = label
      break
  ctx.check(seated != 'fresh-active',
            'the blank member was seated as an active data column: the '
            'array would serve zeros for everything that lived on it')
  ctx.metrics['blank_member_seated_as'] = seated
  ctx.note(f'{victim} seated the blank member as: {seated}')
  # The operator's next step: rebuild onto the blank disk IN PLACE (same
  # path, same member pool). Record whether the tool supports that; the
  # integrity checks below hold either way.
  t0 = time.time()
  rc, out = cl.rebuild_disk_inplace(victim, 2)
  ctx.metrics['inplace_rebuild_rc'] = rc
  ctx.metrics['inplace_rebuild_s'] = round(time.time() - t0, 1)
  if rc != 0:
    ctx.note(f'in-place rebuild onto the blank member refused: {out[-400:]}')
  th.join(timeout=150 + 1200)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a dead disk was swapped for a blank one offline')
  if rc == 0:
    # Redundancy is claimed back: max_failures OTHER members may now die
    # and the rebuilt column must carry its share.
    cl.kill_disk(victim, 0)
    cl.kill_disk(victim, SAFE_MEMBERS - 1)
    _check_filesets(ctx, base, n, nfiles, logs, replies,
                    f'with {SAFE_PARITY} more members dead behind the '
                    'in-place rebuild')
  p = ctx.p('after_swap')
  ctx.ok(vi, 'write_file', path=p, size=8 << 20, seed=17, fsync=True)
  v = ctx.ok((vi + 1) % n, 'verify_file', path=p, size=8 << 20, seed=17)
  ctx.check(v['ok'], f'write on the repaired node after the swap: {v}')
  restart_cluster(ctx, crash=True)
  cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after the blank swap and a full crash restart')


# ---------------------------------------------------------------------------
# Repeated whole-cluster crashes under load: recovery must stay cheap
# ---------------------------------------------------------------------------
def _crash_all_keep_writers(ctx):
  """SIGKILL every daemon and FUSE client and bring the cluster back, WITHOUT
  closing the node agents (writers started through them keep running and
  retry through the outage). Returns the restart time in seconds."""
  cl = ctx.cl
  parallel(lambda h: cl.kill_fuse(h), cl.hosts)
  parallel(lambda h: cl.kill_runtime(h), cl.hosts)
  time.sleep(2)
  t0 = time.time()
  parallel(lambda h: cl.start_runtime(h), cl.hosts)
  ups = parallel(cl.runtime_up, cl.hosts)
  ctx.check(all(u is True for u in ups), f'runtime restart failed: {ups}')
  time.sleep(3)
  ms = parallel(cl.mount, cl.hosts)
  ctx.check(all(m is True for m in ms), f'remount failed: {ms}')
  for h in cl.hosts:
    cl.agents.pop(h, None)  # new calls get fresh agents; writers keep theirs
  return round(time.time() - t0, 1)


def _array_file_sizes(ctx):
  """Bytes of each node's safe_array allocation log, degraded-write journal
  and member manifest, as {node: {name: bytes}}."""
  lr = ctx.cl.local_root
  out = {}
  for i, h in enumerate(ctx.hosts):
    rc, txt = sh(h, f'stat -c "%s %n" {lr}/data/safe_array.alog* 2>/dev/null',
                 timeout=30)
    sizes = {}
    for ln in (txt or '').splitlines():
      parts = ln.split()
      if len(parts) == 2:
        sizes[parts[1].rsplit('/', 1)[-1]] = int(parts[0])
    out[f'node{i}'] = sizes
  return out


@test('safe_crash_cycles', 'safe', min_nodes=2, redeploy_after=True,
      timeout=5400)
def t_crash_cycles(ctx):
  """Six whole-cluster SIGKILLs a minute apart while every node writes
  fsynced record files with a data disk dead in every array (so every
  crash lands on degraded writes and the degraded-write journal is in
  use). Recovery must not get more expensive as the journal, allocation
  log and manifest accumulate: every restart is timed and the array files
  are measured after each cycle; the last restart may take at most 3x the
  first (+5 s). Every fsynced version must read back intact at the end, and
  the filesystem must take new writes."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('cc')
  ctx.ok(0, 'mkdir', path=base)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  cycles = 6
  secs = 60 * cycles + 60
  th, replies, logs, nfiles = _writers(ctx, base, secs, 'cycles')
  restarts, sizes = [], []
  for c in range(cycles):
    time.sleep(60)
    restarts.append(_crash_all_keep_writers(ctx))
    sizes.append(_array_file_sizes(ctx))
    ctx.note(f'cycle {c}: restart {restarts[-1]} s, array files '
             f'{sizes[-1].get("node0")}')
  th.join(timeout=secs + 1800)
  ctx.metrics['restart_s_per_cycle'] = restarts
  ctx.metrics['journal_bytes_node0_per_cycle'] = [
      s.get('node0', {}).get('safe_array.alog.journal', 0) for s in sizes]
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  ctx.check(restarts[-1] <= 3 * restarts[0] + 5,
            f'recovery got slower across crashes: {restarts}')
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  f'after {cycles} whole-cluster crashes under degraded writes')
  p = ctx.p('after_cycles')
  ctx.ok(0, 'write_file', path=p, size=8 << 20, seed=23, fsync=True)
  v = ctx.ok(n - 1, 'verify_file', path=p, size=8 << 20, seed=23)
  ctx.check(v['ok'], f'write after the crash cycles: {v}')


# ---------------------------------------------------------------------------
# A parity disk replaced while a data disk is also dead (#1199)
# ---------------------------------------------------------------------------
@test('safe_parity_replaced_with_data_down', 'safe', min_nodes=1,
      redeploy_after=True, timeout=5400)
def t_parity_replaced_with_data_down(ctx):
  """Two members down in every array -- a data disk and a parity disk, i.e.
  max_failures -- while every node writes fsynced record files. The parity
  disk is then swapped for a fresh one and rebuilt with the data disk still
  dead: the rebuild must succeed (the down data column is decoded from the
  other parity), redundancy must be back (a SECOND data disk then dies and
  every fsynced version still reads back), and a crash restart must keep
  it all."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('pr')
  ctx.ok(0, 'mkdir', path=base)
  th, replies, logs, nfiles = _writers(ctx, base, 150, 'parrep')
  time.sleep(20)
  for h in cl.hosts:
    cl.kill_disk(h, 1)                 # a data member
  time.sleep(15)
  for h in cl.hosts:
    cl.kill_disk(h, SAFE_MEMBERS - 2)  # the first parity member
  time.sleep(15)
  t0 = time.time()
  res = parallel(lambda h: cl.replace_disk(h, SAFE_MEMBERS - 2), cl.hosts)
  ctx.metrics['parity_rebuild_s'] = round(time.time() - t0, 1)
  failed = {h: str(r)[-400:] for h, r in zip(cl.hosts, res)
            if isinstance(r, Exception) or r[0] != 0}
  ctx.check(not failed, 'parity rebuild with a data member down failed: '
                        f'{failed}')
  for h in cl.hosts:
    cl.kill_disk(h, 2)                 # a second data member: the rebuilt
                                       # parity must now carry the stripe
  th.join(timeout=150 + 1200)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'with two data members dead behind a rebuilt parity disk')
  restart_cluster(ctx, crash=True)
  cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a crash restart with the rebuilt parity seated')


# ---------------------------------------------------------------------------
# A node cut off from the cluster while it stays alive (split brain)
# ---------------------------------------------------------------------------
def _partition_symmetric(cl, victim):
  """Cut `victim` off: it cannot send to any other node and no other node
  can send to it (the CLIO_TEST_PARTITION_FILE hook on every daemon). The
  victim's daemon and FUSE mount stay alive, so its own clients keep
  going against its local containers while the rest of the cluster
  declares it dead and fails over to its successor."""
  vid = cl.hosts.index(victim)  # node ids are the 0-based host indices
  others = [h for h in cl.hosts if h != victim]
  cl.partition(victim, [cl.hosts.index(h) for h in others])
  for h in others:
    cl.partition(h, [vid])


@test('safe_partition_during_writes', 'safe', min_nodes=3,
      redeploy_after=True, timeout=5400)
def t_partition_during_writes(ctx):
  """Split brain under load. Every node writes fsynced record files with a
  data disk dead in every array; node1 is then cut off from the cluster
  for 45 s while it stays up: the others declare it dead and fail over to
  its successor, its own writer keeps hitting its local containers, and
  writes to blobs homed elsewhere fail fast. After the partition heals
  and the writers finish, every fsynced version must read back from
  every node (an acknowledged write on either side of the split may never
  be lost or shadowed by a stale copy), and a crash restart of the whole
  cluster must keep it so."""
  n = len(ctx.hosts)
  cl = ctx.cl
  victim = cl.hosts[1]
  base = ctx.p('sp')
  ctx.ok(0, 'mkdir', path=base)
  for h in cl.hosts:
    cl.kill_disk(h, 0)
  th, replies, logs, nfiles = _writers(ctx, base, 150, 'split')
  time.sleep(20)
  t0 = time.time()
  _partition_symmetric(cl, victim)
  time.sleep(45)
  for h in cl.hosts:
    cl.heal(h)
  ctx.metrics['partition_s'] = round(time.time() - t0, 1)
  th.join(timeout=150 + 1200)
  ctx.metrics['writer_errors'] = _writer_errors(replies, n)
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after a 45 s partition of node1 under degraded writes')
  p = ctx.p('after_split')
  ctx.ok(1, 'write_file', path=p, size=8 << 20, seed=29, fsync=True)
  v = ctx.ok(0, 'verify_file', path=p, size=8 << 20, seed=29)
  ctx.check(v['ok'], f'write on the rejoined node after the split: {v}')
  restart_cluster(ctx, crash=True)
  cl.agents.clear()
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  'after the split healed and a full crash restart')
