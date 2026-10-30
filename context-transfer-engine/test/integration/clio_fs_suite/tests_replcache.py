"""Aimed node faults for the replication and cache chimods (group 'repl').

The node-fault tests in tests_fault.py kill whichever node is clear of a
dataset's directories; whether that node owned a page, held its remote copy
or cached it is luck. These tests use placement.py (the Python twin of the
CTE's BlobHash) to aim each fault at a chosen role:

  owner   the node whose container owns a page (primary copy)
  holder  the next node after the owner, which keeps the page's remote copy
          (replication remote_copies: 1) and serves it while the owner is down
  reader  a node that has read the file, so its cache chimod holds copies

The rule every test checks: an fsync-acknowledged write is never lost to a
single node failure at a time, and no read ever returns bytes older than the
newest acknowledged write. EIO past the tolerance is allowed; old bytes never
are.
"""

import time

from cluster import parallel
from placement import MiB, copy_holders, page_owner
from suite import TestUnsupported, test
from tests_fault import OP_DEADLINE, fs_dir_owner

SIZE = 16 * MiB


def _path_owners(ctx, path):
  """Nodes that must stay up for `path` to be usable: the owners of every
  ancestor directory, and the file's inode home (FsInodeHomeFor hashes the
  file's mount-relative path the same way)."""
  n = len(ctx.hosts)
  rel = path[len(ctx.cl.mnt):]
  parts = [p for p in rel.split('/') if p]
  dirs = ['/'] + ['/' + '/'.join(parts[:k]) for k in range(1, len(parts))]
  return {fs_dir_owner(d, n) for d in dirs} | {fs_dir_owner(rel, n)}


def _ino(ctx, i, path):
  return ctx.ok(i, 'stat', path=path)['ino']


def _stop_node(ctx, i, crash=True):
  """SIGKILL (or stop) node i's daemon and FUSE client."""
  cl = ctx.cl
  h = ctx.hosts[i]
  ag = cl.agents.pop(h, None)
  if ag is not None:
    ag.close()
  if crash:
    cl.kill_fuse(h)
    cl.kill_runtime(h)
  else:
    cl.unmount(h)
    cl.stop_runtime(h)
  time.sleep(3)


def _start_node(ctx, i):
  """Start node i's daemon again and remount; the agent reopens on use."""
  cl = ctx.cl
  h = ctx.hosts[i]
  cl.start_runtime(h)
  ctx.check(cl.runtime_up(h) is True, f'node{i} {h}: runtime did not restart')
  time.sleep(3)
  ctx.check(cl.mount(h) is True, f'node{i} {h}: remount failed')
  cl.agents.pop(h, None)


def _verify(ctx, i, path, size, seed, deadline=OP_DEADLINE):
  """'ok', 'eio' (or another error: the read failed), 'hang', or
  'WRONG ...' (the read succeeded with bytes that are not the newest)."""
  r = ctx.a(i).call('verify_file', timeout=deadline, path=path, size=size,
                    seed=seed)
  if r.get('hang') or r.get('agent_dead'):
    ctx.hang_hosts.add(ctx.hosts[i])
    return 'hang'
  if not r.get('ok'):
    return f'err {r.get("errno")}'
  if r['ret'].get('ok'):
    return 'ok'
  ctx.metrics.setdefault('_first_wrong', (i, path, r['ret']))
  return f'WRONG {r["ret"]}'


def _diagnose_first_wrong(ctx):
  """Record every node's view of the page the first wrong read hit (owner
  vs remote copy vs local cache), and which nodes own / hold it."""
  fw = ctx.metrics.pop('_first_wrong', None)
  if fw is None:
    return
  i, path, verdict = fw
  from tests_fault import blob_info_notes
  try:
    ino = ctx.ok(0, 'stat', path=path)['ino']
    page = ((verdict.get('mismatch') or {}).get('offset', 0)) // MiB
    n = len(ctx.hosts)
    ctx.note(f'first wrong read: node{i} {path.rsplit("/", 1)[-1]} page '
             f'{page}: owner node{page_owner(ino, page, n)}, copy on '
             f'node{copy_holders(ino, page, n)[0]}')
    blob_info_notes(ctx, path, verdict)
  except Exception as e:  # pylint: disable=broad-except
    ctx.note(f'diagnosis failed: {e!r}')


def _pick_file(ctx, root, writer, tag, avoid, want):
  """Create files under root from `writer` until one has a page whose roles
  satisfy want(owner, holder) and stay clear of `avoid` and the file's path
  owners. Returns (path, ino, page, owner, holder)."""
  n = len(ctx.hosts)
  for k in range(24):
    p = f'{root}/{tag}{k}'
    ctx.ok(writer, 'write_file', path=p, size=SIZE, seed=1000 + k, fsync=True,
           timeout=600)
    ino = _ino(ctx, writer, p)
    busy = _path_owners(ctx, p) | set(avoid)
    for page in range(SIZE // MiB):
      o = page_owner(ino, page, n)
      hd = copy_holders(ino, page, n)[0]
      if o not in busy and hd not in busy and want(o, hd):
        return p, ino, page, o, hd, 1000 + k
  raise TestUnsupported(f'no file in 24 tries had a page with the wanted '
                        f'owner/holder roles clear of the path ({n} nodes)')


@test('repl_holder_outage_then_owner_loss', 'repl', min_nodes=4,
      redeploy_after=True, timeout=2400)
def t_holder_outage_then_owner_loss(ctx):
  """One node failure at a time, never two at once. A page's remote-copy
  holder goes down; while it is down the page is overwritten (fsynced) and a
  new file is written whose page has the same owner/holder pair. The holder
  comes back and the cluster gets time to repair. Then the OWNER goes down,
  so every read is served from the holder. The overwritten page must read
  the NEW bytes (the holder's copy is from before its outage; serving it is
  returning stale data as current), and the new file must be readable.

  MirrorRange skips a dead holder ("one copy fewer for now"); this checks
  whether anything brings the copy back up to date when it returns."""
  n = len(ctx.hosts)
  root = ctx.p('ho')
  ctx.ok(0, 'mkdir', path=root)
  path, ino, page, owner, holder, seed0 = _pick_file(
      ctx, root, 0, 'f', avoid={0}, want=lambda o, h: True)
  ctx.note(f'{path.rsplit("/", 1)[-1]} ino {ino}: page {page} owner '
           f'node{owner} holder node{holder}')
  ctx.metrics['owner'] = owner
  ctx.metrics['holder'] = holder

  _stop_node(ctx, holder)
  # Overwrite: the holder's copy of this file is now the old version.
  seed1 = seed0 + 500
  ctx.ok(0, 'write_file', path=path, size=SIZE, seed=seed1, fsync=True,
         timeout=600)
  # A new file whose pages are written with no live holder at all.
  new = None
  for k in range(24):
    p = f'{root}/new{k}'
    if owner in _path_owners(ctx, p) or holder in _path_owners(ctx, p):
      continue
    ctx.ok(0, 'write_file', path=p, size=SIZE, seed=3000 + k, fsync=True,
           timeout=600)
    ino2 = _ino(ctx, 0, p)
    if any(page_owner(ino2, q, n) == owner for q in range(SIZE // MiB)):
      new = (p, 3000 + k)
      break
  _start_node(ctx, holder)
  settle = max(30, 3 * ctx.cl.replicate_period_ms // 1000)
  time.sleep(settle)  # the repair window a real outage would give
  ctx.metrics['settle_s'] = settle
  ctx.check(_verify(ctx, 0, path, SIZE, seed1) == 'ok',
            'the overwrite does not read back before any owner fault')

  _stop_node(ctx, owner)
  readers = [i for i in range(n) if i not in (owner,)]
  res = {}
  for i in readers:
    res[f'node{i}'] = _verify(ctx, i, path, SIZE, seed1)
    if new is not None:
      res[f'node{i}_new'] = _verify(ctx, i, new[0], SIZE, new[1])
  wrong = {k: v for k, v in res.items() if v.startswith('WRONG')}
  lost = {k: v for k, v in res.items() if v not in ('ok',) and k not in wrong}
  ctx.metrics['stale_reads'] = len(wrong)
  ctx.metrics['failed_reads'] = len(lost)
  ctx.check(not wrong, f'reads served the holder\'s pre-outage copy as '
                       f'current after the owner went down: '
                       f'{list(wrong.items())[:3]}')
  ctx.check(not lost, f'fsynced data written while the holder was down is '
                      f'gone after a later, separate owner failure: '
                      f'{list(lost.items())[:3]}')
  _start_node(ctx, owner)
  time.sleep(5)
  bad = {f'node{i}': v for i in range(n)
         for v in [_verify(ctx, i, path, SIZE, seed1)] if v != 'ok'}
  ctx.check(not bad, f'after the owner returned: {bad}')


@test('repl_rolling_single_failures', 'repl', min_nodes=4,
      redeploy_after=True, timeout=5400)
def t_rolling_single_failures(ctx):
  """Every node in turn goes down and comes back while node0 (never killed)
  keeps rewriting a set of files with fsync. Only one node is ever down, so
  replication must keep every file readable throughout and every file must
  hold its last acknowledged version at the end, read from every node.
  This is the sequence a rolling upgrade produces."""
  n = len(ctx.hosts)
  root = ctx.p('rr')
  ctx.ok(0, 'mkdir', path=root)
  files = []
  for k in range(12):
    p = f'{root}/f{k}'
    files.append(p)
  # Every node but the writer goes down in turn, including the homes of the
  # files' directories and inodes: with one failure at a time the successor
  # serves all of it (directory blocks through their replicated CTE mirror,
  # inode records through the copy each home keeps on its successor).
  victims = list(range(1, n))
  ctx.note(f'victims (in order): {victims}')
  latest = {}
  ver = 0
  for p in files:
    ver += 1
    ctx.ok(0, 'write_file', path=p, size=4 * MiB, seed=ver, fsync=True,
           timeout=600)
    latest[p] = ver
  problems = []
  for v in victims:
    _stop_node(ctx, v)
    for p in files:  # rewrite every file while v is down
      ver += 1
      r = ctx.call(0, 'write_file', path=p, size=4 * MiB, seed=ver,
                   fsync=True, timeout=600)
      if r.get('ok'):
        latest[p] = ver
      else:
        problems.append((f'node{v} down', 'write', p, r.get('err')))
    for i in [x for x in range(n) if x != v]:
      for p in files:
        st = _verify(ctx, i, p, 4 * MiB, latest[p])
        if st != 'ok':
          problems.append((f'node{v} down', f'read node{i}', p, st))
    if problems and '_first_wrong' in ctx.metrics:
      # While the fault that produced it is still in place.
      _diagnose_first_wrong(ctx)
    _start_node(ctx, v)
    time.sleep(10)
  for i in range(n):
    for p in files:
      st = _verify(ctx, i, p, 4 * MiB, latest[p])
      if st != 'ok':
        problems.append(('end', f'read node{i}', p, st))
  ctx.metrics['problems'] = len(problems)
  ctx.metrics['victims'] = len(victims)
  _diagnose_first_wrong(ctx)
  wrong = [x for x in problems if str(x[-1]).startswith('WRONG')]
  ctx.check(not wrong, f'{len(wrong)} reads returned old bytes: {wrong[:3]}')
  ctx.check(not problems, f'{len(problems)} failures with only one node '
                          f'down at a time: {problems[:4]}')


def _cache_setup(ctx, tag):
  """A file written by node0 and read in full by a reader node that owns
  none of its path, so the reader's cache holds copies of remote pages.
  Returns (path, reader, seed)."""
  n = len(ctx.hosts)
  root = ctx.p(tag)
  ctx.ok(0, 'mkdir', path=root)
  for k in range(24):
    p = f'{root}/c{k}'
    busy = _path_owners(ctx, p) | {0}
    readers = [i for i in range(n - 1, 0, -1) if i not in busy]
    if not readers:
      continue
    ctx.ok(0, 'write_file', path=p, size=SIZE, seed=50 + k, fsync=True,
           timeout=600)
    r = readers[0]
    ino = _ino(ctx, 0, p)
    remote = sum(1 for q in range(SIZE // MiB)
                 if page_owner(ino, q, n) != r)
    ctx.note(f'reader node{r} caches {remote}/{SIZE // MiB} remote pages')
    for _ in range(2):  # twice: the second read is the one served cached
      ctx.check(_verify(ctx, r, p, SIZE, 50 + k) == 'ok',
                f'node{r} cannot read the fresh file')
    return p, r, 50 + k
  raise TestUnsupported('no file path left a reader node clear of it')


@test('cache_holder_crash_misses_overwrite', 'repl', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_cache_holder_crash(ctx):
  """A node with cached copies of a file crashes; while it is down the file
  is overwritten (fsynced) from node0, so the invalidation cannot reach it.
  When it comes back it must read the new bytes, not whatever its cache tier
  kept across the crash."""
  path, reader, seed = _cache_setup(ctx, 'cc')
  _stop_node(ctx, reader)
  ctx.ok(0, 'write_file', path=path, size=SIZE, seed=seed + 7, fsync=True,
         timeout=600)
  _start_node(ctx, reader)
  st = _verify(ctx, reader, path, SIZE, seed + 7)
  ctx.metrics['reader_after_restart'] = st
  ctx.check(st == 'ok', f'node{reader} after restarting: {st} (wanted the '
                        f'overwrite)')


@test('cache_holder_partitioned_misses_overwrite', 'repl', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_cache_holder_partitioned(ctx):
  """A node with cached copies is cut off from every peer (alive, so it
  keeps its cache, but no invalidation can reach it) while node0 overwrites
  the file with fsync. During the split the reader may fail or block within
  the deadline, but it must never return the old bytes once the split heals
  and must converge to the new bytes."""
  n = len(ctx.hosts)
  cl = ctx.cl
  path, reader, seed = _cache_setup(ctx, 'cp')
  rh = ctx.hosts[reader]
  others = [i for i in range(n) if i != reader]
  cl.partition(rh, others)
  for i in others:
    cl.partition(ctx.hosts[i], [reader])
  # The cut-off node owns some of the file's pages; until the others have
  # declared it dead (3 missed liveness probes, ~10 s) writes to those pages
  # have nowhere to go and fail EIO. Wait for failover, then retry briefly.
  time.sleep(15)
  try:
    t0 = time.time()
    w = ctx.call(0, 'write_file', path=path, size=SIZE, seed=seed + 9,
                 fsync=True, timeout=600)
    while not w.get('ok') and time.time() - t0 < 60:
      time.sleep(5)
      w = ctx.call(0, 'write_file', path=path, size=SIZE, seed=seed + 9,
                   fsync=True, timeout=600)
    ctx.metrics['overwrite_after_split_s'] = round(time.time() - t0, 1)
    ctx.check(w.get('ok'), f'node0 overwrite during the split failed 75 s '
                           f'after it began: {w.get("err")}')
    during = _verify(ctx, reader, path, SIZE, seed + 9)
    ctx.metrics['reader_during_split'] = during
    if during == 'ok':
      ctx.note('reader read the new bytes during the split')
    elif during.startswith('WRONG'):
      ctx.note(f'reader during the split: {during} (cut off; checked again '
               f'after heal)')
  finally:
    parallel(cl.heal, cl.hosts)
  t0 = time.time()
  st = _verify(ctx, reader, path, SIZE, seed + 9)
  while st != 'ok' and time.time() - t0 < 30:
    time.sleep(2)
    st = _verify(ctx, reader, path, SIZE, seed + 9)
  ctx.metrics['converge_s'] = round(time.time() - t0, 1)
  if st != 'ok':
    _diagnose_first_wrong(ctx)
  ctx.check(st == 'ok', f'node{reader} 30 s after the split healed: {st}')
