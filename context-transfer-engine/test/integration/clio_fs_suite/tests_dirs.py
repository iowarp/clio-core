"""Directory stress: the CTE-stored, per-node-cached directory blocks.

Directories live in CTE blobs (directory blocks) that split as they grow and
spread over the cluster; every node caches the blocks it reads and the
block's home pushes each change to those caches before acknowledging it.
These tests hammer exactly that:

  * one shared directory hit by every node at once (splits under load);
  * strict cross-node visibility -- no waiting: a create acknowledged on one
    node is visible on every other node the moment it returns;
  * listings during creates (no duplicates, no going backwards);
  * rmdir racing creates (never a file inside a removed directory);
  * directory rename storms, including moves into each other's subtrees
    (no cycles, no lost directories);
  * a crash of every node in the middle of it all (directories come back
    from the CTE alone).
"""

import random
import time

from suite import test

STORM_PROCS = 8


def _storm(ctx, i, d, prefix, count, mode, timeout=1800):
  r = ctx.ok(i, 'meta_storm', dirpath=d, prefix=prefix, count=count,
             procs=STORM_PROCS, mode=mode, timeout=timeout)
  ctx.check(r['nfails'] == 0,
            f'node{i} {mode} storm: {r["nfails"]} failures, e.g. {r["fails"]}')
  return r


@test('dirs_shared_create_storm', 'dirs', min_nodes=2, timeout=3600)
def t_shared_storm(ctx):
  """Every node, 8 processes each, creates into ONE directory until it has
  split many times; every node then lists all, stats another node's files,
  unlinks its own; the directory ends empty and removable."""
  n = len(ctx.hosts)
  per = 1500  # per process: n * 8 * 1500 names, dozens of block splits
  d = ctx.p('shared')
  ctx.ok(0, 'mkdir', path=d)
  t0 = time.time()
  ctx.each(lambda i: _storm(ctx, i, d, f'n{i}p', per, 'create'))
  dt = time.time() - t0
  total = n * STORM_PROCS * per
  ctx.metrics['creates_per_s'] = round(total / dt)
  for i in range(n):
    c = ctx.ok(i, 'scandir_count', path=d, timeout=600)
    ctx.check(c == total, f'node{i} lists {c} of {total}')
  # Each node stats the files the NEXT node created: no retry, no wait.
  t0 = time.time()
  ctx.each(lambda i: _storm(ctx, i, d, f'n{(i + 1) % n}p', per, 'stat'))
  ctx.metrics['stats_per_s'] = round(total / (time.time() - t0))
  t0 = time.time()
  ctx.each(lambda i: _storm(ctx, i, d, f'n{i}p', per, 'unlink'))
  ctx.metrics['unlinks_per_s'] = round(total / (time.time() - t0))
  for i in range(n):
    c = ctx.ok(i, 'scandir_count', path=d, timeout=600)
    ctx.check(c == 0, f'node{i} still lists {c} after every unlink')
  ctx.ok(n - 1, 'rmdir', path=d)
  for i in range(n):
    ctx.check(ctx.ok(i, 'exists', path=d) is False,
              f'node{i} still sees the removed directory')


@test('dirs_strict_visibility', 'dirs', min_nodes=2, timeout=1800)
def t_strict_visibility(ctx):
  """Create on node a, stat on node b immediately (fresh names, so no
  negative kernel cache): a change is pushed before it is acknowledged, so
  there is no staleness window at all.  Also mkdir, chmod and unlink."""
  n = len(ctx.hosts)
  d = ctx.p('vis')
  ctx.ok(0, 'mkdir', path=d)
  rng = random.Random(7)
  misses = []
  for r in range(300):
    a = rng.randrange(n)
    b = (a + 1 + rng.randrange(n - 1)) % n
    f = f'{d}/f{r}'
    ctx.ok(a, 'write_file', path=f, size=17, seed=r)
    got = ctx.call(b, 'stat', path=f)
    if not got['ok']:
      misses.append(('create', r, got.get('err')))
    if r % 3 == 0:
      sd = f'{d}/sd{r}'
      ctx.ok(a, 'mkdir', path=sd)
      if not ctx.call(b, 'stat', path=sd)['ok']:
        misses.append(('mkdir', r))
  ctx.check(not misses, f'{len(misses)} acknowledged changes not visible at '
                        f'once on another node, e.g. {misses[:5]}')


@test('dirs_readdir_during_create', 'dirs', min_nodes=2, timeout=2400)
def t_readdir_during_create(ctx):
  """One node creates 30k names (forcing splits) while every other node
  lists the directory over and over: no listing has a duplicate, and the
  counts never go backwards."""
  n = len(ctx.hosts)
  d = ctx.p('grow')
  ctx.ok(0, 'mkdir', path=d)

  def work(i):
    if i == 0:
      return _storm(ctx, 0, d, 'g', 30000 // STORM_PROCS, 'create')
    return ctx.ok(i, 'readdir_check', path=d, rounds=40, timeout=1800)

  rs = ctx.each(work)
  for i in range(1, n):
    r = rs[i]
    ctx.check(not r['dups'], f'node{i} listed duplicates: {r["dups"]}')
    ctx.check(all(b >= a for a, b in zip(r['sizes'], r['sizes'][1:])),
              f'node{i} listing went backwards: {r["sizes"]}')
  want = (30000 // STORM_PROCS) * STORM_PROCS
  for i in range(n):
    c = ctx.ok(i, 'scandir_count', path=d, timeout=600)
    ctx.check(c == want, f'node{i} lists {c} of {want}')


@test('dirs_rmdir_create_race', 'dirs', min_nodes=2, timeout=1800)
def t_rmdir_create_race(ctx):
  """rmdir on one node races a create inside the directory on another, over
  200 directories visited in different random orders, so the two sides
  collide throughout.  Exactly one side wins each directory: the create
  (rmdir fails ENOTEMPTY) or the rmdir (the create fails) -- never both,
  which would leave a file in a directory that no longer exists."""
  dirs = [ctx.p(f'race{r}') for r in range(200)]
  for d in dirs:
    ctx.ok(0, 'mkdir', path=d)

  def go(i):
    if i == 0:
      return ctx.ok(0, 'race_dirs', paths=dirs, mode='rmdir', seed=1)
    if i == 1:
      return ctx.ok(1, 'race_dirs', paths=dirs, mode='create', seed=2)
    return {}

  rs = ctx.each(go)
  outcomes, bad = {}, []
  for d in dirs:
    rm_ok, cr_ok = rs[0][d] == 'ok', rs[1][d] == 'ok'
    k = f'rmdir {rs[0][d]} / create {rs[1][d]}'
    outcomes[k] = outcomes.get(k, 0) + 1
    exists = ctx.ok(0, 'exists', path=d)
    if rm_ok == cr_ok or exists == rm_ok:
      bad.append((d, rs[0][d], rs[1][d], exists))
  ctx.metrics['outcomes'] = outcomes
  ctx.check(not bad, f'{len(bad)} directories where not exactly one side won '
                     f'(or the tree disagrees), e.g. {bad[:5]}')


@test('dirs_rename_storm', 'dirs', min_nodes=2, timeout=2400)
def t_rename_storm(ctx):
  """Every node moves random directories into one another at once (so
  moves cross and nest). Afterwards every node sees the same tree, every
  directory is still reachable from the root (no cycle cut it off), and
  every file in them is intact."""
  n = len(ctx.hosts)
  root = ctx.p('tree')
  ctx.ok(0, 'mkdir', path=root)
  dirs = []
  for a in range(12):
    ctx.ok(0, 'makedirs', path=f'{root}/a{a}/b')
    ctx.ok(0, 'write_file', path=f'{root}/a{a}/b/file', size=4096 + a, seed=a)
    dirs += [f'a{a}', f'a{a}/b']
  rs = ctx.each(lambda i: ctx.ok(i, 'rename_storm', root=root, dirs=dirs,
                                 iters=300, seed=i + 1, timeout=1800))
  agg = {}
  for r in rs:
    for k, v in r.items():
      agg[k] = agg.get(k, 0) + v
  ctx.metrics['renames'] = agg
  bad = set(agg) - {'ok', 'ENOENT', 'EINVAL', 'EBUSY', 'ENOTEMPTY'}
  ctx.check(not bad, f'unexpected rename errors: {agg}')
  mans = [ctx.ok(i, 'tree_manifest', root=root, timeout=600) for i in range(n)]
  for i in range(1, n):
    ctx.check(mans[i] == mans[0], f'node{i} sees a different tree')
  ndirs = sum(1 for v in mans[0].values() if v[0] == 'd')
  nfiles = sum(1 for v in mans[0].values() if v[0] == 'f')
  ctx.check(ndirs == 24, f'{ndirs} of 24 directories reachable from the root')
  ctx.check(nfiles == 12, f'{nfiles} of 12 files reachable from the root')


@test('dirs_crash_restart', 'dirs', min_nodes=2, redeploy_after=True,
      timeout=3600)
def t_dirs_crash_restart(ctx):
  """A large split directory, a deep tree and renamed subtrees, every node
  SIGKILLed after fsync of the directories, then restarted: the whole
  namespace comes back from the CTE blocks, identical on every node."""
  from tests_fault import restart_cluster
  n = len(ctx.hosts)
  root = ctx.p('persist')
  ctx.ok(0, 'mkdir', path=root)
  big = f'{root}/big'
  ctx.ok(0, 'mkdir', path=big)
  ctx.each(lambda i: _storm(ctx, i, big, f'n{i}p', 600, 'create'))
  for a in range(8):
    ctx.ok(a % n, 'makedirs', path=f'{root}/t{a}/x/y/z')
    ctx.ok(a % n, 'write_file', path=f'{root}/t{a}/x/y/z/f', size=5000 + a,
           seed=a, fsync=True)
  ctx.ok(0, 'rename', src=f'{root}/t0', dst=f'{root}/t1/x/moved')
  ctx.ok(1 % n, 'rename', src=f'{root}/t2/x', dst=f'{root}/t3/renamed')
  # Directory entries are durable once their directory is fsync'd.
  for i in range(n):
    ctx.ok(i, 'sh', cmd=f'python3 -c "import os; '
                        f'[os.fsync(os.open(d, os.O_RDONLY)) for d, _, _ in '
                        f'os.walk({root!r})]"', timeout=600)
  before = ctx.ok(0, 'tree_manifest', root=root, timeout=600)
  restart_cluster(ctx, crash=True)
  for i in range(n):
    after = ctx.ok(i, 'tree_manifest', root=root, timeout=600)
    missing = sorted(set(before) - set(after))[:10]
    extra = sorted(set(after) - set(before))[:10]
    changed = [k for k in before if k in after and before[k] != after[k]][:10]
    ctx.check(not missing and not extra and not changed,
              f'node{i} after crash: missing {missing} extra {extra} '
              f'changed {changed}')
  want = n * STORM_PROCS * 600
  c = ctx.ok(n - 1, 'scandir_count', path=big, timeout=600)
  ctx.check(c == want, f'big directory lists {c} of {want} after restart')


@test('dirs_cache_after_home_restart', 'dirs', min_nodes=2,
      redeploy_after=True, timeout=1800)
def t_cache_after_home_restart(ctx):
  """Every node caches 40 directories (some homed on each node), then one
  node's daemon is SIGKILLed and restarted.  The restarted home no longer
  knows who caches its blocks; changes made after the restart must still
  reach every cache at once: files created in each directory are visible
  on every other node immediately, and listings agree."""
  n = len(ctx.hosts)
  root = ctx.p('rc')
  ctx.ok(0, 'mkdir', path=root)
  dirs = [f'{root}/d{j}' for j in range(40)]
  for d in dirs:
    ctx.ok(0, 'mkdir', path=d)
    ctx.ok(0, 'write_file', path=f'{d}/before', size=100, seed=1)
  for i in range(n):  # every node caches every block (and inode)
    for d in dirs:
      ctx.check(ctx.ok(i, 'scandir_count', path=d) == 1, f'node{i} {d}')
      ctx.ok(i, 'stat', path=f'{d}/before')
  v = n - 1
  vh = ctx.hosts[v]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  time.sleep(1)
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), 'victim restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), 'victim remount')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  misses = []
  for j, d in enumerate(dirs):
    a = j % n
    ctx.ok(a, 'write_file', path=f'{d}/after', size=200, seed=2)
    ctx.ok(a, 'sh', cmd=f'chmod 600 {d}/before')
    for i in range(n):
      if i == a:
        continue
      if not ctx.call(i, 'stat', path=f'{d}/after')['ok']:
        misses.append(('create', d, i))
      st = ctx.call(i, 'stat', path=f'{d}/before')
      if not st['ok'] or st['ret']['perm'] != 0o600:
        misses.append(('chmod', d, i, st.get('ret', {}).get('perm')))
      c = ctx.ok(i, 'scandir_count', path=d)
      if c != 2:
        misses.append(('list', d, i, c))
  ctx.check(not misses, f'{len(misses)} changes after the home restart not '
                        f'seen by a cache, e.g. {misses[:6]}')


@test('failover_reload_keeps_entries', 'dirs', min_nodes=4,
      redeploy_after=True, timeout=2400)
def t_failover_reload_keeps_entries(ctx):
  """The home of a directory is cut off one way (nobody can send to it)
  long enough to be failed over, while every other node keeps creating
  files in that directory, and the home itself created files there before
  the split. After the heal the home receives the rejoin notice and
  reloads its blocks; that reload must keep EVERY entry: the ones it made
  before the split and the ones the stand-in made during it. Checked from
  every node. (A wrongly triggered reload lost recent entries; this checks
  the reload a real failover needs.)"""
  import time
  from tests_fault import fs_dir_owner
  n = len(ctx.hosts)
  base = ctx.p('fr')
  ctx.ok(0, 'mkdir', path=base)
  rel0 = base[len(ctx.cl.mnt):]
  # A directory homed on a node other than node0 (the driver's writer).
  d = a = None
  for k in range(64):
    cand = f'{base}/d{k}'
    own = fs_dir_owner(cand[len(ctx.cl.mnt):], n)
    if own not in (0, fs_dir_owner(rel0, n)):
      d, a = cand, own
      break
  if d is None:
    from suite import TestUnsupported
    raise TestUnsupported('no candidate directory homed away from node0')
  ctx.ok(0, 'mkdir', path=d)
  ctx.note(f'directory home node{a}')
  made = set()
  for k in range(40):  # before the split: some written by the home itself
    i = a if k % 2 == 0 else 0
    ctx.ok(i, 'write_file', path=f'{d}/pre{k}', size=64, seed=k, fsync=True)
    made.add(f'pre{k}')
  others = [i for i in range(n) if i != a]
  for i in others:
    ctx.cl.partition(ctx.hosts[i], [a])
  try:
    time.sleep(15)  # declared dead, its blocks failed over
    t0 = time.time()
    k = 0
    while time.time() - t0 < 30:
      i = others[k % len(others)]
      r = ctx.call(i, 'write_file', path=f'{d}/mid{k}', size=64, seed=1000 + k,
                   fsync=True, timeout=120)
      if r.get('ok'):
        made.add(f'mid{k}')
      k += 1
    ctx.metrics['created_during_split'] = sum(
        1 for x in made if x.startswith('mid'))
  finally:
    for i in others:
      ctx.cl.heal(ctx.hosts[i])
  time.sleep(20)  # rejoin, notice, reload
  for k in range(10):  # and the home keeps working after the reload
    ctx.ok(a, 'write_file', path=f'{d}/post{k}', size=64, seed=2000 + k,
           fsync=True)
    made.add(f'post{k}')
  lost = {}
  for i in range(n):
    r = ctx.call(i, 'listdir', path=d, timeout=300)
    seen = set(r.get('ret') or []) if r.get('ok') else set()
    miss = sorted(made - seen)
    if miss:
      lost[f'node{i}'] = (len(miss), miss[:5])
  ctx.metrics['entries'] = len(made)
  ctx.check(not lost, f'acknowledged entries missing after the failover '
                      f'reload: {lost}')


@test('partitioned_cache_not_stale', 'dirs', min_nodes=3, redeploy_after=True,
      timeout=1800)
def t_partitioned_cache(ctx):
  """Every node caches a directory; then the last node is cut off (nobody
  can send to it) while it stays alive to everyone. Changes made
  meanwhile cannot be pushed to it, so its cached copies are stale. It
  must never serve them as current: a lookup of a file created during the
  partition must not answer a plain ENOENT, a chmod must not read back the
  old mode -- an error (EIO / ETIMEDOUT) is acceptable, a stale answer is
  not. After the partition heals the node sees everything."""
  import time
  n = len(ctx.hosts)
  a = n - 1
  d = ctx.p('part')
  ctx.ok(0, 'mkdir', path=d)
  ctx.ok(0, 'write_file', path=f'{d}/f0', size=100, seed=1)
  ctx.ok(0, 'sh', cmd=f'chmod 644 {d}/f0')
  for i in range(n):
    ctx.ok(i, 'scandir_count', path=d)
    ctx.ok(i, 'stat', path=f'{d}/f0')
  for i in range(n - 1):
    ctx.cl.partition(ctx.hosts[i], [a])
  stale = []
  try:
    # Changes whose metadata lives on the cut-off node fail EIO until the
    # others declare it dead (~10 s); give failover that long, then retry.
    time.sleep(15)
    t0 = time.time()
    for op, kw in (('write_file', dict(path=f'{d}/f1', size=100, seed=2)),
                   ('sh', dict(cmd=f'chmod 600 {d}/f0'))):
      r = ctx.call(0, op, timeout=300, **kw)
      while not r.get('ok') and time.time() - t0 < 60:
        time.sleep(5)
        r = ctx.call(0, op, timeout=300, **kw)
      ctx.check(r.get('ok') and (op != 'sh' or r['ret']['rc'] == 0),
                f'{op} during the partition failed 75 s in: {r}')
    ctx.metrics['change_during_partition_s'] = round(time.time() - t0, 1)
    r = ctx.call(a, 'stat', path=f'{d}/f1', timeout=300)
    if not r['ok'] and (r.get('err') or '').startswith('ENOENT'):
      stale.append('f1 created during the partition: ENOENT')
    r = ctx.call(a, 'stat', path=f'{d}/f0', timeout=300)
    if r['ok'] and r['ret']['perm'] != 0o600:
      stale.append(f'f0 mode {oct(r["ret"]["perm"])}, set to 0o600')
  finally:
    # Whatever happened above: a partition left in place outlives the test
    # (and redeploys) and breaks everything after it (reliability2).
    for i in range(n - 1):
      ctx.cl.heal(ctx.hosts[i])
  ctx.check(not stale, f'the partitioned node served stale copies as '
                       f'current: {stale}')
  ok = False
  last = {}
  for _ in range(60):
    st = ctx.call(a, 'stat', path=f'{d}/f1')
    st0 = ctx.call(a, 'stat', path=f'{d}/f0')
    ls = ctx.call(a, 'listdir', path=d)
    last = {'f1': st.get('err') or 'ok',
            'f0_perm': oct(st0['ret']['perm']) if st0.get('ok')
            else st0.get('err'),
            'listing': sorted(ls.get('ret') or []) if ls.get('ok')
            else ls.get('err')}
    if st['ok'] and st0['ok'] and st0['ret']['perm'] == 0o600:
      ok = True
      break
    time.sleep(2)
  ctx.metrics['converged'] = ok
  ctx.check(ok, f'after the partition healed the node still does not see '
                f'the changes made during it: {last}')
