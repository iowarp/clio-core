"""Failure / restart tests: persistence and behaviour under node loss.

What a persistent, reliable deployment must guarantee (and what is checked):
  * Everything acknowledged by fsync()/close() before a graceful stop or a
    crash (SIGKILL of every daemon) is back, byte-exact, with its metadata
    (names, sizes, modes, links, xattrs) after a stop and `clio_run start`.
  * Losing one node never hangs the survivors: every operation on a
    survivor returns (success or an error) inside a bounded time.
  * A node that comes back rejoins and its data is readable cluster-wide.
  * A crashed FUSE client does not take the daemon down and can remount.
Each test leaves the cluster redeployed from scratch (redeploy_after).
"""

import random
import time

from cluster import parallel, sh
from suite import test
from tests_dist import _tags_match, check_records, runs_to_set

MiB = 1 << 20
OP_DEADLINE = 60  # seconds any single op may take while a node is down


def fs_dir_owner(path, n):
  """Container (= node index) owning directory `path` (mount-relative) among
  n nodes; mirrors FsPathHash / FsDirContainer in fs_shard.h."""
  m = (1 << 64) - 1
  h = 0xCBF29CE484222325
  for c in path.encode():
    h = ((h ^ c) * 0x100000001B3) & m
  h = (h + 0x9E3779B97F4A7C15) & m
  h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & m
  h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & m
  h ^= h >> 31
  return ((h ^ (h >> 32)) & 0xffffffff) % n


def victim_off_path(ctx, root):
  """Highest node index owning none of root's ancestor directories, so the
  loss hits a shard of the dataset instead of the path every entry is under
  (that case is the known no-metadata-replica limit, not what is measured).
  Falls back to the last node."""
  n = len(ctx.hosts)
  rel = root[len(ctx.cl.mnt):]
  parts = [p for p in rel.split('/') if p]
  dirs = ['/'] + ['/' + '/'.join(parts[:k]) for k in range(1, len(parts) + 1)]
  owners = {fs_dir_owner(d, n) for d in dirs}
  for i in range(n - 1, 0, -1):
    if i not in owners:
      return i
  return n - 1


def build_dataset(ctx, tag, per_node=12):
  """Populate a mixed tree from every node; return the expected manifest.

  The manifest maps relpath -> (kind, size, perm, seed/target[, xattrs])
  (xattrs: name -> hex value, checked by audit) and is what
  every node must see after the fault + recovery.
  """
  n = len(ctx.hosts)
  root = ctx.p(tag)
  ctx.ok(0, 'mkdir', path=root)
  exp = {}

  def fill(i):
    rng = random.Random(hash((tag, i)) & 0xffffffff)
    d = f'{root}/node{i}'
    ctx.ok(i, 'makedirs', path=f'{d}/sub/deeper')
    local = {f'node{i}': ('d',), f'node{i}/sub': ('d',),
             f'node{i}/sub/deeper': ('d',)}
    for k in range(per_node):
      size = rng.choice([0, 1, 4095, 65536, MiB, 3 * MiB + 11, 9 * MiB])
      rel = f'node{i}/' + rng.choice(['', 'sub/', 'sub/deeper/']) + f'f{k}'
      seed = i * 10000 + k
      h = ctx.ok(i, 'open', path=f'{root}/{rel}', flags='wct', timeout=120)
      if size:
        ctx.ok(i, 'fpwrite', h=h, off=0, length=size, seed=seed, timeout=300)
      ctx.ok(i, 'fsync', h=h, timeout=120)
      ctx.ok(i, 'close', h=h)
      mode = rng.choice([0o644, 0o600, 0o755])
      ctx.ok(i, 'chmod', path=f'{root}/{rel}', mode=mode)
      local[rel] = ('f', size, mode, seed)
    ctx.ok(i, 'symlink', target='f0', path=f'{d}/link_to_f0')
    local[f'node{i}/link_to_f0'] = ('l', 'f0')
    xv = f'{tag}{i}'.encode().hex()
    ctx.ok(i, 'setxattr', path=f'{d}/sub', name='user.tag', value_hex=xv)
    local[f'node{i}/sub'] = ('d', {'user.tag': xv})
    # A file's xattr too (the manifest's optional last field: name -> hex).
    frel = next((k for k in sorted(local) if local[k][0] == 'f'), None)
    if frel is not None:
      fv = f'file-{tag}{i}'.encode().hex()
      ctx.ok(i, 'setxattr', path=f'{root}/{frel}', name='user.kind',
             value_hex=fv)
      local[frel] = local[frel] + ({'user.kind': fv},)
    return local
  for local in ctx.each(fill):
    exp.update(local)
  # Rename a file across node dirs so metadata of one node references
  # another node's data (exercises the cross-node rename path + recovery).
  cands = [k for k in sorted(exp) if k.startswith('node0/') and
           exp[k][0] == 'f']
  if n > 1 and cands:
    src = cands[0]
    dst = f'node{n - 1}/moved_from_node0'
    ctx.ok(n - 1, 'rename', src=f'{root}/{src}', dst=f'{root}/{dst}')
    exp[dst] = exp.pop(src)
  return root, exp


def audit(ctx, i, root, exp, deadline=OP_DEADLINE, subset=None):
  """Check node i sees exp under root.  Returns (bad list, hang count)."""
  bad = []
  hangs = 0
  keys = subset if subset is not None else sorted(exp)
  for rel in keys:
    e = exp[rel]
    p = f'{root}/{rel}'
    try:
      if e[0] == 'f':
        r = ctx.a(i).call('verify_file', timeout=deadline, path=p,
                          size=e[1], seed=e[3])
        if r.get('hang'):
          hangs += 1
          bad.append((rel, 'HANG'))
          if hangs >= 3:
            break
          continue
        if not r['ok']:
          bad.append((rel, r['err']))
        elif not r['ret']['ok']:
          bad.append((rel, r['ret']))
        else:
          s = ctx.a(i).call('stat', timeout=deadline, path=p)
          if s['ok'] and s['ret']['perm'] != e[2]:
            bad.append((rel, f'mode {oct(s["ret"]["perm"])} != {oct(e[2])}'))
      elif e[0] == 'd':
        r = ctx.a(i).call('stat', timeout=deadline, path=p)
        if r.get('hang'):
          hangs += 1
          bad.append((rel, 'HANG'))
        elif not r['ok'] or not r['ret']['isdir']:
          bad.append((rel, r.get('err') or 'not a dir'))
      elif e[0] == 'l':
        r = ctx.a(i).call('readlink', timeout=deadline, path=p)
        if r.get('hang'):
          hangs += 1
          bad.append((rel, 'HANG'))
        elif not r['ok'] or r['ret'] != e[1]:
          bad.append((rel, r.get('err') or r.get('ret')))
      want_x = e[-1] if isinstance(e[-1], dict) else {}
      for name, val in want_x.items():
        r = ctx.a(i).call('getxattr', timeout=deadline, path=p, name=name)
        if r.get('hang'):
          hangs += 1
          bad.append((rel, 'HANG'))
        elif not r['ok'] or r['ret'] != val:
          bad.append((rel, f'xattr {name}: {r.get("err") or r.get("ret")}, '
                           f'want {val}'))
    except Exception as ex:  # pylint: disable=broad-except
      bad.append((rel, repr(ex)))
  return bad, hangs


def audit_all(ctx, root, exp, what):
  """Every node must see the complete dataset."""
  res = ctx.each(lambda i: audit(ctx, i, root, exp))
  msgs = []
  for i, (bad, hangs) in enumerate(res):
    if bad:
      msgs.append(f'node{i}: {len(bad)}/{len(exp)} bad (hangs={hangs}) '
                  f'e.g. {bad[:4]}')
  ctx.check(not msgs, f'{what}: ' + ' | '.join(msgs))


def restart_cluster(ctx, crash):
  """Stop (gracefully or SIGKILL) every daemon and bring it back with
  `clio_run start` (which recovers), then remount."""
  cl = ctx.cl
  cl.close_agents()
  if crash:
    parallel(lambda h: cl.kill_fuse(h), cl.hosts)
    parallel(lambda h: cl.kill_runtime(h), cl.hosts)
  else:
    parallel(cl.unmount, cl.hosts)
    parallel(cl.stop_runtime, cl.hosts)
  time.sleep(2)
  t0 = time.time()
  parallel(lambda h: cl.start_runtime(h), cl.hosts)
  ups = parallel(cl.runtime_up, cl.hosts)
  ctx.check(all(u is True for u in ups), f'runtime restart failed: {ups}')
  time.sleep(3)
  ms = parallel(cl.mount, cl.hosts)
  ctx.check(all(m is True for m in ms), f'remount failed: {ms}')
  ctx.metrics['restart_s'] = round(time.time() - t0, 1)


@test('graceful_restart_all', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_graceful(ctx):
  """Graceful stop of every daemon + `clio_run start`: all data back."""
  root, exp = build_dataset(ctx, 'g')
  audit_all(ctx, root, exp, 'before restart')
  restart_cluster(ctx, crash=False)
  audit_all(ctx, root, exp, 'after graceful restart')
  # And the filesystem is writable again, in the same namespace.
  ctx.ok(0, 'write_file', path=f'{root}/after', size=MiB, seed=5)
  ctx.check(ctx.ok(len(ctx.hosts) - 1, 'verify_file', path=f'{root}/after',
                   size=MiB, seed=5)['ok'], 'write after restart')


@test('crash_restart_all', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_crash(ctx):
  """SIGKILL every daemon and FUSE client after fsync; restart; all data
  acknowledged by fsync must be back."""
  root, exp = build_dataset(ctx, 'c')
  restart_cluster(ctx, crash=True)
  audit_all(ctx, root, exp, 'after crash restart')


@test('double_crash_restart', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_double_crash(ctx):
  """Crash, restart, write more, crash again: both generations survive
  (the WAL must be appendable after a replay)."""
  root1, exp1 = build_dataset(ctx, 'd1', per_node=6)
  restart_cluster(ctx, crash=True)
  root2, exp2 = build_dataset(ctx, 'd2', per_node=6)
  restart_cluster(ctx, crash=True)
  audit_all(ctx, root1, exp1, 'generation 1 after 2 crashes')
  audit_all(ctx, root2, exp2, 'generation 2 after crash')


@test('plain_start_keeps_data', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_plain_start(ctx):
  """Operator stops the deployment and runs `clio_run start` again: a
  persistent deployment keeps all its data (start recovers by default;
  only --fresh discards)."""
  root, exp = build_dataset(ctx, 's', per_node=4)
  cl = ctx.cl
  cl.close_agents()
  parallel(cl.unmount, cl.hosts)
  parallel(cl.stop_runtime, cl.hosts)
  parallel(lambda h: cl.start_runtime(h), cl.hosts)
  parallel(cl.runtime_up, cl.hosts)
  time.sleep(3)
  parallel(cl.mount, cl.hosts)
  bad, _ = audit(ctx, 0, root, exp)
  ctx.check(not bad, f'`clio_run start` on an existing persistent '
                     f'deployment lost {len(bad)}/{len(exp)} entries '
                     f'(e.g. {bad[:3]})')


@test('fresh_start_discards', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_fresh_start(ctx):
  """`clio_run start --fresh` is the one way to discard state: after it the
  old dataset is gone -- and STAYS gone across a later (recovering) start,
  i.e. nothing of the previous run's logs is replayed -- while the
  filesystem works normally."""
  root, exp = build_dataset(ctx, 'f', per_node=3)
  cl = ctx.cl

  def cycle(fresh):
    cl.close_agents()
    parallel(cl.unmount, cl.hosts)
    parallel(cl.stop_runtime, cl.hosts)
    parallel(lambda h: cl.start_runtime(h, fresh=fresh), cl.hosts)
    ups = parallel(cl.runtime_up, cl.hosts)
    ctx.check(all(u is True for u in ups), f'start (fresh={fresh}) failed')
    time.sleep(3)
    ms = parallel(cl.mount, cl.hosts)
    ctx.check(all(m is True for m in ms), f'remount (fresh={fresh}) failed')

  cycle(fresh=True)
  ctx.check(ctx.ok(0, 'exists', path=root) is False,
            'the dataset survived `start --fresh`')
  # --fresh discarded this test's own directory too: make it again.
  ctx.ok(0, 'makedirs', path=ctx.dir)
  p = ctx.p('after_fresh')
  ctx.ok(0, 'write_file', path=p, size=1 << 20, seed=9, fsync=True)
  cycle(fresh=False)
  for i in range(len(ctx.hosts)):
    ctx.check(ctx.ok(i, 'exists', path=root) is False,
              f'node{i}: the pre-fresh dataset came back after a later start')
    v = ctx.ok(i, 'verify_file', path=p, size=1 << 20, seed=9)
    ctx.check(v['size_ok'] and not v['mismatch'],
              f'node{i}: the file written after --fresh was not recovered')


@test('node_loss_survivors_bounded', 'fault', min_nodes=2,
      redeploy_after=True, timeout=3600)
def t_node_loss(ctx):
  """SIGKILL one node's daemon.  Survivors: no op may hang (all return
  within OP_DEADLINE), their own new I/O keeps working; after the node
  restarts every node sees the whole dataset again."""
  n = len(ctx.hosts)
  root, exp = build_dataset(ctx, 'n', per_node=8)
  victim = n - 1
  vh = ctx.hosts[victim]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  t_kill = time.time()
  time.sleep(2)
  survivors = list(range(n - 1))
  hang_total = 0
  unreadable = 0
  for i in survivors:
    bad, hangs = audit(ctx, i, root, exp, deadline=OP_DEADLINE)
    hang_total += hangs
    unreadable += len(bad)
    if bad:
      ctx.note(f'node{i} while node{victim} down: {len(bad)}/{len(exp)} '
               f'entries unavailable, e.g. {bad[:3]}')
  ctx.metrics['entries_unavailable_while_down'] = unreadable
  # New I/O on survivors, in a fresh directory.
  new_ok = 0
  new_fail = []
  for i in survivors:
    for k in range(5):
      p = ctx.p(f'during_{i}_{k}')
      r = ctx.a(i).call('write_file', timeout=OP_DEADLINE, path=p,
                        size=256 * 1024, seed=k)
      if r.get('hang'):
        hang_total += 1
        new_fail.append((p, 'HANG'))
        break
      if r['ok']:
        new_ok += 1
      else:
        new_fail.append((p, r['err']))
  ctx.metrics['new_writes_ok_while_down'] = new_ok
  ctx.metrics['new_writes_failed_while_down'] = len(new_fail)
  if new_fail:
    ctx.note(f'new writes failing while node down: {new_fail[:3]}')
  ctx.check(hang_total == 0, f'{hang_total} operations HUNG (> '
                             f'{OP_DEADLINE}s) on survivors after node '
                             f'{vh} died')
  # Bring the victim back.
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  ctx.metrics['down_s'] = round(time.time() - t_kill, 1)
  time.sleep(2)
  audit_all(ctx, root, exp, f'after {vh} rejoined')
  for i in survivors:
    for k in range(5):
      p = ctx.p(f'during_{i}_{k}')
      r = ctx.call(victim, 'verify_file', path=p, size=256 * 1024, seed=k)
      if (p, 'HANG') in new_fail or any(p == f[0] for f in new_fail):
        continue
      ctx.check(r['ok'] and r['ret']['ok'],
                f'write acknowledged while {vh} was down is not readable '
                f'from it after rejoin: {p}: {r.get("err") or r.get("ret")}')


@test('node_kill_during_io', 'fault', min_nodes=2, redeploy_after=True,
      timeout=3600)
def t_kill_during_io(ctx):
  """Every node streams fsync'd writes; one node is SIGKILLed mid-stream.
  Survivors' ops must return within OP_DEADLINE; after restart every write
  a survivor saw acknowledged must read back byte-exact."""
  n = len(ctx.hosts)
  victim = n - 1
  vh = ctx.hosts[victim]
  for i in range(n):
    ctx.ok(i, 'makedirs', path=ctx.p(f'w{i}'))
  acked = {}
  hangs = []
  stop_at = time.time() + 25

  def writer(i):
    k = 0
    mine = []
    while time.time() < stop_at:
      p = ctx.p(f'w{i}', f'f{k}')
      r = ctx.a(i).call('write_file', timeout=OP_DEADLINE, path=p,
                        size=512 * 1024 + k, seed=i * 100000 + k, fsync=True)
      if r.get('hang') or r.get('agent_dead'):
        hangs.append((i, p))
        return mine
      if r['ok']:
        mine.append((p, 512 * 1024 + k, i * 100000 + k))
      k += 1
    return mine

  def killer(_):
    time.sleep(8)
    ctx.cl.kill_fuse(vh)
    ctx.cl.kill_runtime(vh)
    return []
  outs = parallel(lambda f: f[0](f[1]),
                  [(writer, i) for i in range(n - 1)] + [(killer, 0)])
  for i, o in enumerate(outs[:-1]):
    if isinstance(o, Exception):
      raise o
    acked[i] = o
  ctx.metrics['acked_writes'] = sum(len(v) for v in acked.values())
  ctx.check(not hangs, f'survivor writes hung after {vh} died: {hangs}')
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), 'victim restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), 'victim remount')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  lost = []
  for i, files in acked.items():
    for p, size, seed in files:
      r = ctx.call((i + 1) % n, 'verify_file', path=p, size=size, seed=seed)
      if not (r['ok'] and r['ret']['ok']):
        lost.append((p, r.get('err') or r.get('ret')))
  ctx.check(not lost, f'{len(lost)} fsync-acknowledged writes lost/corrupt '
                      f'after node loss + rejoin, e.g. {lost[:3]}')


@test('fuse_client_crash', 'fault', min_nodes=1, redeploy_after=True,
      timeout=1800)
def t_fuse_crash(ctx):
  """SIGKILL the FUSE daemon mid-write: callers get an error (not a hang),
  the runtime stays up, remount works, fsync'd data is intact."""
  i = len(ctx.hosts) - 1
  h = ctx.hosts[i]
  ctx.ok(i, 'write_file', path=ctx.p('safe'), size=5 * MiB, seed=1,
         fsync=True)
  big = ctx.p('inflight')
  box = {}

  def w(_):
    box['r'] = ctx.a(i).call('write_file', timeout=120, path=big,
                             size=2048 * MiB, seed=2, chunk=4 * MiB)
  def k(_):
    time.sleep(3)
    ctx.cl.kill_fuse(h)
  parallel(lambda f: f(None), [w, k])
  r = box['r']
  ctx.check(not r.get('hang'), f'writer hung after FUSE daemon died: {r}')
  ctx.check(not r['ok'], 'write through a killed FUSE daemon succeeded?')
  ctx.check(bool(ctx.cl.runtime_pid(h)), 'runtime died with its FUSE client')
  ctx.cl.agents.pop(h, None)
  ctx.check(ctx.cl.mount(h), 'remount after FUSE crash failed')
  v = ctx.ok(i, 'verify_file', path=ctx.p('safe'), size=5 * MiB, seed=1)
  ctx.check(v['ok'], f'fsync\'d data after FUSE crash: {v}')
  st = ctx.call(i, 'stat', path=big)
  ctx.note(f'partial file after crash: {st.get("ret", {}).get("size")} '
           f'bytes ({st.get("err")})')
  ctx.ok(i, 'unlink', path=big)


@test('chaos_cycles', 'fault', min_nodes=2, redeploy_after=True,
      timeout=5400)
def t_chaos(ctx):
  """5 rounds: write a batch from every node, then crash a random node
  (daemon or FUSE) and recover it; all acknowledged data must survive
  every round."""
  n = len(ctx.hosts)
  rng = random.Random(42)
  exp_all = []
  for rnd in range(5):
    root, exp = build_dataset(ctx, f'x{rnd}', per_node=4)
    exp_all.append((root, exp))
    v = rng.randrange(n)
    vh = ctx.hosts[v]
    what = rng.choice(['runtime', 'fuse'])
    ctx.note(f'round {rnd}: kill {what} on node{v}')
    ctx.cl.kill_fuse(vh)
    if what == 'runtime':
      ctx.cl.kill_runtime(vh)
      time.sleep(1)
      ctx.cl.start_runtime(vh)
      ctx.check(ctx.cl.runtime_up(vh), f'round {rnd} restart')
      time.sleep(3)
    ctx.check(ctx.cl.mount(vh), f'round {rnd} remount')
    ctx.cl.agents.pop(vh, None)
    time.sleep(2)
    for r0, e0 in exp_all:
      audit_all(ctx, r0, e0, f'round {rnd} ({what} crash on node{v})')


@test('any_node_loss_bounded', 'fault', min_nodes=2, redeploy_after=True,
      timeout=3600)
def t_node_loss_partial(ctx):
  """SIGKILL one node (one owning no ancestor of the dataset root).  The
  namespace is hash-sharded, so only the entries and inodes that node owns
  become unavailable: every op on a
  survivor must finish inside OP_DEADLINE (fail fast on the dead shard,
  never hang), most of the namespace must stay reachable, and after the node
  restarts every node sees the whole dataset again."""
  n = len(ctx.hosts)
  root, exp = build_dataset(ctx, 'h', per_node=6)
  vi = victim_off_path(ctx, root)
  victim = ctx.hosts[vi]
  ctx.note(f'victim node{vi} (owns none of the dataset root\'s ancestors)')
  ctx.cl.kill_fuse(victim)
  ctx.cl.kill_runtime(victim)
  time.sleep(2)
  hangs, errs, oks = 0, 0, 0
  for i in [k for k in range(n) if k != vi]:
    for rel in sorted(exp):
      r = ctx.a(i).call('stat', timeout=OP_DEADLINE, path=f'{root}/{rel}')
      if r.get('hang'):
        hangs += 1
      elif r['ok']:
        oks += 1
      else:
        errs += 1
    r = ctx.a(i).call('write_file', timeout=OP_DEADLINE,
                      path=f'{root}/while_node_down_{i}', size=4096, seed=1)
    if r.get('hang'):
      hangs += 1
  total = max(1, oks + errs + hangs)
  ctx.metrics.update({'stat_ok_while_node_down': oks,
                      'stat_err_while_node_down': errs,
                      'ops_hung_while_node_down': hangs,
                      'reachable_fraction': round(oks / total, 2)})
  ctx.check(hangs == 0, f'{hangs} ops HUNG while one node was down')
  ctx.check(oks > 0, 'no part of the namespace stayed reachable')
  ctx.cl.start_runtime(victim)
  ctx.check(ctx.cl.runtime_up(victim), 'victim restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(victim), 'victim remount')
  ctx.cl.agents.pop(victim, None)
  time.sleep(2)
  audit_all(ctx, root, exp, 'after the lost node restarted')


@test('tag_names_after_restart', 'fault', min_nodes=2, redeploy_after=True,
      timeout=1800)
def t_tag_names_restart(ctx):
  """A restarted node rebuilds its tag-name mirror: it republishes its own
  directory shard and pulls every peer's, so path search on it (and on the
  peers) again matches the namespace, including names created while it was
  down."""
  n = len(ctx.hosts)
  r = ctx.dir
  want = {'/'}
  for i in range(n):
    ctx.ok(i, 'mkdir', path=f'{r}/n{i}')
    res = ctx.ok(i, 'create_many', dirpath=f'{r}/n{i}', prefix='f',
                 count=100)
    ctx.check(not res['fails'], f'create_many node{i}: {res["fails"][:3]}')
    want.add(f'/n{i}')
    want.update(f'/n{i}/f{k}' for k in range(100))
  ctx.ok(0, 'rename', src=f'{r}/n0', dst=f'{r}/n0moved')
  want = {w.replace('/n0', '/n0moved', 1) if w.startswith('/n0') else w
          for w in want}
  for i in range(n):
    _tags_match(ctx, i, r, want, 'before restart')
  victim = n - 1
  vh = ctx.hosts[victim]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  time.sleep(2)
  # Names created on survivors while the victim is down.  Directories owned
  # by the victim are unreachable, so create in fresh ones and keep those
  # that succeed.
  for k in range(8):
    p = f'{r}/during{k}'
    res = ctx.a(0).call('mkdir', timeout=OP_DEADLINE, path=p)
    if res.get('ok'):
      want.add(f'/during{k}')
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  for i in range(n):
    _tags_match(ctx, i, r, want, f'after {vh} restarted')


def inode_home(ctx, i, path):
  """Container that homes `path`'s inode (encoded in its inode number), or
  None if the id carries no home."""
  ino = ctx.ok(i, 'stat', path=path)['ino']
  major = ino >> 32
  if major & (0x40000000 | 0x80000000):
    return major & 0xFFFF
  return None


@test('append_home_restart', 'fault', min_nodes=2, redeploy_after=True,
      timeout=1800)
def t_append_home_restart(ctx):
  """Writers keep O_APPENDing (fsync every 500 records) to a file while its
  home -- which owns its size and merges its appends -- is SIGKILLed and
  restarted. Every append fsync'd before the crash survives exactly once, in
  its writer's order; appends not yet fsync'd may be lost (as POSIX allows),
  never duplicated or garbled, and writers carry on after the restart."""
  import threading
  n = len(ctx.hosts)
  p = ctx.p('journal')
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  home = inode_home(ctx, 0, p)
  victim = home
  if home is None or home >= n:
    victim = n - 1
    ctx.note(f'inode number does not carry the home ({home}); killing '
             f'node{victim} instead')
  writers = [i for i in range(n) if i != victim]
  res = {}

  def app(i):
    res[i] = ctx.a(i).call('append_records', timeout=600, path=p, writer=i,
                           count=10 ** 7, max_secs=30, fsync_every=500)
  ts = [threading.Thread(target=app, args=(i,)) for i in writers]
  for t in ts:
    t.start()
  time.sleep(4)
  vh = ctx.hosts[victim]
  t_kill = time.time()
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  time.sleep(5)
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  for t in ts:
    t.join()
  durable, maybe = {}, {}
  for i in writers:
    r = res.get(i) or {}
    ctx.check(r.get('ok'), f'node{i} writer: {r.get("err") or r}')
    ret = r.get('ret') or {}
    acked = runs_to_set(ret.get('acked', []))
    # fsyncs that completed a clear second before the kill (node clocks agree
    # to NTP precision) bound what must survive.
    upto = max([k for k, ts_ in ret.get('synced', []) if ts_ < t_kill - 1.0],
               default=-1)
    durable[i] = {k for k in acked if k <= upto}
    top = max(acked) if acked else -1
    maybe[i] = set(range(0, top + 1)) - durable[i]
    ctx.metrics[f'node{i}_acked'] = len(acked)
    ctx.metrics[f'node{i}_durable_before_kill'] = len(durable[i])
    ctx.metrics[f'node{i}_failed'] = ret.get('nfail', 0)
    # The writer recovers: after its last error (a write or fsync that hit
    # the outage) at least one fsync succeeds.
    last_err = max([k for k, _ in ret.get('fails', [])] +
                   [k for k, _ in ret.get('fsync_fails', [])], default=-1)
    ok_after = [k for k, _ in ret.get('synced', []) if k > last_err]
    ctx.metrics[f'node{i}_fsync_failed'] = len(ret.get('fsync_fails', []))
    ctx.check(ok_after, f'node{i}: no fsync succeeded after its last error '
                        f'(seq {last_err}); fsync errors '
                        f'{ret.get("fsync_fails", [])[:5]}')
  for i in writers + [victim]:
    check_records(ctx, i, p, durable, maybe, zeros_ok=True)


@test('data_available_while_page_owner_down', 'fault', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_data_failover(ctx):
  """Files whose names and inodes live on survivors stay fully readable and
  writable while the node holding some of their pages is down (the
  replication chimod's remote copies + core failover), and every change
  made meanwhile is on the returning node afterwards (handoff)."""
  n = len(ctx.hosts)
  root = ctx.p('fo')
  ctx.ok(0, 'mkdir', path=root)
  rel = root[len(ctx.cl.mnt):]
  owners = {fs_dir_owner(d, n) for d in ['/'] + [
      '/' + '/'.join(rel.strip('/').split('/')[:k])
      for k in range(1, len(rel.strip('/').split('/')) + 1)]}
  victim = next((i for i in range(n - 1, 0, -1) if i not in owners), None)
  if victim is None:
    ctx.note('every node owns part of the path; nothing to test')
    return
  size = 8 * MiB
  files = [f'{root}/f{k}' for k in range(8)]
  for k, p in enumerate(files):
    ctx.ok(0, 'write_file', path=p, size=size, seed=100 + k, fsync=True)
  vh = ctx.hosts[victim]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  time.sleep(3)
  survivors = [i for i in range(n) if i != victim]
  bad = []
  for i in survivors:
    for k, p in enumerate(files):
      r = ctx.a(i).call('verify_file', timeout=OP_DEADLINE, path=p, size=size,
                        seed=100 + k)
      if not (r.get('ok') and r['ret']['ok']):
        bad.append((i, p, r.get('err') or r.get('ret')))
  ctx.metrics['unreadable_while_down'] = len(bad)
  ctx.check(not bad, f'{len(bad)} reads failed while node{victim} was down, '
                     f'e.g. {bad[:2]}')
  # Changes while it is down: new files and an overwrite.
  w = survivors[0]
  ctx.ok(w, 'write_file', path=files[0], size=size, seed=900, fsync=True)
  new = [f'{root}/n{k}' for k in range(3)]
  for k, p in enumerate(new):
    ctx.ok(w, 'write_file', path=p, size=3 * MiB, seed=950 + k, fsync=True)
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  want = [(files[0], size, 900)] + [(p, size, 100 + k)
                                    for k, p in enumerate(files) if k > 0]
  want += [(p, 3 * MiB, 950 + k) for k, p in enumerate(new)]
  stale = []
  for i in range(n):
    for p, sz, seed in want:
      r = ctx.call(i, 'verify_file', path=p, size=sz, seed=seed)
      if not (r.get('ok') and r['ret']['ok']):
        stale.append((i, p, r.get('err') or r.get('ret')))
  ctx.check(not stale, f'{len(stale)} reads wrong after node{victim} '
                       f'returned, e.g. {stale[:2]}')


@test('handoff_survives_successor_restart', 'fault', min_nodes=3,
      redeploy_after=True, timeout=1800)
def t_handoff_successor_restart(ctx):
  """A node covering for a dead owner restarts before the owner returns:
  the changes it made meanwhile must still be handed back (the replication
  chimod's handoff log), not forgotten with its memory."""
  n = len(ctx.hosts)
  root = ctx.p('hs')
  ctx.ok(0, 'mkdir', path=root)
  rel = root[len(ctx.cl.mnt):]
  owners = {fs_dir_owner(d, n) for d in ['/'] + [
      '/' + '/'.join(rel.strip('/').split('/')[:k])
      for k in range(1, len(rel.strip('/').split('/')) + 1)]}
  # The owner that goes down must hold none of the path's directories (it
  # stays down while the files are used); its successor, which covers for
  # it, is only down while nothing touches the path. Node 0 writes, so it
  # is neither.
  pick = next(((v, (v + 1) % n) for v in range(n - 1, 0, -1)
               if v not in owners and (v + 1) % n != 0), None)
  if pick is None:
    ctx.note('no owner/successor pair clear of the path; nothing to test')
    return
  victim, succ = pick
  size = 8 * MiB
  files = [f'{root}/f{k}' for k in range(8)]
  for k, p in enumerate(files):
    ctx.ok(0, 'write_file', path=p, size=size, seed=300 + k, fsync=True)
  vh, sh_ = ctx.hosts[victim], ctx.hosts[succ]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  time.sleep(3)
  # Overwrite every file while the owner is down (some pages are the
  # victim's, now written at its successor).
  for k, p in enumerate(files):
    ctx.ok(0, 'write_file', path=p, size=size, seed=700 + k, fsync=True)
  # The successor crashes and comes back while the owner is still down.
  ctx.cl.kill_fuse(sh_)
  ctx.cl.kill_runtime(sh_)
  time.sleep(2)
  ctx.cl.start_runtime(sh_)
  ctx.check(ctx.cl.runtime_up(sh_), f'{sh_} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(sh_), f'{sh_} remount failed')
  ctx.cl.agents.pop(sh_, None)
  # Now the owner returns and must get the successor's changes.
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  stale = []
  for i in range(n):
    for k, p in enumerate(files):
      r = ctx.call(i, 'verify_file', path=p, size=size, seed=700 + k)
      if not (r.get('ok') and r['ret']['ok']):
        stale.append((i, p, r.get('err') or r.get('ret')))
  ctx.metrics['stale_after_return'] = len(stale)
  if stale:
    blob_info_notes(ctx, stale[0][1], stale[0][2])
  ctx.check(not stale, f'{len(stale)} reads wrong after node{victim} '
                       f'returned (node{succ} restarted meanwhile), '
                       f'e.g. {stale[:2]}')


def blob_info_notes(ctx, path, verdict):
  """Note every node's view (cte_search --blob-info) of the page blob a
  failed verify mismatched in, to tell a stale copy from a lost one."""
  mm = verdict.get('mismatch') if isinstance(verdict, dict) else None
  page = (mm or {}).get('offset', 0) // MiB
  # Re-read at once from every node: a transient window (a stale liveness
  # view routing to a restart-emptied copy) reads right again seconds later.
  off = (mm or {}).get('offset', 0)
  for i in range(len(ctx.hosts)):
    rv = ctx.a(i).call('read_hex', timeout=30, path=path, off=off, length=16)
    st = ctx.a(i).call('stat', timeout=30, path=path)
    ino = (st.get('ret') or {}).get('ino') if st.get('ok') else st.get('err')
    ctx.note(f'node{i} ino {ino} re-read @{off}: '
             f'{str(rv.get("ret") if rv.get("ok") else rv.get("err"))[:60]}')
  ino = ctx.ok(0, 'stat', path=path)['ino']
  tag = f'{ino >> 32}.{ino & 0xffffffff}'
  cl = ctx.cl
  for h in ctx.hosts:
    for flag in ('', ' --local'):
      _, out = sh(h, f'{cl.env_prefix()} {cl.bin_dir}/cte_search {tag} {page} '
                     f'--blob-info{flag} 2>&1 | grep -v INFO | tail -4',
                  timeout=60)
      ctx.note(f'{h} page {page} of {tag}{flag or " (owner)"}: '
               f'{out.strip()}')


@test('truncate_during_failover', 'fault', min_nodes=2, redeploy_after=True,
      timeout=1800)
def t_truncate_during_failover(ctx):
  """A file's home (which owns its size) is SIGKILLed; while it is down a
  survivor truncates the file (after failover takes the inode over) and
  appends a tail. When the home comes back it must not resurrect the old,
  larger size: every node sees the truncated size, the kept prefix and the
  new tail, and nothing of the old bytes past the cut."""
  n = len(ctx.hosts)
  MiB = 1 << 20
  p = ctx.p('shrink')
  ctx.ok(0, 'write_file', path=p, size=8 * MiB, seed=5, fsync=True)
  home = inode_home(ctx, 0, p)
  victim = home if home is not None and home < n else n - 1
  a = (victim + 1) % n
  vh = ctx.hosts[victim]
  ctx.note(f'home node{victim}; truncating from node{a}')
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  t0 = time.time()
  done = False
  while time.time() - t0 < 90:
    if ctx.call(a, 'truncate', path=p, size=1 * MiB)['ok']:
      done = True
      break
    time.sleep(2)
  ctx.metrics['truncate_after_s'] = round(time.time() - t0, 1)
  ctx.check(done, 'truncate never succeeded while the home was down '
                  '(no failover within 90 s)')
  # Two timed steps, so a hang names its half: the append, then the fsync.
  r = ctx.ok(a, 'sh', cmd=f'timeout 60 python3 -c "import os; '
                          f'fd=os.open({p!r}, os.O_WRONLY|os.O_APPEND); '
                          f'os.write(fd, b\'T\'*4096); os.close(fd)"')
  ctx.check(r['rc'] == 0, f'append after the truncate failed or took over '
                          f'60 s (rc {r["rc"]}): {r["err"][-300:]}')
  r = ctx.ok(a, 'sh', cmd=f'timeout 60 python3 -c "import os; '
                          f'fd=os.open({p!r}, os.O_RDONLY); os.fsync(fd); '
                          f'os.close(fd)"')
  ctx.check(r['rc'] == 0, f'fsync while the home is down failed or took '
                          f'over 60 s (rc {r["rc"]}): {r["err"][-300:]}')
  want = 1 * MiB + 4096
  for i in range(n):
    if i != victim:
      sz = ctx.ok(i, 'stat', path=p)['size']
      ctx.check(sz == want, f'node{i} sees size {sz} during the outage, '
                            f'want {want}')
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  def tails():
    out = {}
    for i in range(n):
      out[i] = ctx.ok(i, 'sh', cmd=f'python3 -c "f=open({p!r},\'rb\');'
                                   f' f.seek({1 * MiB}); d=f.read(); '
                                   f'print(len(d), set(d)==set(b\'T\'))"'
                      )['out'].strip()
    return {i: t for i, t in out.items() if not t.endswith(f'{4096} True')}
  for i in range(n):
    sz = ctx.ok(i, 'stat', path=p)['size']
    ctx.check(sz == want, f'node{i} sees size {sz} after the home returned, '
                          f'want {want} (old size resurrected?)')
  bad = tails()
  if bad:
    # Stale on every node, or only where a cached copy was made before the
    # outage? And does it heal once the hand-back has had time to run?
    time.sleep(20)
    later = tails()
    ctx.check(False, f'bytes past the cut are not the 4096 x "T" tail: '
                     f'{bad} (writer node0, page-1 owner = home node{victim}'
                     f'?); 20 s later: {later or "all healed"}')
  v = ctx.ok(n - 1 if victim != n - 1 else 0, 'verify_file', path=p,
             size=1 * MiB, seed=5)
  ctx.check(not v['mismatch'], f'kept prefix damaged: {v["mismatch"]}')



@test('open_handle_survives_home_loss', 'fault', min_nodes=2,
      redeploy_after=True, timeout=1800)
def t_open_handle_home_loss(ctx):
  """A process keeps a file open on one node while the file's home (which
  holds the open handle's state and the file's size) is SIGKILLed. Through
  the SAME descriptor it keeps writing, fsyncs, stats and reads after
  failover, and closes it; once the home is back every node reads exactly
  what was written, before and after the loss."""
  n = len(ctx.hosts)
  MiB = 1 << 20
  p = ctx.p('held')
  ctx.ok(0, 'write_file', path=p, size=4 * MiB, seed=0, fsync=True)
  home = inode_home(ctx, 0, p)
  victim = home if home is not None and home < n else n - 1
  a = (victim + 1) % n
  ctx.note(f'home node{victim}; descriptor on node{a}')
  h = ctx.ok(a, 'open', path=p, flags='rw')
  ctx.ok(a, 'fpwrite', h=h, off=0, length=MiB, seed=1)
  ctx.ok(a, 'fsync', h=h)
  vh = ctx.hosts[victim]
  ctx.cl.kill_fuse(vh)
  ctx.cl.kill_runtime(vh)
  # One write and ONE fsync, never retried: a write-back error is reported
  # to one fsync only (as on Linux), so a retried fsync would succeed on
  # bytes that were lost.
  t0 = time.time()
  r = ctx.call(a, 'fpwrite', timeout=300, h=h, off=MiB, length=MiB, seed=1)
  ctx.check(r['ok'], f'write through the open descriptor after the home '
                     f'died: {r.get("err")}')
  r = ctx.call(a, 'fsync', timeout=300, h=h)
  ctx.metrics['write_fsync_after_loss_s'] = round(time.time() - t0, 1)
  ctx.check(r['ok'], f'fsync through the open descriptor after the home '
                     f'died: {r.get("err")}')
  r = ctx.call(a, 'fstat', timeout=60, h=h)
  ctx.check(r['ok'] and r['ret']['size'] == 4 * MiB,
            f'fstat after the home died: {r.get("err") or r.get("ret")}')
  for off, seed in ((0, 1), (MiB, 1), (2 * MiB, 0)):
    r = ctx.call(a, 'fpread_verify', timeout=60, h=h, off=off, length=MiB,
                 seed=seed)
    ctx.check(r['ok'] and r['ret'] is None,
              f'read at {off} through the open descriptor: '
              f'{r.get("err") or r.get("ret")}')
  r = ctx.call(a, 'close', timeout=60, h=h)
  ctx.check(r['ok'], f'close after the home died: {r.get("err")}')
  ctx.cl.start_runtime(vh)
  ctx.check(ctx.cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(vh), f'{vh} remount failed')
  ctx.cl.agents.pop(vh, None)
  time.sleep(2)
  for i in range(n):
    g = ctx.ok(i, 'open', path=p, flags='r')
    for off, seed in ((0, 1), (MiB, 1), (2 * MiB, 0), (3 * MiB, 0)):
      mm = ctx.ok(i, 'fpread_verify', h=g, off=off, length=MiB, seed=seed)
      ctx.check(mm is None, f'node{i} after the home returned, bytes at '
                            f'{off}: {mm}')
    st = ctx.ok(i, 'fstat', h=g)
    ctx.check(st['size'] == 4 * MiB, f'node{i} size {st["size"]}')
    ctx.ok(i, 'close', h=g)
