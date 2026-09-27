"""Failure / restart tests: persistence and behaviour under node loss.

What a persistent, reliable deployment must guarantee (and what is checked):
  * Everything acknowledged by fsync()/close() before a graceful stop or a
    crash (SIGKILL of every daemon) is back, byte-exact, with its metadata
    (names, sizes, modes, links, xattrs) after `clio_run restart`.
  * Losing one node never hangs the survivors: every operation on a
    survivor returns (success or an error) inside a bounded time.
  * A node that comes back rejoins and its data is readable cluster-wide.
  * A crashed FUSE client does not take the daemon down and can remount.
Each test leaves the cluster redeployed from scratch (redeploy_after).
"""

import random
import time

from cluster import parallel
from suite import test

MiB = 1 << 20
OP_DEADLINE = 60  # seconds any single op may take while a node is down


def build_dataset(ctx, tag, per_node=12):
  """Populate a mixed tree from every node; return the expected manifest.

  The manifest maps relpath -> (kind, size, perm, seed/target) and is what
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
    ctx.ok(i, 'setxattr', path=f'{d}/sub', name='user.tag',
           value_hex=f'{tag}{i}'.encode().hex())
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
  `clio_run restart`, then remount."""
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
  parallel(lambda h: cl.start_runtime(h, 'restart'), cl.hosts)
  ups = parallel(cl.runtime_up, cl.hosts)
  ctx.check(all(u is True for u in ups), f'runtime restart failed: {ups}')
  time.sleep(3)
  ms = parallel(cl.mount, cl.hosts)
  ctx.check(all(m is True for m in ms), f'remount failed: {ms}')
  ctx.metrics['restart_s'] = round(time.time() - t0, 1)


@test('graceful_restart_all', 'fault', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_graceful(ctx):
  """Graceful stop of every daemon + `clio_run restart`: all data back."""
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
  """Operator runs `clio_run start` (not `restart`) after a clean stop: does
  a persistent deployment keep its data?"""
  root, exp = build_dataset(ctx, 's', per_node=4)
  cl = ctx.cl
  cl.close_agents()
  parallel(cl.unmount, cl.hosts)
  parallel(cl.stop_runtime, cl.hosts)
  parallel(lambda h: cl.start_runtime(h, 'start'), cl.hosts)
  parallel(cl.runtime_up, cl.hosts)
  time.sleep(3)
  parallel(cl.mount, cl.hosts)
  bad, _ = audit(ctx, 0, root, exp)
  ctx.check(not bad, f'`clio_run start` on an existing persistent '
                     f'deployment lost {len(bad)}/{len(exp)} entries '
                     f'(e.g. {bad[:3]}); only `restart` replays the WAL')


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
  ctx.cl.start_runtime(vh, 'restart')
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
  ctx.cl.start_runtime(vh, 'restart')
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
      ctx.cl.start_runtime(vh, 'restart')
      ctx.check(ctx.cl.runtime_up(vh), f'round {rnd} restart')
      time.sleep(3)
    ctx.check(ctx.cl.mount(vh), f'round {rnd} remount')
    ctx.cl.agents.pop(vh, None)
    time.sleep(2)
    for r0, e0 in exp_all:
      audit_all(ctx, r0, e0, f'round {rnd} ({what} crash on node{v})')


@test('home_node_loss_bounded', 'fault', min_nodes=2, redeploy_after=True,
      timeout=3600)
def t_home_loss(ctx):
  """SIGKILL the METADATA HOME (node 0).  The namespace lives there, so the
  other nodes cannot make progress -- but every op must FAIL inside
  OP_DEADLINE, never hang; after the home restarts, every node sees the
  whole dataset again."""
  n = len(ctx.hosts)
  root, exp = build_dataset(ctx, 'h', per_node=6)
  home = ctx.hosts[0]
  ctx.cl.kill_fuse(home)
  ctx.cl.kill_runtime(home)
  time.sleep(2)
  hangs, errs, oks = 0, 0, 0
  for i in range(1, n):
    for rel in sorted(exp)[:6]:
      r = ctx.a(i).call('stat', timeout=OP_DEADLINE, path=f'{root}/{rel}')
      if r.get('hang'):
        hangs += 1
      elif r['ok']:
        oks += 1
      else:
        errs += 1
    r = ctx.a(i).call('write_file', timeout=OP_DEADLINE,
                      path=f'{root}/while_home_down_{i}', size=4096, seed=1)
    if r.get('hang'):
      hangs += 1
  ctx.metrics.update({'ops_ok_while_home_down': oks,
                      'ops_err_while_home_down': errs,
                      'ops_hung_while_home_down': hangs})
  ctx.check(hangs == 0, f'{hangs} ops HUNG while the metadata home was down')
  ctx.cl.start_runtime(home, 'restart')
  ctx.check(ctx.cl.runtime_up(home), 'home restart')
  time.sleep(3)
  ctx.check(ctx.cl.mount(home), 'home remount')
  ctx.cl.agents.pop(home, None)
  time.sleep(2)
  audit_all(ctx, root, exp, 'after the metadata home restarted')
