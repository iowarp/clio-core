"""Cross-node namespace / data coherence and concurrency at N nodes.

Every node mounts the same clio-fs namespace.  The contract tested is the
one NFS-class filesystems give: close-to-open data consistency, and
namespace changes visible on every other node within the attribute-cache
TTL (CLIO_FUSE_ATTR_CACHE_S, default 1 s).  Staleness beyond
VIS_TIMEOUT is a failure; staleness inside it is reported as a metric.
"""

import os
import random
import re
import shlex
import time

from cluster import sh
from suite import test

MiB = 1 << 20
VIS_TIMEOUT = 5.0


def _vis(ctx, i, fn, what):
  """Wait for fn() to hold on node i; fail after VIS_TIMEOUT."""
  done, info, dt = ctx.eventually(fn, timeout=VIS_TIMEOUT)
  ctx.check(done, f'node{i}: {what} not visible after {VIS_TIMEOUT}s: {info}')
  key = f'max_stale_s_{what.split()[0]}'
  ctx.metrics[key] = round(max(ctx.metrics.get(key, 0), dt), 2)


def tag_index_names(ctx, i, rx):
  """CTE tag names matching rx, as node i's cte_search --tag-query sees them
  (the cluster-wide search).  Returns (set, err)."""
  cmd = (f'{ctx.cl.env_prefix()} {ctx.cl.bin_dir}/cte_search '
         f'{shlex.quote(rx)} --tag-query')
  rc, out = sh(ctx.hosts[i], cmd, timeout=60)
  if rc != 0:
    return None, out[-500:]
  return {l.strip() for l in out.splitlines()
          if l.strip() and l.strip() != '(no results)'}, None


def _tags_match(ctx, i, rx, want, what, timeout=5):
  """Require node i's tag search for rx to equal want.  A namespace
  operation changes its CTE tag names before it returns, so there is no
  grace period beyond a few seconds of slack (unless a node was down)."""
  last = {}

  def f():
    got, err = tag_index_names(ctx, i, rx)
    last['got'], last['err'] = got, err
    return got == want, None
  done, _, dt = ctx.eventually(f, timeout=timeout, period=0.5)
  got = last['got'] or set()
  ctx.check(done, f'node{i} tag names {what}: missing '
                  f'{sorted(want - got)[:8]} extra {sorted(got - want)[:8]}'
                  f'{" err " + last["err"] if last["err"] else ""}')
  key = 'max_tag_lag_s'
  ctx.metrics[key] = round(max(ctx.metrics.get(key, 0), dt), 2)


@test('xnode_tag_names', 'dist', min_nodes=2, timeout=1800)
def t_tag_names(ctx):
  """Every link of a clio-fs inode is a flat CTE tag name "<leaf>/<inode>".
  Create, rename (a directory with a subtree), cross-directory move, rename
  over an existing file, unlink, rmdir and hard link, issued back to back
  from different nodes, are all reflected in every node's tag search by the
  time each operation returns."""
  n = len(ctx.hosts)
  r = ctx.dir
  px = f'xtn{os.getpid() % 100000}_'   # this run's leaves
  rx = '^' + re.escape(px) + '[^/]*/[0-9]+$'
  links = {}  # path relative to r -> inode

  def ino(i, rel):
    return ctx.ok(i, 'stat', path=r + rel)['ino']

  def mk(i, rel):
    ctx.ok(i, 'mkdir', path=r + rel)
    links[rel] = ino(i, rel)

  def wr(i, rel):
    ctx.ok(i, 'write_file', path=r + rel, size=64, seed=len(rel))
    links[rel] = ino(i, rel)

  def mv(i, a, b):
    ctx.ok(i, 'rename', src=r + a, dst=r + b)
    for w in [w for w in links if w == a or w.startswith(a + '/')]:
      links[b + w[len(a):]] = links.pop(w)

  def want():
    return {os.path.basename(k) + '/' + str(v) for k, v in links.items()}

  d1, d2, sub, deep = px + 'd1', px + 'd2', px + 'sub', px + 'deep'
  x, f1 = px + 'x', px + 'f1'
  mk(0, f'/{d1}'); mk(0, f'/{d1}/{sub}'); mk(0, f'/{d1}/{sub}/{deep}')
  wr(0, f'/{d1}/{f1}'); wr(0, f'/{d1}/{sub}/{px}f2')
  wr(0, f'/{d1}/{sub}/{deep}/{px}f3')
  mv(1 % n, f'/{d1}', f'/{d2}')
  mk(1 % n, f'/{x}')
  mv(0, f'/{d2}/{sub}', f'/{x}/{px}moved')
  wr(0, f'/{x}/{px}victim'); wr(1 % n, f'/{x}/{px}src')
  mv(0, f'/{x}/{px}src', f'/{x}/{px}victim')   # rename over an existing file
  wr(1 % n, f'/{x}/{px}gone'); ctx.ok(0, 'unlink', path=f'{r}/{x}/{px}gone')
  links.pop(f'/{x}/{px}gone')
  mk(0, f'/{x}/{px}rmd'); ctx.ok(1 % n, 'rmdir', path=f'{r}/{x}/{px}rmd')
  links.pop(f'/{x}/{px}rmd')
  ctx.ok(0, 'link', src=f'{r}/{d2}/{f1}', dst=f'{r}/{x}/{px}f1link')
  links[f'/{x}/{px}f1link'] = links[f'/{d2}/{f1}']
  # A directory per node with many entries, owners spread by hashing.
  for i in range(n):
    mk(i, f'/{px}many{i}')
    res = ctx.ok(i, 'create_many', dirpath=f'{r}/{px}many{i}', prefix=px + 'e',
                 count=200)
    ctx.check(not res['fails'], f'create_many node{i}: {res["fails"][:3]}')
    for k in range(200):
      rel = f'/{px}many{i}/{px}e{k}'
      links[rel] = ino(i, rel)
  mv(0, f'/{px}many0', f'/{x}/{px}many0moved')
  for i in range(n):
    _tags_match(ctx, i, rx, want(), 'after namespace churn')
  # Unlinking the original name leaves the hard link's name.
  ctx.ok(0, 'unlink', path=f'{r}/{d2}/{f1}')
  links.pop(f'/{d2}/{f1}')
  for i in range(n):
    _tags_match(ctx, i, rx, want(), 'after unlinking a hard-linked name')


@test('xnode_truncate_cache_coherence', 'dist', min_nodes=2)
def t_trunc_coherence(ctx):
  """Node B caches a file's pages; node A truncates it and writes past the
  old end. B must read zeros in the hole and the new bytes after it -- never
  the old bytes from its cached copies of the truncated pages."""
  from agent import pattern
  a, b = 0, 1
  p = ctx.p('tc')
  size = 2 * MiB
  ctx.ok(a, 'write_file', path=p, size=size, seed=1)
  r = ctx.call(b, 'verify_file', path=p, size=size, seed=1)  # B caches
  ctx.check(r['ok'] and r['ret']['ok'], f'initial read on node{b}: {r}')
  model = bytearray(pattern(1, 0, size))  # what the file must contain
  for rnd in range(3):
    keep = 100 + rnd * 4096
    tail_off = MiB + MiB // 2 + rnd * 777
    ctx.ok(a, 'truncate', path=p, size=keep)
    del model[keep:]
    h = ctx.ok(a, 'open', path=p, flags='w')
    ctx.ok(a, 'fpwrite', h=h, off=tail_off, length=1000, seed=2 + rnd)
    ctx.ok(a, 'close', h=h)
    model.extend(bytes(tail_off - len(model)))
    model.extend(pattern(2 + rnd, tail_off, 1000))
    want = bytes(model)

    def f(want=want):
      got = bytes.fromhex(ctx.ok(b, 'read_hex', path=p))
      if got == want:
        return True, None
      bad = next((k for k in range(min(len(got), len(want)))
                  if got[k] != want[k]), min(len(got), len(want)))
      return False, f'len {len(got)} vs {len(want)}, first diff at {bad}'
    done, info, dt = ctx.eventually(f, timeout=VIS_TIMEOUT)
    ctx.check(done, f'round {rnd}: node{b} read stale bytes after node{a} '
                    f'truncated and re-extended: {info}')
    # B re-caches the new content for the next round.
    r = ctx.call(b, 'read_hex', path=p, off=0, length=4096)


@test('xnode_fsx', 'dist', min_nodes=2, timeout=2400)
def t_xfsx(ctx):
  """fsx model check with every op on a different node (close-to-open)."""
  from fsx_model import run_fsx
  if ctx.cl.attr_cache_s not in ('0', 0, '0.0'):
    # Size/attr checks rely on stat right after another node's close; with
    # an attribute cache that is allowed to be stale, so re-stat through a
    # fresh open (the agent's verify path opens the file).
    ctx.note(f'attr cache {ctx.cl.attr_cache_s}s: size checks may observe '
             f'TTL staleness')
  run_fsx(ctx, nodes=list(range(len(ctx.hosts))), ops=1500, seed=77,
          maxlen=4 * MiB, full_every=100)


@test('xnode_fpp_throughput', 'dist', min_nodes=1, timeout=1800)
def t_fpp_bw(ctx):
  """File-per-node streaming write+verify (4 files/node, 512 MiB each)."""
  n = len(ctx.hosts)
  size = 512 * MiB
  files = 4
  t0 = time.time()

  def wr(i):
    for f in range(files):
      ctx.ok(i, 'write_file', path=ctx.p(f'n{i}_{f}'), size=size,
             seed=i * 10 + f, chunk=4 * MiB, timeout=900)
  ctx.each(wr)
  t1 = time.time()

  def rd(i):
    j = (i + 1) % n  # read the NEXT node's files: cross-node reads
    for f in range(files):
      v = ctx.ok(i, 'verify_file', path=ctx.p(f'n{j}_{f}'), size=size,
                 seed=j * 10 + f, chunk=4 * MiB, timeout=900)
      ctx.check(v['ok'], f'node{i} reading node{j} file {f}: {v}')
  ctx.each(rd)
  t2 = time.time()
  tot = n * files * size / MiB
  ctx.metrics['write_MiBps'] = round(tot / (t1 - t0))
  ctx.metrics['xnode_read_verify_MiBps'] = round(tot / (t2 - t1))


@test('xnode_storm_mixed', 'dist', min_nodes=2, timeout=2400)
def t_storm(ctx):
  """Random mixed namespace+data ops from all nodes in private dirs, then a
  cross-node audit that every node sees the identical tree."""
  n = len(ctx.hosts)

  def worker(i):
    rng = random.Random(i)
    base = ctx.p(f'w{i}')
    ctx.ok(i, 'mkdir', path=base)
    live = {}
    for k in range(400):
      r = rng.random()
      name = f'{base}/f{rng.randrange(60)}'
      if r < 0.4:
        size = rng.choice([0, 1, 4096, 100000, MiB + 3])
        ctx.ok(i, 'write_file', path=name, size=size, seed=i * 1000 + k)
        live[name] = (size, i * 1000 + k)
      elif r < 0.55 and live:
        victim = rng.choice(sorted(live))
        ctx.ok(i, 'unlink', path=victim)
        del live[victim]
      elif r < 0.7 and live:
        src = rng.choice(sorted(live))
        dst = f'{base}/f{rng.randrange(60)}'
        ctx.ok(i, 'rename', src=src, dst=dst)
        if src != dst:
          live[dst] = live.pop(src)
      elif r < 0.8:
        d = f'{base}/sub{rng.randrange(5)}'
        rr = ctx.call(i, 'mkdir', path=d)
        if not rr['ok'] and 'EEXIST' not in rr['err']:
          ctx.check(False, f'mkdir {d}: {rr["err"]}')
      elif live:
        name = rng.choice(sorted(live))
        size, sd = live[name]
        v = ctx.ok(i, 'verify_file', path=name, size=size, seed=sd)
        ctx.check(v['ok'], f'node{i} own file {name}: {v}')
    return live
  lives = ctx.each(worker)
  time.sleep(2)  # past the attr-cache TTL
  manifests = ctx.all_nodes('tree_manifest', root=ctx.dir, timeout=900)
  m0 = manifests[0]['ret']
  for i, m in enumerate(manifests[1:], 1):
    mi = m['ret']
    ctx.check(mi == m0, f'node{i} tree differs from node0: '
                        f'only0={sorted(set(m0) - set(mi))[:5]} '
                        f'only{i}={sorted(set(mi) - set(m0))[:5]} '
                        f'diff={[k for k in m0 if k in mi and m0[k] != mi[k]][:5]}')
  for i, live in enumerate(lives):
    for name, (size, _) in live.items():
      rel = name[len(ctx.dir) + 1:]
      ctx.check(rel in m0 and m0[rel][1] == size,
                f'{rel}: expected size {size}, manifest {m0.get(rel)}')
