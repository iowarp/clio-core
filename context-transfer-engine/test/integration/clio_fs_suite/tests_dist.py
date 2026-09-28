"""Cross-node namespace / data coherence and concurrency at N nodes.

Every node mounts the same clio-fs namespace.  The contract tested is the
one NFS-class filesystems give: close-to-open data consistency, and
namespace changes visible on every other node within the attribute-cache
TTL (CLIO_FUSE_ATTR_CACHE_S, default 1 s).  Staleness beyond
VIS_TIMEOUT is a failure; staleness inside it is reported as a metric.
"""

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


def tag_index_paths(ctx, i, root):
  """Absolute CTE tag names at or under mount-relative dir root, as seen by
  node i's tag search index (cte_search --tag-query).  Returns (set, err)."""
  rel = root[len(ctx.cl.mnt):] or '/'
  rx = '^' + re.escape(rel) + '(/.*)?$'
  cmd = (f'{ctx.cl.env_prefix()} {ctx.cl.bin_dir}/cte_search '
         f'{shlex.quote(rx)} --tag-query')
  rc, out = sh(ctx.hosts[i], cmd, timeout=60)
  if rc != 0:
    return None, out[-500:]
  names = {l.strip() for l in out.splitlines() if l.startswith('/')}
  return {n[len(rel):] or '/' for n in names}, None


def _tags_match(ctx, i, root, want, what):
  """Wait until node i's tag index under root equals want (relative names)."""
  last = {}

  def f():
    got, err = tag_index_paths(ctx, i, root)
    last['got'], last['err'] = got, err
    return got == want, None
  done, _, dt = ctx.eventually(f, timeout=60, period=1.0)
  got = last['got'] or set()
  ctx.check(done, f'node{i} tag index {what}: missing '
                  f'{sorted(want - got)[:8]} extra {sorted(got - want)[:8]}'
                  f'{" err " + last["err"] if last["err"] else ""}')
  key = 'max_tag_lag_s'
  ctx.metrics[key] = round(max(ctx.metrics.get(key, 0), dt), 2)


@test('xnode_create_visible', 'dist', min_nodes=2)
def t_create_vis(ctx):
  """File written+closed on node k is readable, byte-exact, on every node."""
  n = len(ctx.hosts)
  for k in range(n):
    ctx.ok(k, 'write_file', path=ctx.p(f'from{k}'), size=3 * MiB + k,
           seed=k + 1)
  for i in range(n):
    for k in range(n):
      p = ctx.p(f'from{k}')

      def f(i=i, p=p, k=k):
        r = ctx.call(i, 'verify_file', path=p, size=3 * MiB + k, seed=k + 1)
        return (r['ok'] and r['ret']['ok']), r.get('ret') or r.get('err')
      _vis(ctx, i, f, f'create from{k}')


@test('xnode_overwrite_close_to_open', 'dist', min_nodes=2)
def t_c2o(ctx):
  """Writer rewrites a file 30x; after each close, readers see that version."""
  n = len(ctx.hosts)
  p = ctx.p('ver')
  for v in range(30):
    w = v % n
    size = random.Random(v).randrange(1, 3 * MiB)
    ctx.ok(w, 'write_file', path=p, size=size, seed=1000 + v)
    for i in range(n):
      if i == w:
        continue
      r = ctx.call(i, 'verify_file', path=p, size=size, seed=1000 + v)
      if not (r['ok'] and r['ret']['ok']):
        # Allow attr-cache staleness on size, then require exactness.
        def f(i=i, size=size, v=v):
          rr = ctx.call(i, 'verify_file', path=p, size=size, seed=1000 + v)
          return rr['ok'] and rr['ret']['ok'], rr.get('ret') or rr.get('err')
        done, info, dt = ctx.eventually(f, timeout=VIS_TIMEOUT)
        ctx.check(done, f'v{v}: node{i} reads stale/corrupt data after '
                        f'writer node{w} closed: {info}')
        ctx.note(f'v{v} node{i} needed {dt:.2f}s (first read: '
                 f'{r.get("ret") or r.get("err")})')


@test('xnode_unlink_rename_visible', 'dist', min_nodes=2)
def t_unlink_rename(ctx):
  """unlink / rename / mkdir / rmdir on node A are seen on node B."""
  n = len(ctx.hosts)
  for a in range(n):
    b = (a + 1) % n
    f1, f2 = ctx.p(f'u{a}'), ctx.p(f'r{a}')
    ctx.ok(a, 'write_file', path=f1, size=100, seed=a)
    _vis(ctx, b, lambda: (ctx.call(b, 'exists', path=f1).get('ret') is True,
                          None), 'create')
    ctx.ok(a, 'rename', src=f1, dst=f2)
    _vis(ctx, b, lambda: (ctx.call(b, 'exists', path=f1).get('ret') is False
                          and ctx.call(b, 'exists', path=f2).get('ret')
                          is True, ctx.call(b, 'listdir', path=ctx.dir)
                          .get('ret')), 'rename')
    r = ctx.call(b, 'verify_file', path=f2, size=100, seed=a)
    ctx.check(r['ok'] and r['ret']['ok'], f'renamed data on node{b}: {r}')
    ctx.ok(a, 'unlink', path=f2)
    _vis(ctx, b, lambda: (ctx.call(b, 'exists', path=f2).get('ret') is False,
                          None), 'unlink')
    d = ctx.p(f'd{a}')
    ctx.ok(a, 'mkdir', path=d)
    _vis(ctx, b, lambda: (ctx.call(b, 'exists', path=d).get('ret') is True,
                          None), 'mkdir')
    ctx.ok(b, 'write_file', path=f'{d}/child', size=5, seed=1)
    ctx.err(a, 'rmdir', 'ENOTEMPTY', path=d)
    ctx.ok(b, 'unlink', path=f'{d}/child')
    # node A must be able to rmdir as soon as B's unlink returned.
    r = ctx.call(a, 'rmdir', path=d)
    if not r['ok']:
      done, info, dt = ctx.eventually(
          lambda: (ctx.call(a, 'rmdir', path=d)['ok'], None),
          timeout=VIS_TIMEOUT)
      ctx.check(done, f'node{a} rmdir after node{b} emptied dir: {r["err"]}')
      ctx.note(f'rmdir after remote unlink needed {dt:.2f}s ({r["err"]})')


@test('xnode_readdir_union', 'dist', min_nodes=2, timeout=1800)
def t_readdir_union(ctx):
  """All nodes create 1000 files each in ONE dir; every node lists all."""
  n = len(ctx.hosts)
  per = 1000
  d = ctx.p('shared')
  ctx.ok(0, 'mkdir', path=d)
  _vis(ctx, n - 1, lambda: (ctx.call(n - 1, 'exists', path=d).get('ret'),
                            None), 'mkdir')
  t0 = time.time()
  rs = ctx.all_nodes('create_many', dirpath=d, prefix='', count=per,
                     size=0, timeout=1500)
  dt = time.time() - t0
  # prefix must differ per node: redo with node-specific prefixes.
  ctx.check(all(r['ok'] for r in rs), f'create_many: {rs}')
  ctx.each(lambda i: ctx.ok(i, 'create_many', dirpath=d, prefix=f'n{i}_',
                            count=per, timeout=1500))
  dt = time.time() - t0
  ctx.metrics['creates_per_s_aggregate'] = round(2 * n * per / dt)
  want = n * per + per  # node-specific + the shared-name set (0..per-1)
  for i in range(n):
    def f(i=i):
      c = ctx.call(i, 'scandir_count', path=d, timeout=300)
      return c.get('ret') == want, c.get('ret')
    _vis(ctx, i, f, 'readdir')
  ctx.ok(n - 1, 'rmtree', path=d, timeout=1500)
  for i in range(n):
    _vis(ctx, i, lambda i=i: (ctx.call(i, 'exists', path=d).get('ret')
                              is False, None), 'rmtree')


@test('xnode_excl_create_race', 'dist', min_nodes=2)
def t_excl_race(ctx):
  """O_CREAT|O_EXCL on the same name from every node: exactly one wins."""
  n = len(ctx.hosts)
  for rnd in range(20):
    p = ctx.p(f'lock{rnd}')
    rs = ctx.all_nodes('write_file', path=p, size=64, seed=rnd * 10,
                       flags='wcx')
    winners = [i for i, r in enumerate(rs) if r['ok']]
    losers = [r.get('err') for r in rs if not r['ok']]
    ctx.check(len(winners) == 1, f'round {rnd}: {len(winners)} winners '
                                 f'{winners}; losers {losers}')
    ctx.check(all('EEXIST' in (e or '') for e in losers),
              f'round {rnd}: loser errors {losers}')


@test('xnode_mkdir_race', 'dist', min_nodes=2)
def t_mkdir_race(ctx):
  """mkdir of the same name from every node: exactly one succeeds."""
  for rnd in range(20):
    p = ctx.p(f'd{rnd}')
    rs = ctx.all_nodes('mkdir', path=p)
    winners = [i for i, r in enumerate(rs) if r['ok']]
    ctx.check(len(winners) == 1, f'round {rnd}: {len(winners)} mkdir winners; '
                                 f'{[r.get("err") for r in rs]}')


@test('xnode_rename_race', 'dist', min_nodes=2)
def t_rename_race(ctx):
  """Every node renames its own file onto one target: the target ends up as
  exactly one intact source file and every source is gone."""
  n = len(ctx.hosts)
  for rnd in range(10):
    tgt = ctx.p(f'tgt{rnd}')
    for i in range(n):
      ctx.ok(i, 'write_file', path=ctx.p(f's{rnd}_{i}'), size=4096 + i,
             seed=rnd * 100 + i)
    rs = ctx.each(lambda i: ctx.call(i, 'rename', src=ctx.p(f's{rnd}_{i}'),
                                     dst=tgt))
    ctx.check(all(r['ok'] for r in rs), f'rename errors {rs}')
    st = ctx.ok(0, 'stat', path=tgt)
    i = st['size'] - 4096
    ctx.check(0 <= i < n, f'target size {st["size"]} matches no source')
    v = ctx.ok(0, 'verify_file', path=tgt, size=4096 + i, seed=rnd * 100 + i)
    ctx.check(v['ok'], f'target content is torn/mixed: {v}')
    left = [x for x in ctx.ok(0, 'listdir', path=ctx.dir)
            if x.startswith(f's{rnd}_')]
    ctx.check(not left, f'sources survived rename: {left}')


@test('xnode_shared_file_disjoint', 'dist', min_nodes=2, timeout=1800)
def t_shared_disjoint(ctx):
  """N-1 shared file: node i writes stripes i, i+N, ... (1 MiB + odd sizes);
  after all close, every node reads the whole file byte-exact."""
  n = len(ctx.hosts)
  p = ctx.p('shared.bin')
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  _vis(ctx, n - 1, lambda: (ctx.call(n - 1, 'exists', path=p).get('ret'),
                            None), 'create')
  for stripe in (MiB, 777 * 1024 + 3, 4096):
    count = 16 * n
    seed = stripe

    def writer(i):
      h = ctx.ok(i, 'open', path=p, flags='rw')
      for s in range(i, count, n):
        ctx.ok(i, 'fpwrite', h=h, off=s * stripe, length=stripe, seed=seed)
      ctx.ok(i, 'fsync', h=h)
      ctx.ok(i, 'close', h=h)
    ctx.each(writer)
    total = count * stripe

    def reader(i):
      def f():
        r = ctx.call(i, 'verify_file', path=p, size=total, seed=seed)
        return r['ok'] and r['ret']['ok'], r.get('ret') or r.get('err')
      _vis(ctx, i, f, f'shared stripe={stripe}')
    ctx.each(reader)
    ctx.ok(0, 'truncate', path=p, size=0)


@test('xnode_append_many_writers', 'dist', min_nodes=2)
def t_xappend(ctx):
  """O_APPEND from every node concurrently: no record lost or overwritten."""
  n = len(ctx.hosts)
  p = ctx.p('journal')
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  _vis(ctx, n - 1, lambda: (ctx.call(n - 1, 'exists', path=p).get('ret'),
                            None), 'create')
  per = 200

  def app(i):
    h = ctx.ok(i, 'open', path=p, flags='wa')
    for k in range(per):
      rec = f'N{i:02d}R{k:05d}'.ljust(31, '.') + '\n'
      ctx.ok(i, 'fwrite', h=h, data_hex=rec.encode().hex())
    ctx.ok(i, 'close', h=h)
  ctx.each(app)
  data = bytes.fromhex(ctx.ok(0, 'read_hex', path=p)).decode(errors='replace')
  lines = data.split('\n')[:-1]
  want = {f'N{i:02d}R{k:05d}'.ljust(31, '.') for i in range(n)
          for k in range(per)}
  got = set(lines)
  ctx.check(len(data) == 32 * n * per,
            f'journal size {len(data)} != {32 * n * per} '
            f'(lost/overwritten appends)')
  ctx.check(got == want, f'missing {len(want - got)} records, '
                         f'garbage {len(got - want)}')


@test('xnode_metadata_attrs', 'dist', min_nodes=2)
def t_xattrs(ctx):
  """chmod / utimens / xattr / truncate set on node A are seen on node B."""
  n = len(ctx.hosts)
  p = ctx.p('meta')
  ctx.ok(0, 'write_file', path=p, size=5000, seed=1)
  for a in range(n):
    b = (a + 1) % n
    mode = [0o600, 0o640, 0o755, 0o700][a % 4]
    ctx.ok(a, 'chmod', path=p, mode=mode)
    _vis(ctx, b, lambda: (ctx.call(b, 'stat', path=p)['ret']['perm'] == mode,
                          oct(ctx.call(b, 'stat', path=p)['ret']['perm'])),
         'chmod')
    t = 1_600_000_000_000_000_000 + a * 10**9
    ctx.ok(a, 'utime', path=p, atime_ns=t, mtime_ns=t)
    _vis(ctx, b, lambda: (abs(ctx.call(b, 'stat', path=p)['ret']['mtime_ns']
                              - t) < 10**6, ctx.call(b, 'stat', path=p)
                          ['ret']['mtime_ns']), 'utime')
    val = f'node{a}'.encode().hex()
    ctx.ok(a, 'setxattr', path=p, name='user.who', value_hex=val)
    _vis(ctx, b, lambda: (ctx.call(b, 'getxattr', path=p, name='user.who')
                          .get('ret') == val, ctx.call(
                              b, 'getxattr', path=p, name='user.who')),
         'xattr')
    sz = 1000 + a
    ctx.ok(a, 'truncate', path=p, size=sz)
    _vis(ctx, b, lambda: (ctx.call(b, 'stat', path=p)['ret']['size'] == sz,
                          ctx.call(b, 'stat', path=p)['ret']['size']),
         'truncate')


@test('xnode_hardlink_symlink', 'dist', min_nodes=2)
def t_xlinks(ctx):
  """Links made on node A resolve on node B; nlink agrees."""
  n = len(ctx.hosts)
  ctx.ok(0, 'write_file', path=ctx.p('t'), size=MiB, seed=3)
  ctx.ok(0, 'link', src=ctx.p('t'), dst=ctx.p('hl'))
  ctx.ok(0, 'symlink', target='t', path=ctx.p('sl'))
  for i in range(1, n):
    def f(i=i):
      r1 = ctx.call(i, 'verify_file', path=ctx.p('hl'), size=MiB, seed=3)
      r2 = ctx.call(i, 'verify_file', path=ctx.p('sl'), size=MiB, seed=3)
      return (r1['ok'] and r1['ret']['ok'] and r2['ok'] and r2['ret']['ok'],
              [r1.get('err'), r2.get('err')])
    _vis(ctx, i, f, 'links')
    def g(i=i):
      s = ctx.call(i, 'stat', path=ctx.p('hl'))
      return s['ok'] and s['ret']['nlink'] == 2, s.get('ret', {}).get('nlink')
    _vis(ctx, i, g, 'nlink')
  # Write through the hard link on the last node; original sees it.
  ctx.ok(n - 1, 'write_file', path=ctx.p('hl'), size=2 * MiB, seed=4,
         flags='w')
  def h():
    r = ctx.call(0, 'verify_file', path=ctx.p('t'), size=2 * MiB, seed=4)
    return r['ok'] and r['ret']['ok'], r.get('ret') or r.get('err')
  _vis(ctx, 0, h, 'link-write')


@test('xnode_tag_names', 'dist', min_nodes=2, timeout=1800)
def t_tag_names(ctx):
  """The CTE tag namespace mirrors clio-fs paths on every node: create,
  rename (a directory with a subtree), cross-directory move, rename over an
  existing file, unlink, rmdir and hard link, issued back to back with no
  pauses, are all reflected by path in every node's tag search index."""
  n = len(ctx.hosts)
  r = ctx.dir
  want = {'/'}

  def mk(i, rel):
    ctx.ok(i, 'mkdir', path=r + rel)
    want.add(rel)

  def wr(i, rel):
    ctx.ok(i, 'write_file', path=r + rel, size=64, seed=len(rel))
    want.add(rel)

  def mv(i, a, b):
    ctx.ok(i, 'rename', src=r + a, dst=r + b)
    for w in [w for w in want if w == a or w.startswith(a + '/')]:
      want.discard(w)
      want.add(b + w[len(a):])
  # Deep tree from node 0, then renames from other nodes, no pauses.
  mk(0, '/d1'); mk(0, '/d1/sub'); mk(0, '/d1/sub/deep')
  wr(0, '/d1/f1'); wr(0, '/d1/sub/f2'); wr(0, '/d1/sub/deep/f3')
  mv(1 % n, '/d1', '/d2')
  mk(1 % n, '/x')
  mv(0, '/d2/sub', '/x/moved')
  wr(0, '/x/victim'); wr(1 % n, '/x/src')
  mv(0, '/x/src', '/x/victim')          # rename over an existing file
  wr(1 % n, '/x/gone'); ctx.ok(0, 'unlink', path=r + '/x/gone')
  mk(0, '/x/rmd'); ctx.ok(1 % n, 'rmdir', path=r + '/x/rmd')
  ctx.ok(0, 'link', src=r + '/d2/f1', dst=r + '/x/f1link')
  want.add('/x/f1link')
  want -= {'/x/gone', '/x/rmd'}
  # A directory per node with many entries, owners spread by hashing.
  for i in range(n):
    mk(i, f'/many{i}')
    res = ctx.ok(i, 'create_many', dirpath=f'{r}/many{i}', prefix='e',
                 count=200)
    ctx.check(not res['fails'], f'create_many node{i}: {res["fails"][:3]}')
    want.update(f'/many{i}/e{k}' for k in range(200))
  mv(0, '/many0', '/x/many0moved')
  for i in range(n):
    _tags_match(ctx, i, r, want, 'after namespace churn')
  # Unlinking the original name leaves the hard link's name.
  ctx.ok(0, 'unlink', path=r + '/d2/f1')
  want.discard('/d2/f1')
  for i in range(n):
    _tags_match(ctx, i, r, want, 'after unlinking a hard-linked name')


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
