"""Scaling measurements (mdtest / IOR shaped) -- they PASS as long as every
operation is correct; the numbers go to the metrics column of the report.

Run at 1, 2, 4, 8 nodes to see how metadata rate and bandwidth scale.
"""

import time

from suite import test

MiB = 1 << 20


def _rate(n_ops, t0):
  dt = time.time() - t0
  return round(n_ops / dt, 1) if dt > 0 else 0.0


@test('scale_metadata_private_dirs', 'perf', timeout=3600)
def t_md_private(ctx):
  """Each node: create / stat / unlink 2000 empty files in its OWN dir."""
  n = len(ctx.hosts)
  per = 2000
  for i in range(n):
    ctx.ok(i, 'makedirs', path=ctx.p(f'n{i}'))
  t0 = time.time()
  rs = ctx.each(lambda i: ctx.ok(i, 'create_many', dirpath=ctx.p(f'n{i}'),
                                 prefix='f', count=per, timeout=3000))
  ctx.metrics['create_ops_s'] = _rate(n * per, t0)
  ctx.check(all(not r['fails'] for r in rs), f'create failures {rs}')
  t0 = time.time()
  rs = ctx.each(lambda i: ctx.ok(i, 'stat_many', dirpath=ctx.p(f'n{i}'),
                                 prefix='f', count=per, size=0, timeout=3000))
  ctx.metrics['stat_ops_s'] = _rate(n * per, t0)
  ctx.check(all(not r['bad'] for r in rs), f'stat failures {rs}')
  t0 = time.time()
  rs = ctx.each(lambda i: ctx.ok(i, 'unlink_many', dirpath=ctx.p(f'n{i}'),
                                 prefix='f', count=per, timeout=3000))
  ctx.metrics['unlink_ops_s'] = _rate(n * per, t0)
  ctx.check(all(not r['fails'] for r in rs), f'unlink failures {rs}')


@test('scale_metadata_shared_dir', 'perf', timeout=3600)
def t_md_shared(ctx):
  """All nodes create 1000 files each in ONE shared dir, then readdir it."""
  n = len(ctx.hosts)
  per = 1000
  d = ctx.p('shared')
  ctx.ok(0, 'mkdir', path=d)
  time.sleep(1.2)
  t0 = time.time()
  rs = ctx.each(lambda i: ctx.ok(i, 'create_many', dirpath=d, prefix=f'n{i}_',
                                 count=per, timeout=3000))
  ctx.metrics['create_ops_s'] = _rate(n * per, t0)
  ctx.check(all(not r['fails'] for r in rs), f'create failures {rs}')
  time.sleep(1.2)
  t0 = time.time()
  counts = ctx.each(lambda i: ctx.ok(i, 'scandir_count', path=d, timeout=600))
  ctx.metrics['readdir_s_per_node'] = round((time.time() - t0), 2)
  ctx.check(all(c == n * per for c in counts), f'readdir counts {counts}')


@test('scale_fpp_bandwidth', 'perf', timeout=3600)
def t_fpp(ctx):
  """File per node, 1 GiB each, 4 MiB writes; then read back a NEIGHBOR's."""
  n = len(ctx.hosts)
  size = 1024 * MiB
  t0 = time.time()
  ctx.each(lambda i: ctx.ok(i, 'write_file', path=ctx.p(f'f{i}'), size=size,
                            seed=i, chunk=4 * MiB, fsync=True, timeout=3000))
  ctx.metrics['write_MiBps'] = _rate(n * size / MiB, t0)
  t0 = time.time()

  def rd(i):
    j = (i + 1) % n
    v = ctx.ok(i, 'verify_file', path=ctx.p(f'f{j}'), size=size, seed=j,
               chunk=4 * MiB, timeout=3000)
    ctx.check(v['ok'], f'node{i} reading f{j}: {v}')
  ctx.each(rd)
  ctx.metrics['read_verify_MiBps'] = _rate(n * size / MiB, t0)


@test('scale_shared_file_bandwidth', 'perf', timeout=3600)
def t_shared_bw(ctx):
  """One shared file, node i writes 1 MiB stripes i, i+N, ... (256 MiB per
  node); after fsync every node verifies the whole file."""
  n = len(ctx.hosts)
  per = 256
  p = ctx.p('shared.bin')
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  time.sleep(1.2)
  t0 = time.time()

  def wr(i):
    h = ctx.ok(i, 'open', path=p, flags='rw')
    for s in range(i, per * n, n):
      ctx.ok(i, 'fpwrite', h=h, off=s * MiB, length=MiB, seed=7, timeout=300)
    ctx.ok(i, 'fsync', h=h, timeout=600)
    ctx.ok(i, 'close', h=h)
  ctx.each(wr)
  ctx.metrics['write_MiBps'] = _rate(n * per, t0)
  time.sleep(1.2)
  t0 = time.time()

  def rd(i):
    v = ctx.ok(i, 'verify_file', path=p, size=n * per * MiB, seed=7,
               chunk=4 * MiB, timeout=3000)
    ctx.check(v['ok'], f'node{i}: {v}')
  ctx.each(rd)
  ctx.metrics['read_verify_MiBps'] = _rate(n * n * per, t0)
