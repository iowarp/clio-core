"""Tiering performance studies (group 'tier'), run by CI/ares.

Meant for profile 'ares_tiers' (RAM, NVMe, SATA SSD and HDD as separate CTE
targets on every node) and, as the comparison point, the suite's default
'persistent' profile (RAM + one NVMe file tier). Every test is also a
correctness test: all bytes are pattern-checked, and a mismatch FAILs the
test however fast it was. Numbers go to `metrics`; CI/ares/compare.py
compares them against a baseline run to flag regressions.

Sizes scale with the configured tiers so the same test exercises "fits in
RAM", "spills to NVMe" and "spills to SSD/HDD" on any tier layout.
"""

import time

from suite import test
from tests_stress import _durable_cap_mb, _tier_usage, _working_set_mb

KiB = 1 << 10
MiB = 1 << 20


def _fast_mb(cl):
  """MB of the tiers above the slow ones: RAM (+ NVMe on tiered layouts)."""
  if cl.profile in ('tiered', 'safe', 'ares_tiers'):
    return cl.ram_mb + cl.fast_mb
  return cl.ram_gb * 1024


def _all(ctx, fn):
  """Run fn(i) on every node at once; returns the replies in node order."""
  return ctx.each(fn)


def _agg(ctx, prefix, rs):
  """Record aggregate + per-node-worst metrics of op_io_bench replies."""
  ctx.metrics[f'{prefix}_mb_s'] = round(sum(r['mb_s'] for r in rs), 2)
  ctx.metrics[f'{prefix}_iops'] = round(sum(r['iops'] for r in rs), 1)
  ctx.metrics[f'{prefix}_p99_us'] = max(r['lat_us_p99'] for r in rs)
  ctx.check(all(r['bad'] == 0 for r in rs),
            f'{prefix}: {sum(r["bad"] for r in rs)} reads returned wrong '
            f'bytes')


def _bench(ctx, i, path, size, bs, mode, access, seed, fsync=False,
           secs_cap=0):
  return ctx.ok(i, 'io_bench', path=path, size=size, bs=bs, mode=mode,
                access=access, seed=seed, fsync=fsync, secs_cap=secs_cap,
                timeout=3600)


def _drop(ctx):
  """Drop every node's page cache; notes it when that is not permitted."""
  ok = all(ctx.each(lambda i: ctx.call(i, 'drop_caches', timeout=120)
                    .get('ret')))
  if not ok:
    ctx.notes.append('page cache not dropped (no passwordless sudo): '
                     '"cold" reads may be served from the FUSE page cache')
  return ok


@test('tier_small_io', 'tier', timeout=3600)
def t_small_io(ctx):
  """4 KiB requests on every node at once: sequential and random writes
  (fsynced), then sequential and random reads, of a 64 MiB file per node.
  Small-I/O IOPS and tail latency are what metadata and request overheads
  show up in first."""
  size = 64 * MiB
  bs = 4 * KiB
  for access in ('seq', 'rand'):
    rs = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'small_{access}_{i}'),
                                    size, bs, 'write', access, 11 + i,
                                    fsync=True, secs_cap=300))
    _agg(ctx, f'w4k_{access}', rs)
  for access in ('seq', 'rand'):
    # Read back what the seq write produced (complete, unlike a capped
    # rand write that may have stopped early).
    rs = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'small_seq_{i}'),
                                    size, bs, 'read', access, 11 + i,
                                    secs_cap=300))
    _agg(ctx, f'r4k_{access}', rs)


@test('tier_large_io', 'tier', timeout=5400)
def t_large_io(ctx):
  """4 MiB sequential requests, every node streaming a file the size of its
  RAM tier (so the flush to the lower tiers is in the measured window),
  fsynced; then read back cold and hot."""
  size = min(max(512, _fast_mb(ctx.cl) // 2), _durable_cap_mb(ctx)) * MiB
  bs = 4 * MiB
  ctx.metrics['file_mb_per_node'] = size // MiB
  rs = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'large_{i}'), size, bs,
                                  'write', 'seq', 21 + i, fsync=True))
  _agg(ctx, 'w4m_seq', rs)
  _drop(ctx)
  rs = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'large_{i}'), size, bs,
                                  'read', 'seq', 21 + i))
  _agg(ctx, 'r4m_cold', rs)
  rs = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'large_{i}'), size, bs,
                                  'read', 'seq', 21 + i))
  _agg(ctx, 'r4m_hot', rs)
  ctx.metrics['tier_usage'] = _tier_usage(ctx)


@test('tier_spill_to_slow_tiers', 'tier', timeout=7200)
def t_spill(ctx):
  """Write more per node than RAM + NVMe hold, so data has to land on (and
  be read back from) the SSD and HDD tiers. Measures write bandwidth across
  the spill, cold read of the oldest (most demoted) data, and the newest;
  tier_usage shows where the bytes actually went."""
  cl = ctx.cl
  per_file = 256 * MiB
  total = min(_working_set_mb(ctx, 1.5), _durable_cap_mb(ctx)) * MiB
  if total // MiB <= _fast_mb(cl):
    ctx.note('durable capacity caps the write below the fast tiers: '
             'this layout cannot spill with every copy kept')
  nfiles = max(2, total // per_file)
  ctx.metrics['spill_mb_per_node'] = nfiles * per_file // MiB
  for i in range(len(ctx.hosts)):
    ctx.ok(i, 'makedirs', path=ctx.p(f'n{i}'))

  def write_all(i):
    out = []
    for k in range(nfiles):
      out.append(_bench(ctx, i, ctx.p(f'n{i}', f'f{k}'), per_file, 4 * MiB,
                        'write', 'seq', 1000 * i + k, fsync=True))
    return out

  t0 = time.time()
  rs = _all(ctx, write_all)
  dt = time.time() - t0
  ctx.metrics['spill_write_mb_s'] = round(
      len(ctx.hosts) * nfiles * per_file / dt / MiB, 2)
  ctx.metrics['spill_write_slowest_file_mb_s'] = min(
      r['mb_s'] for node in rs for r in node)
  time.sleep(10)  # let the organizer settle placement
  ctx.metrics['tier_usage'] = _tier_usage(ctx)
  _drop(ctx)
  oldest = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'n{i}', 'f0'), per_file,
                                      4 * MiB, 'read', 'seq', 1000 * i))
  _agg(ctx, 'read_oldest', oldest)
  newest = _all(ctx, lambda i: _bench(
      ctx, i, ctx.p(f'n{i}', f'f{nfiles - 1}'), per_file, 4 * MiB, 'read',
      'seq', 1000 * i + nfiles - 1))
  _agg(ctx, 'read_newest', newest)


@test('tier_hot_set_under_streaming', 'tier', timeout=5400)
def t_hot_set(ctx):
  """A small hot file is read over and over while a large cold stream is
  written past it. A healthy organizer keeps the hot set in the fast tiers,
  so its tail latency should stay flat; a regression shows up as hot reads
  slowing to SSD/HDD speed."""
  cl = ctx.cl
  hot = 32 * MiB
  cold = min(_working_set_mb(ctx, 1.2), _durable_cap_mb(ctx) - 64) * MiB
  for i in range(len(ctx.hosts)):
    ctx.ok(i, 'makedirs', path=ctx.p(f'n{i}'))
  _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'n{i}', 'hot'), hot, 64 * KiB,
                             'write', 'seq', 500 + i, fsync=True))
  before = _all(ctx, lambda i: _bench(ctx, i, ctx.p(f'n{i}', 'hot'), hot,
                                      64 * KiB, 'read', 'rand', 500 + i))
  _agg(ctx, 'hot_before', before)

  def stream_and_probe(i):
    # Alternate: one cold chunk written, one full random pass over the hot
    # file -- latency samples taken while the fast tiers are under pressure.
    probes = []
    chunk = 256 * MiB
    for k in range(max(1, cold // chunk)):
      _bench(ctx, i, ctx.p(f'n{i}', f'cold{k}'), chunk, 4 * MiB, 'write',
             'seq', 600 + 100 * i + k, fsync=True)
      probes.append(_bench(ctx, i, ctx.p(f'n{i}', 'hot'), hot, 64 * KiB,
                           'read', 'rand', 500 + i))
    return probes

  rs = _all(ctx, stream_and_probe)
  last = [node[-1] for node in rs]
  _agg(ctx, 'hot_under_stream', last)
  ctx.metrics['tier_usage'] = _tier_usage(ctx)


def _clock_offset(ctx, i, rounds=5):
  """Estimate node i's wall-clock offset from the driver (seconds): remote
  time minus the midpoint of the round trip, best of `rounds` (shortest
  round trip). Ares nodes have been seen ~27 s apart, so cross-node lag is
  meaningless without this."""
  best = None
  for _ in range(rounds):
    t0 = time.time()
    remote = ctx.ok(i, 'clock')
    t1 = time.time()
    if best is None or t1 - t0 < best[0]:
      best = (t1 - t0, remote - (t0 + t1) / 2)
  return best[1]


def _pipeline(ctx, tag, count, item_size):
  """Producer on node 0, consumer on the last node (the same node when
  there is only one); returns (producer, consumer) replies."""
  n = len(ctx.hosts)
  c = n - 1
  d = ctx.p(tag)
  ctx.ok(0, 'mkdir', path=d)
  time.sleep(1.5)  # let the directory become visible on the consumer
  out = {}

  def role(i):
    # The consumer gives up after timeout_s and reports what it had not
    # seen as "missing". Under the slow-tier mixes the whole study has
    # taken 2500-2900 s, so a 3000 s window turned slowness into apparent
    # loss (tiermix5 balanced: 67 missing, 0 corrupt, at 3068 s). Give it
    # room so "missing" means missing; the time itself is a metric.
    if i == 0:
      out['p'] = ctx.ok(0, 'produce', dirpath=d, count=count,
                        item_size=item_size, seed=7000, timeout=7000)
    if i == c:
      out['c'] = ctx.ok(c, 'consume', dirpath=d, count=count,
                        item_size=item_size, seed=7000, timeout_s=6000,
                        timeout=7000)

  ctx.each(role) if n > 1 else (role(0))
  return out['p'], out['c']


@test('tier_producer_consumer', 'tier', timeout=14400)
def t_producer_consumer(ctx):
  """Producer-consumer over clio-fs: one node publishes fsynced items by
  atomic rename, another waits for each, reads and verifies it. Small
  (64 KiB) and large (8 MiB) items. Throughput is end to end; lag is
  publish-to-read (wall clocks, so it assumes NTP-synced nodes)."""
  for tag, count, size in (('pc_small', 2000, 64 * KiB),
                           ('pc_large', 200, 8 * MiB)):
    p, c = _pipeline(ctx, tag, count, size)
    # lag was measured as consumer clock - producer clock; remove the skew.
    skew_ms = (_clock_offset(ctx, len(ctx.hosts) - 1) -
               _clock_offset(ctx, 0)) * 1e3
    ctx.metrics['clock_skew_ms'] = round(skew_ms, 1)
    for k in ('lag_ms_p50', 'lag_ms_p99'):
      if c.get(k) is not None:
        c[k] = round(c[k] - skew_ms, 2)
    ctx.metrics[f'{tag}_produce_mb_s'] = p['mb_s']
    ctx.metrics[f'{tag}_consume_mb_s'] = c['mb_s']
    ctx.metrics[f'{tag}_lag_ms_p50'] = c['lag_ms_p50']
    ctx.metrics[f'{tag}_lag_ms_p99'] = c['lag_ms_p99']
    ctx.check(c['bad'] == 0 and c['missing'] == 0,
              f'{tag}: consumer saw {c["bad"]} corrupt and {c["missing"]} '
              f'missing items of {count}')
