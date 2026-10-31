"""Cache-module stress (group 'cache'): cross-node staleness under load.

clio_cte_cache keeps a node-local raw copy of each blob it reads or writes
and promises reads are never stale (write-through: the authoritative put
lands before the ack). That promise is easy to keep on one node; these tests
check it across nodes, under load, and through the operations that change a
file's identity or size rather than just its bytes:

  * many files, cached on every node, overwritten from a rotating writer
    while every other node keeps re-reading them;
  * truncate shrink/extend while other nodes hold cached copies of the cut
    pages (GetBlobSize answers from the cache first);
  * unlink and recreate under the same name with different size/content;
  * a working set larger than the fast tiers, so cached copies are evicted
    and re-populated while overwrites continue.

Every read is pattern-verified against the generation that must be visible
after the fsynced write that preceded it. A read that matches an OLDER
generation is STALE -- the defect these tests exist for -- and is reported
separately from errors and hangs.
"""

import random
import time

from placement import MiB
from suite import test
from tests_replcache import _verify

KiB = 1 << 10


def _seed(fid, gen):
  return 90000 + 1000 * fid + gen


def _classify(ctx, i, path, size, fid, gen):
  """'ok', 'hang', 'err ...', or 'STALE gen<k>' / 'WRONG ...'."""
  st = _verify(ctx, i, path, size, _seed(fid, gen))
  if not st.startswith('WRONG'):
    return st
  for old in range(gen - 1, -1, -1):  # which older generation did it serve?
    r = ctx.call(i, 'verify_file', path=path, size=size,
                 seed=_seed(fid, old), timeout=600)
    if r.get('ok') and r['ret'].get('ok'):
      return f'STALE gen{old} (wanted gen{gen})'
  return st


def _sizes(rng, base):
  """File sizes that cross page and chunk boundaries."""
  return rng.choice([base, base // 2, base + 4 * KiB + 1, base * 2 - 7,
                     64 * KiB, 1])


def _storm(ctx, nfiles, base, rounds, tag):
  """Cross-node overwrite storm. Each round: a rotating node overwrites a
  random subset of files (new generation, new size, fsync); then every node
  reads every touched file and must see that generation. Untouched files are
  re-read too, so cached copies stay hot between rounds."""
  n = len(ctx.hosts)
  rng = random.Random(f'cache:{tag}')
  root = ctx.p(tag)
  ctx.ok(0, 'mkdir', path=root)
  gen = [0] * nfiles
  size = [base] * nfiles
  for f in range(nfiles):
    ctx.ok(f % n, 'write_file', path=f'{root}/f{f}', size=base,
           seed=_seed(f, 0), fsync=True, timeout=900)
  time.sleep(1.5)  # namespace visible everywhere
  problems = []

  def read_all(i, which):
    out = []
    for f in which:
      st = _classify(ctx, i, f'{root}/f{f}', size[f], f, gen[f])
      if st != 'ok':
        out.append((i, f, gen[f], st))
    return out

  # Warm every node's cache with every file.
  for res in ctx.each(lambda i: read_all(i, range(nfiles))):
    problems += res
  for r in range(rounds):
    w = r % n
    touched = rng.sample(range(nfiles), max(1, nfiles // 3))
    for f in touched:
      gen[f] += 1
      size[f] = _sizes(rng, base)
      ctx.ok(w, 'write_file', path=f'{root}/f{f}', size=size[f],
             seed=_seed(f, gen[f]), fsync=True, timeout=900)
    for res in ctx.each(lambda i: read_all(i, range(nfiles))):
      problems += res
    if problems:
      break
  ctx.metrics[f'{tag}_rounds'] = r + 1
  ctx.metrics[f'{tag}_reads'] = (r + 2) * n * nfiles
  stale = [p for p in problems if str(p[3]).startswith(('STALE', 'WRONG'))]
  ctx.metrics[f'{tag}_stale'] = len(stale)
  ctx.check(not stale, f'{len(stale)} reads returned old or wrong bytes '
                       f'after an fsynced overwrite on another node: '
                       f'{stale[:4]}')
  ctx.check(not problems, f'{len(problems)} reads failed: {problems[:4]}')


@test('cache_crossnode_overwrite_storm', 'cache', min_nodes=2, timeout=5400)
def t_overwrite_storm(ctx):
  """24 files of 4 MiB, cached on every node, overwritten (new size, new
  bytes, fsync) from a different node each round for 30 rounds; after each
  round every node must read exactly the newest generation of every file."""
  _storm(ctx, nfiles=24, base=4 * MiB, rounds=30, tag='storm')


@test('cache_overwrite_storm_small_files', 'cache', min_nodes=2,
      timeout=5400)
def t_overwrite_storm_small(ctx):
  """The same storm on 200 small files (64 KiB, sizes down to 1 byte):
  metadata and GetBlobSize answered from the cache dominate here."""
  _storm(ctx, nfiles=200, base=64 * KiB, rounds=15, tag='small')


@test('cache_storm_evicting_working_set', 'cache', min_nodes=2,
      timeout=7200)
def t_storm_evicting(ctx):
  """The storm with a working set ~1.5x the RAM + fast tiers of a node, so
  cached copies are evicted and re-populated between and during rounds."""
  cl = ctx.cl
  fast = (cl.ram_mb + cl.fast_mb if cl.profile in
          ('tiered', 'safe', 'ares_tiers') else cl.ram_gb * 1024)
  base = 32 * MiB
  nfiles = max(8, int(fast * 1.5) // 32)
  ctx.metrics['working_set_mb'] = nfiles * 32
  _storm(ctx, nfiles=nfiles, base=base, rounds=6, tag='evict')


@test('cache_truncate_with_cached_copies', 'cache', min_nodes=2,
      timeout=3600)
def t_truncate(ctx):
  """Every node caches a 16 MiB file; node0 shrinks it to 5 MiB + 3 bytes,
  then extends it back to 16 MiB. Every node must see the new size each
  time, and the extended region must read as zeros (not the cached old
  bytes of the cut pages)."""
  n = len(ctx.hosts)
  path = ctx.p('trunc')
  big, small = 16 * MiB, 5 * MiB + 3
  ctx.ok(0, 'write_file', path=path, size=big, seed=_seed(0, 0), fsync=True,
         timeout=900)
  time.sleep(1.5)
  for st in ctx.each(lambda i: _verify(ctx, i, path, big, _seed(0, 0))):
    ctx.check(st == 'ok', f'initial read: {st}')
  ctx.ok(0, 'truncate', path=path, size=small)
  ctx.ok(0, 'sh', cmd=f'sync -f "{path}" 2>/dev/null; true')
  sizes = ctx.each(lambda i: ctx.ok(i, 'stat', path=path)['size'])
  ctx.check(all(s == small for s in sizes),
            f'size after shrink, per node: {sizes} (wanted {small})')
  ctx.ok(0, 'truncate', path=path, size=big)
  sizes = ctx.each(lambda i: ctx.ok(i, 'stat', path=path)['size'])
  ctx.check(all(s == big for s in sizes),
            f'size after extend, per node: {sizes} (wanted {big})')
  zero = bytes(64)
  bad = []
  for i in range(n):
    for off in (small, small + 4093, 8 * MiB, big - 64):
      got = bytes.fromhex(ctx.ok(i, 'read_hex', path=path, off=off,
                                 length=64))
      if got != zero[:len(got)] or len(got) != 64:
        bad.append((i, off, got[:16].hex()))
  ctx.check(not bad, f'extended region is not zero (old cached bytes?): '
                     f'{bad[:4]}')


@test('cache_unlink_recreate_same_name', 'cache', min_nodes=2, timeout=3600)
def t_unlink_recreate(ctx):
  """A file cached on every node is unlinked and recreated under the same
  name from another node, 20 times, with a different size and content each
  time. A node must never read the previous incarnation's bytes or size."""
  n = len(ctx.hosts)
  root = ctx.p('recreate')
  ctx.ok(0, 'mkdir', path=root)
  path = f'{root}/x'
  rng = random.Random('recreate')
  problems = []
  size = 2 * MiB
  ctx.ok(0, 'write_file', path=path, size=size, seed=_seed(7, 0), fsync=True,
         timeout=900)
  for gen in range(1, 21):
    time.sleep(1.5)
    for i, st in enumerate(ctx.each(
        lambda i: _classify(ctx, i, path, size, 7, gen - 1))):
      if st != 'ok':
        problems.append((i, gen - 1, st))
    w = gen % n
    ctx.ok(w, 'unlink', path=path)
    size = rng.choice([1, 4 * KiB, 1 * MiB + 1, 3 * MiB, 6 * MiB - 5])
    ctx.ok((w + 1) % n, 'write_file', path=path, size=size,
           seed=_seed(7, gen), fsync=True, timeout=900)
  time.sleep(1.5)
  for i, st in enumerate(ctx.each(
      lambda i: _classify(ctx, i, path, size, 7, 20))):
    if st != 'ok':
      problems.append((i, 20, st))
  stale = [p for p in problems if str(p[2]).startswith(('STALE', 'WRONG'))]
  ctx.check(not stale, f'{len(stale)} reads served a previous incarnation: '
                       f'{stale[:4]}')
  ctx.check(not problems, f'{len(problems)} reads failed: {problems[:4]}')
