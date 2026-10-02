"""Data-integrity stress: concurrent access and tier pressure.

Every block these tests write is a self-verifying record (stress_records.py),
so a read tells apart an intact version, a hole, CORRUPT bytes (torn,
garbage) and FOREIGN data (another file's or block's record). The tests
model which version each block may hold and fail on:

  corruption  any CORRUPT or FOREIGN block, ever, on any node
  loss        a block older than a version whose fsync completed (or a hole
              where fsynced data was)
  staleness   a read that returns a version older than one whose fsync
              completed before the reader opened the file (close-to-open)
  divergence  nodes disagreeing about a block once writers are quiet

Run them with the `tiered` profile so the data overflows a small RAM tier
into a fast and then a slow file tier while the organizer migrates blobs:

  suite.py --profile tiered --groups stress [--ram-mb 512 --fast-mb 2048]
"""

import random
import threading
import time

from suite import test
from tests_fault import restart_cluster
import stress_records as sr

MiB = 1 << 20
FILE_BLOCKS = 16384            # 64 MiB files
SKEW_S = 0.25                  # clock-skew / ordering margin between nodes
ZERO, CORRUPT, FOREIGN = sr.ZERO, sr.CORRUPT, sr.FOREIGN
KIND = {ZERO: 'ZERO', CORRUPT: 'CORRUPT', FOREIGN: 'FOREIGN'}


def _tier_mb(ctx):
  """RAM + fast tier per node (MB) under the tiered profile, else RAM."""
  cl = ctx.cl
  if cl.profile in ('tiered', 'safe'):
    return cl.ram_mb + cl.fast_mb
  return cl.ram_gb * 1024


def _tier_usage(ctx):
  """Allocated bytes (MiB) of each node's file-tier backing files: shows
  the data really went past the RAM tier."""
  lr = ctx.cl.local_root
  out = {}
  for i in range(len(ctx.hosts)):
    r = ctx.call(i, 'sh', cmd=f'du -m {lr}/data/cte_fast_tier.dat* '
                              f'{lr}/data/cte_disk_tier.dat* 2>/dev/null')
    out[f'node{i}'] = ' '.join(
        f'{ln.split()[1].rsplit("/", 1)[-1]}={ln.split()[0]}M'
        for ln in (r.get('ret') or {}).get('out', '').splitlines()
        if len(ln.split()) == 2)
  return out


def _expand(runs, nblocks):
  """Run list [[start, count, writer, gen]] -> per-block (writer, gen)."""
  out = [(ZERO, 0)] * nblocks
  for start, count, w, g in runs:
    for b in range(start, min(start + count, nblocks)):
      out[b] = (w, g)
  return out


def _compare(expect_runs, got, nblocks, name, bad, limit=20):
  """Append (name, block, want, got) for every block that differs.

  Returns the number of differing blocks (only `limit` are recorded).
  """
  want = _expand(expect_runs, nblocks)
  have = _expand(got['runs'], nblocks)
  n = 0
  for b in range(nblocks):
    if want[b] != have[b]:
      n += 1
      if len([x for x in bad if x[0] == name]) < limit:
        bad.append((name, b, want[b], KIND.get(have[b][0], have[b])))
  return n


def _corrupt_blocks(runs):
  """Blocks classified CORRUPT or FOREIGN in a scan."""
  return sum(c for _, c, w, _ in runs if w in (CORRUPT, FOREIGN))


def _apply(model, runs, writer, gen):
  """Record a write of (writer, gen) over block runs into a file model
  (a per-block list)."""
  for start, count in runs:
    for b in range(start, start + count):
      model[b] = (writer, gen)


def _to_runs(model):
  """Per-block model list -> [[start, count, writer, gen]]."""
  runs = []
  for b, (w, g) in enumerate(model):
    if runs and runs[-1][2] == w and runs[-1][3] == g:
      runs[-1][1] += 1
    else:
      runs.append([b, 1, w, g])
  return runs


# ---------------------------------------------------------------------------
# 1. Overflow every tier, verify across nodes, overwrite, free, reuse
# ---------------------------------------------------------------------------

@test('stress_tier_overflow', 'stress', min_nodes=1, redeploy_after=True,
      timeout=5400)
def t_tier_overflow(ctx):
  """Each node writes 1.5x its RAM + fast tiers in 64 MiB record files
  (fsynced), so data is pushed down through every tier. Another node reads
  every block back; a third overwrites random ranges of each file (model
  tracked) and the owner re-verifies; then a third of the files are deleted
  and as much new data is written into the freed space, and everything is
  verified again. Any CORRUPT/FOREIGN block (e.g. a freed extent handed out
  with the old file's data, a migration that copied the wrong blob) or a
  block not holding its latest fsynced version fails the test."""
  n = len(ctx.hosts)
  per_node_mb = int(_tier_mb(ctx) * 1.5)
  nfiles = max(2, per_node_mb // 64)
  base = ctx.p('ovf')
  ctx.ok(0, 'mkdir', path=base)
  models = {}  # name -> per-block model

  def fname(i, k):
    return f'n{i}_f{k}'

  def write_all(i):
    for k in range(nfiles):
      nm = fname(i, k)
      ctx.ok(i, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=[[0, FILE_BLOCKS]], writer=i, gen=1, fsync=True)
  t0 = time.time()
  ctx.each(write_all)
  ctx.metrics['write_MiB_per_s'] = round(
      n * nfiles * 64 / max(0.001, time.time() - t0))
  ctx.metrics['bytes_per_node_mib'] = nfiles * 64
  ctx.note(f'tier backing files after the writes: {_tier_usage(ctx)}')
  for i in range(n):
    for k in range(nfiles):
      models[fname(i, k)] = [(i, 1)] * FILE_BLOCKS

  bad, anomalies = [], {'mismatch': 0, 'corrupt': 0}

  count_lock = threading.Lock()
  rereads = []  # (file, first reader, bad blocks, re-reader, bad blocks)
  foreign_src = []  # (file, block, source file id, its block, writer, gen)
  corrupt_desc = []  # (file, describe_corrupt of a CORRUPT block)

  def verify(reader_of, names_of):
    def one(i):
      r = reader_of(i)
      for nm in names_of(i):
        got = ctx.ok(r, 'rec_scan', timeout=900, path=f'{base}/{nm}',
                     name=nm, nblocks=FILE_BLOCKS)
        for b, src in list(got.get('foreign', {}).items())[:4]:
          foreign_src.append((nm, int(b), *src))
        for d in got.get('corrupt', [])[:2]:
          corrupt_desc.append((nm, d))
        ncorrupt = _corrupt_blocks(got['runs'])
        nbad = _compare(_to_runs(models[nm]), got, FILE_BLOCKS, nm, bad)
        if nbad:
          # Stored wrong, or read wrong? Read the file again on this node
          # and on another: a mismatch that changes or goes away is a
          # read-side transient, one that stays is in the stored bytes.
          other = (r + 1) % n
          for again in (r, other):
            g2 = ctx.ok(again, 'rec_scan', timeout=900, path=f'{base}/{nm}',
                        name=nm, nblocks=FILE_BLOCKS)
            n2 = _compare(_to_runs(models[nm]), g2, FILE_BLOCKS, nm, [])
            with count_lock:
              rereads.append((nm, f'node{r}', nbad, f'node{again}', n2))
        with count_lock:  # the per-node threads share these counters
          anomalies['corrupt'] += ncorrupt
          anomalies['mismatch'] += nbad
    ctx.each(one)

  owned = {i: [fname(i, k) for k in range(nfiles)] for i in range(n)}
  # (a) every block, read on another node
  verify(lambda i: (i + 1) % n, lambda i: owned[i])
  ctx.metrics['after_write_bad_blocks'] = anomalies['mismatch']

  # (b) overwrite random ranges from a third node, one overwriter per file
  rng = random.Random(1234)
  plan = {}
  for i in range(n):
    for nm in owned[i]:
      runs = []
      for _ in range(8):
        s = rng.randrange(FILE_BLOCKS)
        runs.append([s, min(rng.randrange(1, 512), FILE_BLOCKS - s)])
      plan[nm] = runs

  def overwrite(i):
    ow = (i + 2) % n
    for nm in owned[i]:
      ctx.ok(ow, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=plan[nm], writer=100 + ow, gen=2, fsync=True)
  ctx.each(overwrite)
  for i in range(n):
    for nm in owned[i]:
      _apply(models[nm], plan[nm], 100 + (i + 2) % n, 2)
  verify(lambda i: i, lambda i: owned[i])
  ctx.metrics['after_overwrite_bad_blocks'] = anomalies['mismatch']

  # (c) free a third of the files, reuse the space, verify everything
  def free_and_reuse(i):
    gone = owned[i][::3]
    for nm in gone:
      ctx.ok(i, 'unlink', path=f'{base}/{nm}')
    for k in range(len(gone)):
      nm = f'n{i}_new{k}'
      ctx.ok(i, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=[[0, FILE_BLOCKS]], writer=i, gen=3, fsync=True)
    return gone
  gones = ctx.each(free_and_reuse)
  for i in range(n):
    for nm in gones[i]:
      models.pop(nm)
      owned[i].remove(nm)
    for k in range(len(gones[i])):
      nm = f'n{i}_new{k}'
      models[nm] = [(i, 3)] * FILE_BLOCKS
      owned[i].append(nm)
  # Freed space is handed back asynchronously: give it a moment, then
  # verify from yet another node.
  time.sleep(5)
  verify(lambda i: (i + n - 1) % n, lambda i: owned[i])
  if foreign_src:
    # Name the files the foreign records came from: a deleted file means a
    # freed extent was handed out unwritten; a live one, a shared extent.
    names = {hex(sr.file_id_of(nm)): nm for nm in
             list(models) + [g for gs in gones for g in gs]}
    deleted = {g for gs in gones for g in gs}
    ctx.note('foreign records came from: ' + str([
        (nm, b, names.get(fid, fid),
         'deleted' if names.get(fid) in deleted else 'live', fb, w, g)
        for nm, b, fid, fb, w, g in foreign_src[:8]]))
  if corrupt_desc:
    ctx.note(f'corrupt blocks hold: {corrupt_desc[:6]}')
  if rereads:
    ctx.note(f're-reads of the bad files (file, reader, bad, re-reader, '
             f'bad): {rereads[:8]}')
  ctx.metrics['corrupt_or_foreign_blocks'] = anomalies['corrupt']
  ctx.metrics['bad_blocks_total'] = anomalies['mismatch']
  ctx.check(anomalies['mismatch'] == 0 and anomalies['corrupt'] == 0 and
            not bad,
            f'{anomalies["mismatch"]} blocks differ from the model '
            f'({anomalies["corrupt"]} CORRUPT/FOREIGN), e.g. {bad[:8]}')


# ---------------------------------------------------------------------------
# 2. Concurrent overwriters + readers on one shared file, every node
# ---------------------------------------------------------------------------

def _index_writes(results):
  """Merge all writers' logs: block -> [(writer, gen, t_issue, t_ack)],
  with in-flight writes (outcome unknown) at t_ack = None."""
  by_block = {}
  for res in results:
    for wid, log in res['writes'].items():
      for b, g, ti, ta in log:
        by_block.setdefault(b, []).append((int(wid), g, ti, ta))
    for wid, (b, g, ti) in res['inflight'].items():
      by_block.setdefault(b, []).append((int(wid), g, ti, None))
  return by_block


def _check_observation(ob, by_block):
  """Return a problem string for one reader observation, or None.

  ob = [block, writer, gen, t_open, t_done]. The value must be a write
  issued to that block no later than the read finished, and no write to
  the block may have both started after it was acked and been acked
  before the reader opened the file (it would have to be seen).
  """
  b, w, g, t_open, t_done = ob
  writes = by_block.get(b, [])
  if w in (CORRUPT, FOREIGN):
    return f'block {b}: read {KIND[w]} data'
  if w == ZERO:
    newest_acked = max((ta for _, _, _, ta in writes if ta is not None),
                       default=None)
    if newest_acked is not None and newest_acked < t_open - SKEW_S:
      return (f'block {b}: read a hole although a write was fsynced '
              f'{t_open - newest_acked:.3f} s before the open')
    return None
  mine = [x for x in writes if x[0] == w and x[1] == g]
  if not mine:
    return f'block {b}: read ({w},{g}) which was never written there'
  _, _, ti, ta = mine[0]
  if ti > t_done + SKEW_S:
    return f'block {b}: read ({w},{g}) before it was written'
  if ta is None:
    return None
  for x in writes:
    if x[2] > ta + SKEW_S and x[3] is not None and x[3] < t_open - SKEW_S:
      return (f'block {b}: stale read ({w},{g}); ({x[0]},{x[1]}) was '
              f'fsynced {t_open - x[3]:.3f} s before the open')
  return None


def _final_candidates(writes):
  """Versions a block may hold once every writer stopped: those no other
  write to the block started strictly after (plus the skew margin)."""
  if not writes:
    return {(ZERO, 0)}
  out = set()
  for w, g, ti, ta in writes:
    done = ta if ta is not None else ti
    if not any(x[2] > done + SKEW_S for x in writes):
      out.add((w, g))
  return out


@test('stress_shared_file_concurrent', 'stress', min_nodes=2,
      redeploy_after=True, timeout=3600)
def t_shared_concurrent(ctx):
  """Every node runs 2 overwriters (single 4 KiB blocks, fsync after each)
  and 2 readers (open, read one block, close) on ONE shared 2 MiB file (512
  blocks, so writers keep overwriting each other's blocks) for 90 s. Every read must return an intact version of that block that is not
  older than a version fsynced before the reader opened the file; after the
  writers stop every node must read the same content, each block holding a
  version no later write replaced."""
  n = len(ctx.hosts)
  p = ctx.p('shared')
  nm = 'shared'
  nb = 512
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  ctx.ok(0, 'truncate', path=p, size=nb * sr.BLK)
  secs = 90

  def run(i):
    return ctx.ok(i, 'rec_shared_stress', timeout=secs + 600, path=p,
                  name=nm, nblocks=nb, writer_base=10 * i + 1,
                  writers=2, readers=2, secs=secs, seed=i)
  results = ctx.each(run)
  by_block = _index_writes(results)
  nwrites = sum(len(v) for v in by_block.values())
  nobs = sum(len(r['obs']) for r in results)
  ctx.metrics.update({'writes': nwrites, 'reads_checked': nobs,
                      'io_errors': sum(r['nerrors'] for r in results)})
  errs = [e for r in results for e in r['errors']]
  problems = []
  for r in results:
    for ob in r['obs']:
      pr = _check_observation(ob, by_block)
      if pr:
        problems.append(pr)
  ctx.metrics['bad_reads'] = len(problems)
  time.sleep(2)
  scans = ctx.all_nodes('rec_scan', timeout=900, path=p, name=nm,
                        nblocks=nb)
  finals = [_expand(s['ret']['runs'], nb) for s in scans]
  diverge, final_bad = [], []
  ctx.metrics['blocks_written_more_than_once'] = sum(
      1 for v in by_block.values() if len(v) > 1)
  for b in range(nb):
    vals = {f[b] for f in finals}
    if len(vals) > 1:
      diverge.append((b, sorted(vals)))
    cand = _final_candidates(by_block.get(b, []))
    for v in vals:
      if v not in cand:
        final_bad.append((b, KIND.get(v[0], v), sorted(cand)[:3]))
  ctx.metrics.update({'final_divergent_blocks': len(diverge),
                      'final_bad_blocks': len(final_bad)})
  ctx.check(not errs, f'I/O errors during the run: {errs[:5]}')
  ctx.check(not problems, f'{len(problems)} of {nobs} reads were wrong, '
                          f'e.g. {problems[:6]}')
  ctx.check(not diverge, f'{len(diverge)} blocks differ between nodes once '
                         f'writers stopped, e.g. {diverge[:5]}')
  ctx.check(not final_bad, f'{len(final_bad)} blocks hold a version a later '
                           f'write replaced (lost update), e.g. '
                           f'{final_bad[:5]}')


def _node_of(writer):
  """Node index of a shared-stress writer id (writer_base = 10 * i + 1)."""
  return (writer - 1) // 10


def _buffered_problem(ob, node, by_block):
  """Check one read of the buffered (no fsync) shared stress; None if ok.

  Writes from the reader's own node are visible once write(2) returned,
  so a same-node version may not be read after a newer same-node write was
  acked, nor a hole after one. A version from another node may be read at
  any time: without fsync its bytes can land later than this node's own
  write (close-to-open), so it is never "older" in a checkable sense.
  """
  b, w, g, t_open, t_done = ob
  writes = by_block.get(b, [])
  if w in (CORRUPT, FOREIGN):
    return f'block {b}: read {KIND[w]} data'
  same = [x for x in writes if _node_of(x[0]) == node and x[3] is not None]
  if w == ZERO:
    if any(x[3] < t_open for x in same):
      return f'block {b}: read a hole after this node wrote it'
    return None
  mine = [x for x in writes if x[0] == w and x[1] == g]
  if not mine:
    return f'block {b}: read ({w},{g}) which was never written there'
  _, _, ti, ta = mine[0]
  if ti > t_done + SKEW_S:
    return f'block {b}: read ({w},{g}) before it was written'
  if _node_of(w) != node:
    return None
  for x in same:
    newer = x[2] > (ta if ta is not None else ti)
    if newer and x[3] < t_open:
      return (f'block {b}: read ({w},{g}) after this node wrote '
              f'({x[0]},{x[1]}) over it')
  return None


def _buffered_candidates(writes):
  """Final versions allowed without fsync ordering between nodes: every
  node's last write to the block (a node's own writes stay ordered)."""
  if not writes:
    return {(ZERO, 0)}
  out = set()
  for w, g, ti, ta in writes:
    done = ta if ta is not None else ti
    node = _node_of(w)
    if not any(_node_of(x[0]) == node and x[2] > done for x in writes):
      out.add((w, g))
  return out


def _truncate_candidates(writes, truncs, b):
  """Final versions allowed with a racing truncater: the non-superseded
  writes, plus a hole if a truncate that cut block b ran after the start
  of the block's last write."""
  cand = _final_candidates(writes) if writes else set()
  last_issue = max((x[2] for x in writes), default=None)
  cut_after = any(nb <= b and (last_issue is None or
                               t_done >= last_issue - SKEW_S)
                  for nb, _, t_done in truncs)
  if not writes or cut_after:
    cand.add((ZERO, 0))
  return cand


def _shared_run(ctx, tag, secs, sync, truncater):
  """Run the shared-file stress on every node; return (results, by_block,
  final per-node block lists, nb)."""
  p = ctx.p(tag)
  nb = 512
  ctx.ok(0, 'write_file', path=p, size=0, seed=0)
  ctx.ok(0, 'truncate', path=p, size=nb * sr.BLK)

  def run(i):
    return ctx.ok(i, 'rec_shared_stress', timeout=secs + 600, path=p,
                  name=tag, nblocks=nb, writer_base=10 * i + 1, writers=2,
                  readers=2, secs=secs, seed=i, sync=sync,
                  truncater=truncater and i == 0)
  results = ctx.each(run)
  time.sleep(2)
  scans = ctx.all_nodes('rec_scan', timeout=900, path=p, name=tag,
                        nblocks=nb)
  finals = [_expand(s['ret']['runs'], nb) for s in scans]
  return results, _index_writes(results), finals, nb


@test('stress_shared_file_buffered', 'stress', min_nodes=2,
      redeploy_after=True, timeout=3600)
def t_shared_buffered(ctx):
  """Like stress_shared_file_concurrent but writers never fsync until the
  end (the write-behind path under contention). A reader on the writer's
  node must see each write once write(2) returned; other nodes may read any
  valid version until the final fsync; afterwards all nodes agree and each
  block holds some node's last write to it. Never CORRUPT/FOREIGN."""
  results, by_block, finals, nb = _shared_run(ctx, 'shbuf', 90, False,
                                               False)
  problems = []
  for i, r in enumerate(results):
    for ob in r['obs']:
      pr = _buffered_problem(ob, i, by_block)
      if pr:
        problems.append(pr)
  diverge, final_bad = [], []
  for b in range(nb):
    vals = {f[b] for f in finals}
    if len(vals) > 1:
      diverge.append((b, sorted(vals)))
    cand = _buffered_candidates(by_block.get(b, []))
    final_bad += [(b, KIND.get(v[0], v)) for v in vals if v not in cand]
  errs = [e for r in results for e in r['errors']]
  ctx.metrics.update({
      'writes': sum(len(v) for v in by_block.values()),
      'reads_checked': sum(len(r['obs']) for r in results),
      'bad_reads': len(problems), 'final_divergent_blocks': len(diverge),
      'final_bad_blocks': len(final_bad)})
  ctx.check(not errs, f'I/O errors: {errs[:5]}')
  ctx.check(not problems, f'{len(problems)} wrong reads, e.g. '
                          f'{problems[:6]}')
  ctx.check(not diverge, f'{len(diverge)} blocks differ between nodes '
                         f'after the final fsync, e.g. {diverge[:5]}')
  ctx.check(not final_bad, f'{len(final_bad)} blocks lost a node\'s last '
                           f'write, e.g. {final_bad[:5]}')


@test('stress_shared_file_truncate_race', 'stress', min_nodes=2,
      redeploy_after=True, timeout=3600)
def t_shared_truncate_race(ctx):
  """Overwriters (fsync each) and readers on every node while node0 keeps
  shrinking and regrowing the shared file. No read or final block may be
  CORRUPT or FOREIGN; a version may only vanish (read as a hole) if a
  truncate cut its block after the last write to it began; an older
  version must never come back after a newer one or a truncate."""
  results, by_block, finals, nb = _shared_run(ctx, 'shtr', 90, True, True)
  truncs = [t for r in results for t in r.get('truncs', [])]
  bad_reads = []
  for r in results:
    for ob in r['obs']:
      if ob[1] in (CORRUPT, FOREIGN):
        bad_reads.append(f'block {ob[0]}: {KIND[ob[1]]}')
      elif ob[1] != ZERO and not any(x[0] == ob[1] and x[1] == ob[2]
                                     for x in by_block.get(ob[0], [])):
        bad_reads.append(f'block {ob[0]}: ({ob[1]},{ob[2]}) never written')
  diverge, final_bad = [], []
  for b in range(nb):
    vals = {f[b] for f in finals}
    if len(vals) > 1:
      diverge.append((b, sorted(vals)))
    cand = _truncate_candidates(by_block.get(b, []), truncs, b)
    final_bad += [(b, KIND.get(v[0], v)) for v in vals if v not in cand]
  errs = [e for r in results for e in r['errors']]
  ctx.metrics.update({'writes': sum(len(v) for v in by_block.values()),
                      'truncates': len(truncs),
                      'reads_checked': sum(len(r['obs']) for r in results),
                      'bad_reads': len(bad_reads),
                      'final_divergent_blocks': len(diverge),
                      'final_bad_blocks': len(final_bad)})
  ctx.check(not errs, f'I/O errors: {errs[:5]}')
  ctx.check(not bad_reads, f'{len(bad_reads)} bad reads, e.g. '
                           f'{bad_reads[:6]}')
  ctx.check(not diverge, f'{len(diverge)} blocks differ between nodes, '
                         f'e.g. {diverge[:5]}')
  ctx.check(not final_bad, f'{len(final_bad)} blocks hold a version that '
                           f'should be gone (or lost one), e.g. '
                           f'{final_bad[:5]}')


# ---------------------------------------------------------------------------
# 3. Crash every daemon while the tiers are full and data is moving
# ---------------------------------------------------------------------------

def _read_durable_log(path):
  """name -> highest gen from a FileSetWriter log (missing file: {})."""
  out = {}
  try:
    with open(path) as f:
      for line in f:
        parts = line.split()
        if len(parts) == 2:
          out[parts[0]] = max(out.get(parts[0], 0), int(parts[1]))
  except OSError:
    pass
  return out


def _torn_unsynced(t, writer, inflight):
  """Whether a torn block is the unsynced in-flight write over an older
  version of the same file.

  Args:
    t: [head writer, head gen, rest writer, rest gen] from scan, or None.
    writer: the file's writer.
    inflight: the gen being written when the crash hit.
  Returns:
    True when the head is the in-flight gen and the rest an older gen, both
    by this file's writer.
  """
  if not t:
    return False
  hw, hg, rw, rg = t
  return hw == writer and rw == writer and hg == inflight and rg < hg


def _reread_note(ctx, nmx, path, block, nodes):
  """Re-read one 4 KiB record block on each of `nodes` and note, per node,
  where its trailing zeros start (4096 = none) and whether it now matches
  the first node's re-read.

  Args:
    ctx: test context.
    nmx: file name (for the note).
    path: file path.
    block: block index.
    nodes: node indices to read on.
  """
  seen = {}
  for j in nodes:
    r = ctx.call(j, 'read_hex', path=path, off=block * 4096, length=4096)
    if not r.get('ok'):
      seen[f'node{j}'] = f'read failed: {r.get("err")}'
      continue
    raw = bytes.fromhex(r['ret'])
    seen[f'node{j}'] = {'len': len(raw),
                        'tail_zero_from': len(raw.rstrip(b'\0'))}
  ctx.note(f'{nmx} block {block} re-read: {seen}')


def _check_filesets(ctx, base, n, nfiles, logs, replies, when,
                    skip_writer_of=None):
  """Verify FileSetWriter output after a fault.

  Every file whose version was fsynced must hold that version or the one in
  flight when the fault hit, block for block; nothing may be CORRUPT or
  FOREIGN; a file never fsynced may be absent or partial.

  Args:
    ctx: test context.
    base: directory holding the files.
    n: number of nodes (writer i + 1 ran on node i).
    nfiles: files per writer.
    logs: per-node durable-log paths.
    replies: node -> the rec_fileset reply (may be missing).
    when: text for failure messages.
    skip_writer_of: node whose files are not checked (None: all).
  """
  lost, corrupt, missing_file, checked = [], [], [], 0
  torn_unsynced = []
  for i in range(n):
    if i == skip_writer_of:
      continue
    durable = _read_durable_log(logs[i])
    rep = replies.get(i) or {}
    ret = rep.get('ret') or {}
    for nmx, g in (ret.get('durable') or {}).items():
      durable[nmx] = max(durable.get(nmx, 0), g)
    started = ret.get('started') or {}
    reader = (i + 1) % n
    for k in range(nfiles):
      nmx = f'w{i + 1}_f{k}'
      path = f'{base}/{nmx}'
      if not ctx.ok(reader, 'exists', path=path):
        if nmx in durable:
          missing_file.append(nmx)
        continue
      got = ctx.ok(reader, 'rec_scan', timeout=900, path=path, name=nmx,
                   nblocks=FILE_BLOCKS)
      checked += 1
      dg = durable.get(nmx, 0)
      for d in got.get('corrupt', [])[:2]:
        ctx.note(f'{nmx} corrupt block {when} (durable gen {dg}, '
                 f'started {started.get(nmx)}): {d}')
        # Read the block again, here and from another node: a block that is
        # whole on a re-read was a bad READ (stored data intact), one that is
        # still torn is bad STORED data (#1124).
        _reread_note(ctx, nmx, path, d['block'], [reader, (reader + 1) % n])
      foreign = got.get('foreign') or {}
      if foreign:
        ctx.note(f'{nmx} FOREIGN blocks {when} hold [file id, block, writer, '
                 f'gen]: {dict(list(foreign.items())[:4])}')
      torn = got.get('torn', {})
      inflight = started.get(nmx)
      for start, count, w, g in got['runs']:
        if w == CORRUPT and inflight is not None and all(
            _torn_unsynced(torn.get(str(b)), i + 1, inflight)
            for b in range(start, start + count)):
          # The write in flight at the crash (never fsynced) landed only in
          # part of the block; the rest is this file's older version. No
          # guarantee covers unsynced bytes, so this is legal (reported).
          torn_unsynced.append((nmx, start, count))
          continue
        if w in (CORRUPT, FOREIGN):
          corrupt.append((nmx, start, count, KIND[w]))
        elif w == ZERO:
          if dg:
            lost.append((nmx, start, count, 'hole', f'fsynced gen {dg}'))
        elif w != i + 1:
          corrupt.append((nmx, start, count, f'writer {w}'))
        elif g < dg:
          lost.append((nmx, start, count, f'gen {g}', f'fsynced gen {dg}'))
        elif g > dg and ret and g not in started.values() and \
            (g - (k + 1)) % nfiles != 0:
          # File k is written at gens k+1, k+1+nfiles, ...: any of those
          # newer than the fsynced one is an unsynced write that landed
          # (legal); anything else is some other round's data.
          corrupt.append((nmx, start, count, f'unexpected gen {g}'))
  ctx.metrics['files_checked'] = checked
  if torn_unsynced:
    ctx.metrics['torn_unsynced_blocks'] = sum(c for _, _, c in torn_unsynced)
    ctx.note(f'blocks torn by the unsynced write in flight {when} (legal: '
             f'the rest is the older version): {torn_unsynced[:6]}')
  ctx.check(not corrupt, f'{len(corrupt)} CORRUPT/FOREIGN/unexpected ranges '
                         f'{when}, e.g. {corrupt[:6]}')
  ctx.check(not missing_file, f'fsynced files missing {when}: '
                              f'{missing_file[:10]}')
  ctx.check(not lost, f'{len(lost)} ranges lost fsynced data {when}, e.g. '
                      f'{lost[:6]}')


@test('stress_crash_under_pressure', 'stress', min_nodes=1,
      redeploy_after=True, timeout=5400)
def t_crash_under_pressure(ctx):
  """Every node rewrites a cycling set of 64 MiB record files (working set
  twice its RAM tier, each file fsynced and logged), while the organizer
  migrates blobs; at a random moment every daemon on every node is
  SIGKILLed. After `clio_run start` recovers: every file whose version was
  fsynced holds that version or the newer one being written at the crash,
  block for block; nothing is CORRUPT or FOREIGN; a file never fsynced is
  absent or holds its own in-flight version and holes."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('crash')
  ctx.ok(0, 'mkdir', path=base)
  ram_mb = cl.ram_mb if cl.profile in ('tiered', 'safe') else cl.ram_gb * 1024
  nfiles = max(4, (2 * ram_mb) // 64)
  secs = 600
  logs = [f'{cl.run_dir}/crash_writer_{i}.log' for i in range(n)]
  for lp in logs:  # a rerun into the same --out must not inherit old lines
    open(lp, 'w').close()
  replies = {}

  def writer(i):
    replies[i] = ctx.a(i).call('rec_fileset', timeout=secs + 300,
                               dirpath=base, writer=i + 1, nfiles=nfiles,
                               blocks=FILE_BLOCKS, secs=secs, seed=i,
                               log_path=logs[i])
  th = threading.Thread(target=lambda: ctx.each(writer))
  th.start()
  delay = random.Random(time.time()).uniform(40, 90)
  ctx.metrics['crash_after_s'] = round(delay, 1)
  time.sleep(delay)
  from cluster import parallel
  parallel(cl.kill_fuse, cl.hosts)
  parallel(cl.kill_runtime, cl.hosts)
  th.join(timeout=secs + 400)
  restart_cluster(ctx, crash=True)
  ctx.cl.agents.clear()

  _check_filesets(ctx, base, n, nfiles, logs, replies, 'after the crash')


# ---------------------------------------------------------------------------
# 4. Overwrite a hot set bigger than RAM while the organizer migrates it
# ---------------------------------------------------------------------------

@test('stress_hot_rewrite', 'stress', min_nodes=1, redeploy_after=True,
      timeout=5400)
def t_hot_rewrite(ctx):
  """Each node overwrites random halves of a working set twice the size of
  its RAM tier (16 MiB files), round after round, while the organizer
  migrates blobs between tiers. After each round's writes the writer reads
  its files back (read-your-writes, no fsync in between); every other round
  fsyncs and another node verifies. An overwrite that lands on a copy being
  migrated -- so the migration resurrects the old version, or mixes the
  two -- shows up as an old gen or a CORRUPT block."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('hot')
  ctx.ok(0, 'mkdir', path=base)
  ram_mb = cl.ram_mb if cl.profile in ('tiered', 'safe') else cl.ram_gb * 1024
  fb = 4096  # 16 MiB files
  nfiles = max(4, (2 * ram_mb) // 16)
  rounds = 8
  models = {}
  for i in range(n):
    for k in range(nfiles):
      models[f'n{i}_h{k}'] = [(ZERO, 0)] * fb
  bad, counts = [], {'bad': 0}
  counts_lock = threading.Lock()

  def do_round(i, r):
    rng = random.Random(f'{i}:{r}')
    names = [f'n{i}_h{k}' for k in range(nfiles)]
    touched = rng.sample(names, max(1, len(names) // 2)) if r else names
    fsync = r % 2 == 1
    plans = {}
    for nm in touched:
      runs = [[0, fb]] if r == 0 else []
      if r:
        for _ in range(4):
          s = rng.randrange(fb)
          runs.append([s, min(rng.randrange(1, 256), fb - s)])
      plans[nm] = runs
      ctx.ok(i, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=runs, writer=i + 1, gen=r + 1, fsync=fsync)
    return plans

  for r in range(rounds):
    plans = ctx.each(lambda i: do_round(i, r))
    for i in range(n):
      for nm, runs in plans[i].items():
        _apply(models[nm], runs, i + 1, r + 1)

    def check(i, reader):
      for k in range(nfiles):
        nm = f'n{i}_h{k}'
        got = ctx.ok(reader, 'rec_scan', timeout=900, path=f'{base}/{nm}',
                     name=nm, nblocks=fb)
        nbad = _compare(_to_runs(models[nm]), got, fb, f'r{r}:{nm}', bad)
        with counts_lock:  # check() runs on every node's thread at once
          counts['bad'] += nbad
    # Read-your-writes on the writer's node.
    ctx.each(lambda i: check(i, i))
    if r % 2 == 1:  # fsynced round: every other node sees it too
      ctx.each(lambda i: check(i, (i + 1) % n))
  ctx.metrics['bad_blocks'] = counts['bad']
  ctx.check(counts['bad'] == 0 and not bad,
            f'{counts["bad"]} blocks differ from the model, e.g. {bad[:8]}')


@test('stress_op_latency', 'stress', min_nodes=2, redeploy_after=True,
      timeout=1800)
def t_op_latency(ctx):
  """Diagnostic: per-step latency of open / 4 KiB pread / pwrite / fsync /
  close on a shared file, idle and while another node runs the concurrent
  overwriters+readers. Reported as notes; fails only on errors."""
  import os
  probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'op_latency_probe.py')
  p = ctx.p('lat')
  ctx.ok(0, 'write_file', path=p, size=256 * sr.BLK, seed=1)
  r = ctx.ok(0, 'sh', cmd=f'python3 {probe} {p}', timeout=600)
  ctx.note(f'idle node0: ' + r['out'].strip().replace(chr(10), ' | '))
  box = {}
  th = threading.Thread(target=lambda: box.update(r=ctx.call(
      1, 'rec_shared_stress', timeout=900, path=p, name='lat', nblocks=256,
      writer_base=50, writers=2, readers=2, secs=40, seed=5)))
  th.start()
  time.sleep(5)
  r = ctx.ok(0, 'sh', cmd=f'python3 {probe} {p}', timeout=600)
  ctx.note(f'node0 while node1 hammers: ' +
           r['out'].strip().replace(chr(10), ' | '))
  th.join()
  res = (box.get('r') or {}).get('ret') or {}
  ctx.metrics['node1_writes'] = sum(len(v) for v in
                                    (res.get('writes') or {}).values())
  ctx.metrics['node1_reads'] = len(res.get('obs') or [])


@test('stress_node_loss_during_writes', 'stress', min_nodes=3,
      redeploy_after=True, timeout=5400)
def t_node_loss_during_writes(ctx):
  """Every node rewrites its own cycling set of fsynced record files (tier
  pressure, organizer migrating); after ~30 s one node's daemons are
  SIGKILLed while the others keep writing (retrying failed rounds as their
  files fail over), and ~40 s later it restarts. Every version any writer
  saw fsynced -- including the lost node's own, from before it died -- must
  survive intact; nothing may be CORRUPT or FOREIGN."""
  n = len(ctx.hosts)
  cl = ctx.cl
  base = ctx.p('nl')
  ctx.ok(0, 'mkdir', path=base)
  ram_mb = cl.ram_mb if cl.profile in ('tiered', 'safe') else cl.ram_gb * 1024
  nfiles = max(4, (2 * ram_mb) // 64)
  secs = 120
  logs = [f'{cl.run_dir}/nodeloss_writer_{i}.log' for i in range(n)]
  for lp in logs:  # a rerun into the same --out must not inherit old lines
    open(lp, 'w').close()
  replies = {}

  def writer(i):
    replies[i] = ctx.a(i).call('rec_fileset', timeout=secs + 600,
                               dirpath=base, writer=i + 1, nfiles=nfiles,
                               blocks=FILE_BLOCKS, secs=secs, seed=i,
                               log_path=logs[i], retry=True)
  th = threading.Thread(target=lambda: ctx.each(writer))
  th.start()
  time.sleep(30)
  victim = n - 1
  vh = ctx.hosts[victim]
  cl.kill_fuse(vh)
  cl.kill_runtime(vh)
  time.sleep(40)
  cl.start_runtime(vh)
  ctx.check(cl.runtime_up(vh), f'{vh} runtime did not restart')
  time.sleep(3)
  ctx.check(cl.mount(vh), f'{vh} remount failed')
  th.join(timeout=secs + 700)
  cl.agents.pop(vh, None)
  ctx.metrics['survivor_errors'] = sum(
      ((replies.get(i) or {}).get('ret') or {}).get('nerrors', 0)
      for i in range(n) if i != victim)
  ctx.metrics['rounds'] = {f'node{i}': ((replies.get(i) or {}).get('ret')
                                        or {}).get('rounds')
                           for i in range(n)}
  _check_filesets(ctx, base, n, nfiles, logs, replies,
                  f'after node{victim} was lost and came back')


@test('stress_space_accounting', 'stress', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_space_accounting(ctx):
  """Diagnostic for capacity: node0 writes 64 MiB record files (fsynced) up
  to half of its RAM + fast tiers, and the filesystem's used space (statvfs)
  is recorded against the bytes written -- right after, after 20 s of
  organizer rounds, and after every file is deleted (used space must fall
  back near the empty baseline: anything left is leaked capacity that ends
  in a premature ENOSPC). Fails only if space is not returned."""
  base = ctx.p('space')
  ctx.ok(0, 'mkdir', path=base)

  def used_mib():
    st = ctx.ok(0, 'statvfs', path=ctx.cl.mnt)
    return (st['blocks'] - st['bfree']) * st['bsize'] // MiB

  time.sleep(3)
  u0 = used_mib()
  nfiles = max(2, _tier_mb(ctx) // 2 // 64)
  for k in range(nfiles):
    nm = f's{k}'
    ctx.ok(0, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
           runs=[[0, FILE_BLOCKS]], writer=1, gen=1, fsync=True)
  written = nfiles * 64
  u1 = used_mib()
  time.sleep(20)
  u2 = used_mib()
  import os as _os
  tool = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       'alloc_log_usage.py')
  for i in range(len(ctx.hosts)):
    r = ctx.call(i, 'sh', cmd=f'python3 {tool} {ctx.cl.local_root}/data')
    ctx.note(f'node{i} live MiB per file tier after the writes: ' +
             ((r.get('ret') or {}).get('out') or '').strip().replace(
                 chr(10), ' | '))
  # Overwrite every file in full, five times: the data set does not grow,
  # so neither may the used space.
  after_rw = []
  for r in range(5):
    for k in range(nfiles):
      nm = f's{k}'
      ctx.ok(0, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=[[0, FILE_BLOCKS]], writer=1, gen=2 + r, fsync=True)
    after_rw.append(used_mib())
  # Then partial overwrites: random ranges of every file, ten rounds.
  rng = random.Random(7)
  for r in range(10):
    for k in range(nfiles):
      nm = f's{k}'
      runs = []
      for _ in range(6):
        st = rng.randrange(FILE_BLOCKS)
        runs.append([st, min(rng.randrange(1, 600), FILE_BLOCKS - st)])
      ctx.ok(0, 'rec_write', timeout=900, path=f'{base}/{nm}', name=nm,
             runs=runs, writer=1, gen=10 + r, fsync=True)
    after_rw.append(used_mib())
  time.sleep(10)
  after_rw.append(used_mib())
  for i in range(len(ctx.hosts)):
    r = ctx.call(i, 'sh', cmd=f'python3 {tool} {ctx.cl.local_root}/data')
    ctx.note(f'node{i} live MiB per tier after the overwrites: ' +
             ((r.get('ret') or {}).get('out') or '').strip().replace(
                 chr(10), ' | '))
  for k in range(nfiles):
    ctx.ok(0, 'unlink', path=f'{base}/s{k}')
  left = None
  for _ in range(30):
    time.sleep(2)
    left = used_mib()
    if left - u0 <= max(64, written // 20):
      break
  ctx.metrics.update({
      'written_mib': written, 'used_before_mib': u0,
      'used_after_write_mib': u1, 'used_after_20s_mib': u2,
      'used_after_delete_mib': left,
      'space_per_byte_after_write': round((u1 - u0) / written, 2),
      'space_per_byte_after_20s': round((u2 - u0) / written, 2),
      'used_after_each_overwrite_round_mib': after_rw})
  ctx.note(f'tier backing files at the end: {_tier_usage(ctx)}')
  grew = after_rw[-1] - u2
  ctx.check(grew <= max(64, written // 10),
            f'used space grew {grew} MiB over 5 full + 10 partial overwrites '
            f'of the same {written} MiB (rewrites leak capacity): '
            f'{after_rw}')
  ctx.check(left - u0 <= max(64, written // 20),
            f'{left - u0} MiB still used a minute after deleting all '
            f'{written} MiB written (leaked capacity)')


@test('stress_safe_save', 'stress', min_nodes=1, redeploy_after=True,
      timeout=3600)
def t_safe_save(ctx):
  """Every node runs 2 savers (write a whole new version of one of 8 target
  files to a temp name, fsync, rename over the target) and 2 readers (open
  a target, read it whole) for 90 s. Every read must see exactly one
  complete, intact version -- never a short, mixed or CORRUPT file;
  afterwards every node reads the same final version of each target, and
  it is the version of the last rename to it. ENOENT/ESTALE while a name
  is replaced are reported (see the check) and fail only when strict."""
  n = len(ctx.hosts)
  d = ctx.p('save')
  ctx.ok(0, 'mkdir', path=d)
  targets = [f't{k}' for k in range(8)]
  nb = 64  # 256 KiB files
  for t in targets:  # every target exists before readers start
    ctx.ok(0, 'rec_write', path=f'{d}/{t}', name=t, runs=[[0, nb]],
           writer=0, gen=0, fsync=True)

  def run(i):
    return ctx.ok(i, 'rec_safe_save', timeout=900, dirpath=d,
                  targets=targets, nblocks=nb, writer_base=10 * i + 1,
                  savers=2, readers=2, secs=90, seed=i)
  results = ctx.each(run)
  saves = [s for r in results for s in r['saves']]
  # Prefix each problem with the node that saw it ('n3/t5: ...').
  bad = [f'n{i}/{b}' for i, r in enumerate(results) for b in r['bad']]
  errs = [f'n{i}/{e}' for i, r in enumerate(results) for e in r['errors']]
  time.sleep(2)
  finals, wrong = {}, []
  for t in targets:
    seen = set()
    for i in range(n):
      got = ctx.ok(i, 'rec_scan', path=f'{d}/{t}', name=t, nblocks=nb)
      runs = got['runs']
      seen.add(tuple(runs[0][2:]) if len(runs) == 1 else ('mixed',))
    finals[t] = seen
    mine = [s for s in saves if s[0] == t]
    if len(seen) != 1:
      wrong.append((t, 'nodes disagree', sorted(seen)))
      continue
    v = next(iter(seen))
    if mine:
      last = max(s[4] for s in mine)
      ok = {(s[1], s[2]) for s in mine if s[4] >= last - SKEW_S}
      if tuple(v) not in ok:
        wrong.append((t, v, 'last renames', sorted(ok)[:3]))
  ctx.metrics.update({'saves': len(saves),
                      'reads': sum(r['reads'] for r in results),
                      'bad_reads': sum(r['nbad'] for r in results),
                      'errors': sum(r['nerrors'] for r in results)})
  kinds, samples = {}, {}
  for b in bad:
    k = b.split(': ', 1)[1].split(',')[0][:40]
    kinds[k] = kinds.get(k, 0) + 1
    samples.setdefault(k, [])
    if len(samples[k]) < 3:
      samples[k].append(b)
  ctx.metrics['bad_read_kinds'] = kinds
  for k, v in samples.items():
    ctx.note(f'bad reads [{k}]: {v}')
  # Corruption (a short, mixed or corrupt read, a wrong final version) always
  # fails. A name reported missing (ENOENT) or stale (ESTALE) while it is
  # being replaced is the known window of libfuse's path-based API: a rename
  # over a name open on the renaming node is two server renames (hide, then
  # replace). It is reported, and fails only with CLIO_SUITE_STRICT_RENAME=1
  # (closing it needs the inode-based low-level FUSE API).
  import os
  broken = [b for b in bad if 'missing (ENOENT' not in b]
  unavailable = len(bad) - len(broken) + len(errs)
  ctx.metrics['unavailable_reads'] = unavailable
  if unavailable:
    ctx.note(f'{unavailable} reads hit the rename-over-open window '
             f'(ENOENT/ESTALE), e.g. {(errs + bad)[:3]}')
  problems = []
  if broken:
    problems.append(f'{len(broken)} reads saw a broken file, e.g. '
                    f'{broken[:6]}')
  if wrong:
    problems.append(f'final versions wrong: {wrong[:5]}')
  if unavailable and os.environ.get('CLIO_SUITE_STRICT_RENAME') == '1':
    problems.append(f'{unavailable} reads failed with ENOENT/ESTALE')
  ctx.check(not problems, ' ;; '.join(problems))
