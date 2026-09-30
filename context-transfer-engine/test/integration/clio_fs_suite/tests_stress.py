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
  if cl.profile == 'tiered':
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

  def verify(reader_of, names_of):
    def one(i):
      r = reader_of(i)
      for nm in names_of(i):
        got = ctx.ok(r, 'rec_scan', timeout=900, path=f'{base}/{nm}',
                     name=nm, nblocks=FILE_BLOCKS)
        anomalies['corrupt'] += _corrupt_blocks(got['runs'])
        anomalies['mismatch'] += _compare(_to_runs(models[nm]), got,
                                          FILE_BLOCKS, nm, bad)
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
  ctx.metrics['corrupt_or_foreign_blocks'] = anomalies['corrupt']
  ctx.metrics['bad_blocks_total'] = anomalies['mismatch']
  ctx.check(anomalies['mismatch'] == 0,
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
  ram_mb = cl.ram_mb if cl.profile == 'tiered' else cl.ram_gb * 1024
  nfiles = max(4, (2 * ram_mb) // 64)
  secs = 600
  logs = [f'{cl.run_dir}/crash_writer_{i}.log' for i in range(n)]
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

  lost, corrupt, missing_file, checked = [], [], [], 0
  for i in range(n):
    durable = _read_durable_log(logs[i])
    rep = replies.get(i) or {}
    ret = rep.get('ret') or {}
    for nmx, g in (ret.get('durable') or {}).items():
      durable[nmx] = max(durable.get(nmx, 0), g)
    started = ret.get('started') or {}
    names = [f'w{i + 1}_f{k}' for k in range(nfiles)]
    reader = (i + 1) % n
    for nmx in names:
      path = f'{base}/{nmx}'
      if not ctx.ok(reader, 'exists', path=path):
        if nmx in durable:
          missing_file.append(nmx)
        continue
      got = ctx.ok(reader, 'rec_scan', timeout=900, path=path, name=nmx,
                   nblocks=FILE_BLOCKS)
      checked += 1
      dg = durable.get(nmx, 0)
      ok_gens = {dg} if dg else set()
      if nmx in started:
        ok_gens.add(started[nmx])
      for start, count, w, g in got['runs']:
        if w in (CORRUPT, FOREIGN):
          corrupt.append((nmx, start, count, KIND[w]))
        elif w == ZERO:
          if dg:
            lost.append((nmx, start, count, 'hole', f'fsynced gen {dg}'))
        elif w != i + 1:
          corrupt.append((nmx, start, count, f'writer {w}'))
        elif g < dg:
          lost.append((nmx, start, count, f'gen {g}', f'fsynced gen {dg}'))
        elif g not in ok_gens and g > dg:
          # A newer version whose write was never acknowledged is fine only
          # if it is the one in flight; with the reply lost (the op was cut
          # short by the crash) any gen > dg of this name is in-flight.
          if ret:
            corrupt.append((nmx, start, count, f'unexpected gen {g}'))
  ctx.metrics.update({'files_checked': checked,
                      'fsynced_files': sum(1 for i in range(n)
                                           for _ in _read_durable_log(logs[i]))})
  ctx.check(not corrupt, f'{len(corrupt)} CORRUPT/FOREIGN/unexpected ranges '
                         f'after the crash, e.g. {corrupt[:6]}')
  ctx.check(not missing_file, f'fsynced files missing after the crash: '
                              f'{missing_file[:10]}')
  ctx.check(not lost, f'{len(lost)} ranges lost fsynced data, e.g. '
                      f'{lost[:6]}')


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
  ram_mb = cl.ram_mb if cl.profile == 'tiered' else cl.ram_gb * 1024
  fb = 4096  # 16 MiB files
  nfiles = max(4, (2 * ram_mb) // 16)
  rounds = 8
  models = {}
  for i in range(n):
    for k in range(nfiles):
      models[f'n{i}_h{k}'] = [(ZERO, 0)] * fb
  bad, counts = [], {'bad': 0}

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
        counts['bad'] += _compare(_to_runs(models[nm]), got, fb,
                                  f'r{r}:{nm}', bad)
    # Read-your-writes on the writer's node.
    ctx.each(lambda i: check(i, i))
    if r % 2 == 1:  # fsynced round: every other node sees it too
      ctx.each(lambda i: check(i, (i + 1) % n))
  ctx.metrics['bad_blocks'] = counts['bad']
  ctx.check(counts['bad'] == 0, f'{counts["bad"]} blocks differ from the '
                                f'model, e.g. {bad[:8]}')


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
