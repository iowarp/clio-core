#!/usr/bin/env python3
"""clio-fs distributed POSIX / scale / fault-tolerance test suite.

Run inside a SLURM allocation (the node list comes from SLURM_JOB_NODELIST)
or with --hosts.  Deploys clio_run + the CTE filesystem chain + a
clio_cte_fuse mount on every node (cluster.py), starts a per-node POSIX agent
(agent.py) and runs the registered tests against the FUSE mounts.

  python3 suite.py --bin build_fs/bin --out ~/clio_fs_suite/run1 \
      --groups posix,dist,fault [--nodes 8] [--only name1,name2]

Every test result is one of:
  PASS   behaved exactly as POSIX (or the documented clio contract) requires
  FAIL   wrong answer: data corruption, wrong errno, stale/missing namespace
  HANG   an operation did not return inside its deadline (the FUSE
         connection is aborted and the node redeployed afterwards)
  ERROR  the harness itself could not run the test (deploy failure etc.)
  UNSUP  the feature is intentionally not implemented (e.g. flock), noted
         so the report states the limitation instead of hiding it
Results stream to <out>/results.jsonl; a summary lands in <out>/summary.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cluster import Cluster, parallel  # noqa: E402

TESTS = []


def test(name, group, min_nodes=1, max_nodes=None, timeout=900,
         redeploy_after=False):
  """Register a test function f(ctx)."""
  def deco(fn):
    TESTS.append({'name': name, 'group': group, 'fn': fn,
                  'min_nodes': min_nodes, 'max_nodes': max_nodes,
                  'timeout': timeout, 'redeploy_after': redeploy_after,
                  'doc': (fn.__doc__ or '').strip().split('\n')[0]})
    return fn
  return deco


class TestFailure(Exception):
  """A POSIX / consistency expectation was violated."""


class TestHang(Exception):
  """An operation exceeded its deadline."""


class TestUnsupported(Exception):
  """The feature is a known, documented limitation."""


class Ctx:
  """Per-test handle: node agents, paths and assertion helpers."""

  def __init__(self, cluster, name, hosts):
    self.cl = cluster
    self.name = name
    self.hosts = hosts
    self.notes = []
    self.metrics = {}
    self.dir = f'{cluster.mnt}/t_{name}'
    self.hang_hosts = set()

  def a(self, i):
    """Agent for the i-th node of this test."""
    return self.cl.agent(self.hosts[i % len(self.hosts)])

  def p(self, *parts):
    return '/'.join([self.dir] + [str(x) for x in parts])

  def call(self, i, op, timeout=120, **args):
    """Run op on node i; raise on hang, return reply."""
    r = self.a(i).call(op, timeout=timeout, **args)
    if r.get('hang') or r.get('agent_dead'):
      self.hang_hosts.add(self.hosts[i % len(self.hosts)])
      raise TestHang(f'node{i} {self.hosts[i % len(self.hosts)]} '
                     f'{op}({_short(args)}) -> {r.get("err")}')
    return r

  def ok(self, i, op, timeout=120, **args):
    """Run op on node i and require success; return its value."""
    r = self.call(i, op, timeout=timeout, **args)
    if not r.get('ok'):
      raise TestFailure(f'node{i} {op}({_short(args)}) failed: {r.get("err")}')
    return r['ret']

  def err(self, i, op, expect, timeout=120, **args):
    """Require op on node i to fail with errno name(s) in expect."""
    import errno as E
    r = self.call(i, op, timeout=timeout, **args)
    names = expect if isinstance(expect, (list, tuple)) else [expect]
    codes = [getattr(E, n) for n in names]
    if r.get('ok'):
      raise TestFailure(f'node{i} {op}({_short(args)}) succeeded; '
                        f'expected {names}')
    if r.get('errno') not in codes:
      raise TestFailure(f'node{i} {op}({_short(args)}) -> {r.get("err")}; '
                        f'expected {names}')
    return r

  def check(self, cond, msg):
    if not cond:
      raise TestFailure(msg)

  def note(self, msg):
    self.notes.append(msg)

  def all_nodes(self, op, timeout=120, **args):
    """Run op on every node concurrently; return replies."""
    rs = parallel(lambda i: self.a(i).call(op, timeout=timeout, **args),
                  list(range(len(self.hosts))))
    for i, r in enumerate(rs):
      if isinstance(r, Exception):
        raise TestFailure(f'node{i} {op}: {r}')
      if r.get('hang') or r.get('agent_dead'):
        self.hang_hosts.add(self.hosts[i])
        raise TestHang(f'node{i} {op}({_short(args)}) -> {r.get("err")}')
    return rs

  def each(self, fn):
    """Run fn(i) for every node concurrently; re-raise the first error."""
    rs = parallel(fn, list(range(len(self.hosts))))
    for r in rs:
      if isinstance(r, Exception):
        raise r
    return rs

  def eventually(self, fn, timeout=10.0, period=0.25):
    """Poll fn() (returns (done, info)) until true or timeout; return info.

    Used for cross-node visibility where the adapter's attribute-cache TTL
    allows bounded staleness.  The elapsed time is the staleness observed.
    """
    t0 = time.time()
    done, info = fn()
    while not done and time.time() - t0 < timeout:
      time.sleep(period)
      done, info = fn()
    return done, info, time.time() - t0


def _short(args):
  s = json.dumps(args)
  return s if len(s) < 200 else s[:200] + '...'


def load_tests():
  import tests_posix  # noqa: F401
  import tests_dist  # noqa: F401
  import tests_fault  # noqa: F401
  import tests_apps  # noqa: F401


def slurm_hosts():
  nl = os.environ.get('SLURM_JOB_NODELIST')
  if not nl:
    return [os.uname().nodename.split('.')[0]]
  out = subprocess.run(['scontrol', 'show', 'hostnames', nl],
                       capture_output=True, text=True, check=True).stdout
  return out.split()


def redeploy(cl, log):
  log('redeploying cluster')
  cl.close_agents()
  parallel(cl.force_unmount, cl.hosts)
  cl.down(graceful=False)
  ok, msg = cl.up(wipe=True)
  log(f'redeploy: {msg}')
  return ok


def run_one(cl, t, hosts, log):
  """Execute one test; return a result record."""
  ctx = Ctx(cl, t['name'], hosts)
  rec = {'name': t['name'], 'group': t['group'], 'nodes': len(hosts),
         'doc': t['doc']}
  t0 = time.time()
  try:
    ctx.ok(0, 'makedirs', path=ctx.dir, timeout=60)
    t['fn'](ctx)
    rec['status'] = 'PASS'
  except TestFailure as e:
    rec['status'] = 'FAIL'
    rec['error'] = str(e)
  except TestHang as e:
    rec['status'] = 'HANG'
    rec['error'] = str(e)
  except TestUnsupported as e:
    rec['status'] = 'UNSUP'
    rec['error'] = str(e)
  except Exception as e:  # pylint: disable=broad-except
    rec['status'] = 'ERROR'
    rec['error'] = f'{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}'
  rec['secs'] = round(time.time() - t0, 2)
  rec['notes'] = ctx.notes
  rec['metrics'] = ctx.metrics
  if ctx.hang_hosts:
    rec['hang_hosts'] = sorted(ctx.hang_hosts)
  return rec


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--bin', required=True)
  ap.add_argument('--out', required=True)
  ap.add_argument('--hosts', default='')
  ap.add_argument('--nodes', type=int, default=0,
                  help='use only the first N hosts')
  ap.add_argument('--groups', default='posix,dist,apps,fault')
  ap.add_argument('--only', default='')
  ap.add_argument('--skip', default='')
  ap.add_argument('--profile', default='persistent',
                  choices=['persistent', 'ram'])
  ap.add_argument('--attr-cache', default=None,
                  help='CLIO_FUSE_ATTR_CACHE_S for the mounts')
  ap.add_argument('--port', type=int, default=9519)
  ap.add_argument('--list', action='store_true')
  args = ap.parse_args()

  load_tests()
  if args.list:
    for t in TESTS:
      print(f'{t["group"]:6} {t["name"]:40} n>={t["min_nodes"]}  {t["doc"]}')
    return 0

  hosts = args.hosts.split(',') if args.hosts else slurm_hosts()
  if args.nodes:
    hosts = hosts[:args.nodes]
  os.makedirs(args.out, exist_ok=True)
  logf = open(os.path.join(args.out, 'suite.log'), 'a')

  def log(msg):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    logf.write(line + '\n')
    logf.flush()

  groups = set(args.groups.split(','))
  only = set(filter(None, args.only.split(',')))
  skip = set(filter(None, args.skip.split(',')))
  sel = [t for t in TESTS if t['group'] in groups and
         (not only or t['name'] in only) and t['name'] not in skip and
         len(hosts) >= t['min_nodes']]
  # Fault tests last: they deliberately break the deployment.
  order = {'posix': 0, 'dist': 1, 'apps': 2, 'perf': 3, 'fault': 4}
  sel.sort(key=lambda t: order.get(t['group'], 9))

  cl = Cluster(hosts, os.path.abspath(args.bin), os.path.abspath(args.out),
               profile=args.profile, port=args.port,
               attr_cache_s=args.attr_cache)
  log(f'hosts={hosts} profile={args.profile} tests={len(sel)}')
  ok, msg = cl.up(wipe=True)
  log(f'deploy: {msg}')
  resf = open(os.path.join(args.out, 'results.jsonl'), 'a')
  results = []
  if not ok:
    rec = {'name': 'deploy', 'group': 'setup', 'status': 'ERROR',
           'error': msg, 'nodes': len(hosts)}
    resf.write(json.dumps(rec) + '\n')
    results.append(rec)
  else:
    for t in sel:
      nh = hosts if t['max_nodes'] is None else hosts[:t['max_nodes']]
      log(f'RUN  {t["group"]}/{t["name"]} on {len(nh)} node(s)')
      rec = run_one(cl, t, nh, log)
      log(f'{rec["status"]:5} {t["name"]} ({rec["secs"]}s) '
          f'{rec.get("error", "")[:400]}')
      for n in rec['notes']:
        log(f'      note: {n}')
      resf.write(json.dumps(rec) + '\n')
      resf.flush()
      results.append(rec)
      bad = cl.health() if rec['status'] != 'PASS' or \
          t['redeploy_after'] else {}
      if rec['status'] == 'HANG' or bad or t['redeploy_after']:
        if bad:
          log(f'health: {bad}')
        if not redeploy(cl, log):
          log('redeploy failed; aborting remaining tests')
          break
  cl.down(graceful=True)
  write_summary(args.out, results, hosts, args)
  counts = {}
  for r in results:
    counts[r['status']] = counts.get(r['status'], 0) + 1
  log(f'DONE {counts}')
  return 0 if all(r['status'] in ('PASS', 'UNSUP') for r in results) else 1


def write_summary(out, results, hosts, args):
  counts = {}
  for r in results:
    counts[r['status']] = counts.get(r['status'], 0) + 1
  with open(os.path.join(out, 'summary.md'), 'w') as f:
    f.write(f'# clio-fs suite: {len(hosts)} node(s), profile '
            f'{args.profile}, attr_cache={args.attr_cache}\n\n')
    f.write(' '.join(f'**{k}** {v}' for k, v in sorted(counts.items())))
    f.write('\n\n| status | group | test | nodes | secs | detail |\n'
            '|---|---|---|---|---|---|\n')
    for r in results:
      det = (r.get('error') or '; '.join(r.get('notes', [])))
      det = det.replace('|', '/').replace('\n', ' ')[:300]
      f.write(f'| {r["status"]} | {r["group"]} | {r["name"]} | '
              f'{r["nodes"]} | {r.get("secs", "")} | {det} |\n')
    f.write('\n## Metrics\n\n')
    for r in results:
      if r.get('metrics'):
        f.write(f'- {r["name"]}: {json.dumps(r["metrics"])}\n')


if __name__ == '__main__':
  # Re-import as `suite` so the test modules (which `from suite import test`)
  # register into, and raise exceptions from, the same module object.
  import suite as _suite
  sys.exit(_suite.main())
