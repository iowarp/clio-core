"""Real mdtest (from IOR) over the clio-fs FUSE mount, in IO500's two modes.

  mdtest_easy: every rank creates, stats and removes empty files in its OWN
               directory (IO500 mdtest-easy: -F -P -u -L).
  mdtest_hard: all ranks share ONE directory; each file gets 3901 bytes
               written, then is stat'ed, read back and removed (IO500
               mdtest-hard: -t -F -P -w 3901 -e 3901 -N 1).

Group `mdtest` (not in the default groups). Needs an MPI build of mdtest:
  CLIO_SUITE_MDTEST   path to mdtest   (default: on PATH)
  CLIO_SUITE_MPIRUN   path to mpirun   (default: on PATH)
  CLIO_SUITE_MPIRUN_ARGS  extra mpirun arguments (e.g. the network
                      interfaces: --mca btl_tcp_if_include <if> ...)
  CLIO_SUITE_MD_PPN   ranks per node   (default 8)
  CLIO_SUITE_MD_EASY_N / CLIO_SUITE_MD_HARD_N   files per rank
Rates (ops/s, mdtest's mean over iterations) go to the metrics column.
"""

import os
import re
import shlex
import shutil
import subprocess

from suite import test

_PHASES = {
    'File creation': 'create',
    'File stat': 'stat',
    'File read': 'read',
    'File removal': 'remove',
}


def _tool(env, name):
  """Resolve a tool from an env override or PATH; None when absent."""
  path = os.environ.get(env) or shutil.which(name)
  return path if path and os.access(path, os.X_OK) else None


def _parse_rates(text):
  """Map mdtest's SUMMARY rate table to {phase: mean ops/s}."""
  rates = {}
  in_rate = False
  for line in text.splitlines():
    if line.startswith('SUMMARY rate'):
      in_rate = True
      continue
    if line.startswith('SUMMARY time'):
      break
    if not in_rate:
      continue
    m = re.match(r'\s*(File [a-z]+)\s*:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)', line)
    if m and m.group(1) in _PHASES:
      rates[_PHASES[m.group(1)]] = float(m.group(4))  # Mean column
  return rates


def _run_mdtest(ctx, mode, args, n_default):
  """Launch mdtest across every node of the test; record its rates."""
  mdtest = _tool('CLIO_SUITE_MDTEST', 'mdtest')
  mpirun = _tool('CLIO_SUITE_MPIRUN', 'mpirun')
  ctx.check(mdtest is not None and mpirun is not None,
            'mdtest/mpirun not found (set CLIO_SUITE_MDTEST, '
            'CLIO_SUITE_MPIRUN)')
  ppn = int(os.environ.get('CLIO_SUITE_MD_PPN', '8'))
  n = int(os.environ.get(f'CLIO_SUITE_MD_{mode.upper()}_N', str(n_default)))
  ctx.ok(0, 'makedirs', path=ctx.p(mode))
  hosts = ','.join(f'{h}:{ppn}' for h in ctx.hosts)
  extra = shlex.split(os.environ.get('CLIO_SUITE_MPIRUN_ARGS', ''))
  cmd = [mpirun] + extra + ['--host', hosts, '-np', str(ppn * len(ctx.hosts)),
         '--bind-to', 'none', mdtest, '-n', str(n), '-d', ctx.p(mode)] + args
  ctx.note(' '.join(cmd))
  r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
  out = r.stdout + r.stderr
  with open(os.path.join(ctx.cl.run_dir, f'mdtest_{mode}.log'), 'w') as f:
    f.write(' '.join(cmd) + '\n' + out)
  ctx.check(r.returncode == 0,
            f'mdtest {mode} rc={r.returncode}: {out[-600:]}')
  rates = _parse_rates(out)
  ctx.check(bool(rates), f'mdtest {mode}: no rates in output: {out[-600:]}')
  for phase, v in rates.items():
    ctx.metrics[f'{mode}_{phase}_ops_s'] = round(v, 1)
  ctx.metrics[f'{mode}_files'] = n * ppn * len(ctx.hosts)


@test('mdtest_easy', 'mdtest', timeout=3600)
def t_mdtest_easy(ctx):
  """IO500 mdtest-easy: empty files, one directory per rank."""
  _run_mdtest(ctx, 'easy', ['-F', '-P', '-u', '-L', '-N', '1'], 2000)


@test('mdtest_hard', 'mdtest', timeout=3600)
def t_mdtest_hard(ctx):
  """IO500 mdtest-hard: 3901-byte files written and read, one shared dir."""
  _run_mdtest(ctx, 'hard',
              ['-t', '-F', '-P', '-w', '3901', '-e', '3901', '-N', '1'], 500)
