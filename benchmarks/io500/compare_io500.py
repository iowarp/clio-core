#!/usr/bin/env python3
"""IO500 table from pbs_io500_aurora.sh job outputs, with repeated runs.

  compare_io500.py DIR_OR_FILE... [--base daos] [--out io500.md]
                   [--spread 0.10]

Each argument is a job output file (the PBS -o file of
pbs_io500_aurora.sh) or a directory of them (*.out). Runs are grouped by
file name without a trailing "-<number>": cliofs-1.out, cliofs-2.out and
cliofs-3.out are three runs of the system "cliofs". For every phase and
score a system's column is the mean over its runs with the lowest and
highest run; the ratio column is each system's mean over the mean of the
system named by --base (a group name, else the first group of that mode;
default: daos). A phase whose runs differ by more than --spread
(coefficient of variation) in any system is marked "~": its ratio is not
settled by these runs.

The per-run section lists what could make a run unfit to compare: exit
status, a missing score, different caps, and error lines in the benchmark
outputs (failed fsync, ENOSPC, I/O errors).

Exit 0 when every run finished with a score, the same caps and no error
lines; 1 otherwise.
"""

import argparse
import glob
import math
import os
import re
import sys

RESULT = re.compile(r'\[RESULT\]\s+(\S+)\s+([\d.]+)\s+(\S+)\s*:\s*time\s+([\d.]+)')
SCORE = re.compile(r'\[SCORE\s*\]\s+Bandwidth\s+([\d.]+)\s+GiB/s\s*:\s*IOPS\s+'
                   r'([\d.]+)\s+kiops\s*:\s*TOTAL\s+([\d.]+)')
HEAD = re.compile(r'=== IO500 mode=(\S+) nodes=(\d+) ppn=(\d+).* out=(\S+)')
CAPS = re.compile(r'=== caps: (.*)')
CELL = re.compile(r'IO500CELL .* rc=(\d+)')
ERRLINE = re.compile(r'fsync\(\d+\) failed|No space left|Input/output error|'
                     r'^ERROR|\bERROR:|incorrect data', re.I)
SCORES = (('SCORE bandwidth', 'GiB/s'), ('SCORE IOPS', 'kiops'),
          ('SCORE total', ''))


def count_errors(resultdir):
  """Count error lines in a run's benchmark outputs.

  @param resultdir the run's IO500 result directory (ior/mdtest *.txt)
  @return number of lines reporting a failed operation, or None when the
          directory is not readable from here
  """
  files = glob.glob(os.path.join(resultdir, '*.txt'))
  if not files:
    return None
  n = 0
  for path in files:
    with open(path, errors='replace') as f:
      n += sum(1 for line in f if ERRLINE.search(line))
  return n


def parse(path):
  """Read one job output.

  @param path the PBS output file of a pbs_io500_aurora.sh job
  @return dict: name, group, mode, nodes, ppn, caps, rc, errors, phases
          {name: value}, units {name: unit}, score (bw, iops, total) or None
  """
  name = os.path.splitext(os.path.basename(path))[0]
  run = {'name': name, 'group': re.sub(r'-\d+$', '', name), 'mode': '?',
         'nodes': '?', 'ppn': '?', 'caps': '', 'rc': None, 'errors': None,
         'phases': {}, 'units': {}, 'score': None}
  with open(path, errors='replace') as f:
    for line in f:
      m = HEAD.search(line)
      if m:
        run['mode'], run['nodes'], run['ppn'] = m.group(1, 2, 3)
        run['errors'] = count_errors(os.path.join(m.group(4), 'results'))
      m = CAPS.search(line)
      if m:
        run['caps'] = m.group(1).strip()
      m = RESULT.search(line)
      if m:
        run['phases'][m.group(1)] = float(m.group(2))
        run['units'][m.group(1)] = m.group(3)
      m = SCORE.search(line)
      if m:
        run['score'] = tuple(float(x) for x in m.groups())
      m = CELL.search(line)
      if m:
        run['rc'] = int(m.group(1))
  for i, (label, unit) in enumerate(SCORES):
    if run['score']:
      run['phases'][label] = run['score'][i]
    run['units'][label] = unit
  return run


def stats(values):
  """Mean, min, max and coefficient of variation of a list.

  @param values the per-run values (may be empty)
  @return (mean, lo, hi, cv) or None when there is no value; cv is 0 for a
          single run
  """
  if not values:
    return None
  mean = sum(values) / len(values)
  if len(values) < 2 or mean == 0:
    return mean, min(values), max(values), 0.0
  var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
  return mean, min(values), max(values), math.sqrt(var) / mean


def run_lines(runs):
  """The per-run fitness list.

  @param runs parsed runs
  @return (markdown lines, True when every run is fit to compare)
  """
  lines, ok = [], True
  caps = {r['caps'] for r in runs}
  for r in runs:
    notes = []
    if r['score'] is None:
      notes.append('NO SCORE')
    if r['rc'] not in (0,):
      notes.append(f'rc={r["rc"]}')
    if r['errors']:
      notes.append(f'{r["errors"]} error line(s) in the benchmark outputs')
    if r['errors'] is None:
      notes.append('benchmark outputs not readable (errors not counted)')
    ok = ok and r['score'] is not None and r['rc'] == 0 and not r['errors']
    lines.append(f'- **{r["name"]}**: mode {r["mode"]}, {r["nodes"]} nodes x '
                 f'{r["ppn"]} ranks, {r["caps"] or "caps unknown"}; '
                 f'{"; ".join(notes) if notes else "ok"}')
  if len(caps) > 1:
    ok = False
    lines.append('- **caps differ between runs: not comparable**')
  return lines, ok


def table(runs, base_name, spread):
  """The phase table over groups of runs.

  @param runs parsed runs
  @param base_name group name or mode the ratios are over
  @param spread coefficient of variation above which a phase is marked
  @return markdown lines
  """
  groups = []
  for r in runs:
    if r['group'] not in groups:
      groups.append(r['group'])
  members = {g: [r for r in runs if r['group'] == g] for g in groups}
  base = base_name if base_name in groups else next(
      (g for g in groups if members[g][0]['mode'] == base_name), None)
  others = [g for g in groups if g != base] if base else []
  phases = []
  for r in runs:
    phases += [p for p in r['phases'] if p not in phases]
  phases.sort(key=lambda p: p.startswith('SCORE'))
  hdr = ['phase', 'unit']
  hdr += [f'{g} (n={len(members[g])})' for g in groups]
  hdr += [f'{g} / {base}' for g in others]
  lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
  for p in phases:
    unit = next(r['units'][p] for r in runs if p in r['units'])
    st = {g: stats([r['phases'][p] for r in members[g] if p in r['phases']])
          for g in groups}
    noisy = any(s and s[3] > spread for s in st.values())
    label = f'**{p}**' if p.startswith('SCORE') else p
    cells = [label + (' ~' if noisy else ''), unit]
    for g in groups:
      s = st[g]
      cells.append('n/a' if s is None else
                   f'{s[0]:.2f} ({s[1]:.2f}-{s[2]:.2f}, cv {s[3] * 100:.0f}%)')
    for g in others:
      s, b = st[g], st[base]
      cells.append('n/a' if not s or not b or not b[0]
                   else f'{s[0] / b[0]:.2f}x')
    lines.append('| ' + ' | '.join(cells) + ' |')
  return lines


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('paths', nargs='+')
  ap.add_argument('--base', default='daos',
                  help='group name or mode the ratios are over')
  ap.add_argument('--spread', type=float, default=0.10,
                  help='coefficient of variation above which a phase is '
                       'marked "~"')
  ap.add_argument('--out', default='')
  a = ap.parse_args()

  files = []
  for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, '*.out'))) if os.path.isdir(p) \
        else [p]
  runs = [parse(f) for f in files]
  if not runs:
    print('no job outputs', file=sys.stderr)
    return 1
  fit, ok = run_lines(runs)
  lines = ['# IO500', ''] + fit + [
      '', 'Work per rank is capped, so the scores are not valid IO500 '
      'submissions; compare runs with the same caps only. A cell is the '
      'mean over the runs (lowest-highest run, coefficient of variation); '
      f'"~" marks a phase whose runs differ by more than '
      f'{a.spread * 100:.0f}%.', '']
  lines += table(runs, a.base, a.spread)
  text = '\n'.join(lines) + '\n'
  if a.out:
    with open(a.out, 'w') as f:
      f.write(text)
    print(f'io500 table: {a.out}')
  else:
    print(text)
  return 0 if ok else 1


if __name__ == '__main__':
  sys.exit(main())
