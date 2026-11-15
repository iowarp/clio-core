#!/usr/bin/env python3
"""Side-by-side IO500 table from pbs_io500_aurora.sh job outputs.

  compare_io500.py DIR_OR_FILE... [--base daos] [--out io500.md]

Each argument is a job output file (the PBS -o file of
pbs_io500_aurora.sh) or a directory of them (*.out). A column is one run,
named after its file and labelled with the mode, node count and work per
rank the job printed. Rows are IO500's phases and the three scores; the
ratio column is every run over the run named by --base (first whose mode or
name matches; default: the first daos run).

Exit 0 when every run finished with a score, 1 otherwise.
"""

import argparse
import glob
import os
import re
import sys

RESULT = re.compile(r'\[RESULT\]\s+(\S+)\s+([\d.]+)\s+(\S+)\s*:\s*time\s+([\d.]+)')
SCORE = re.compile(r'\[SCORE\s*\]\s+Bandwidth\s+([\d.]+)\s+GiB/s\s*:\s*IOPS\s+'
                   r'([\d.]+)\s+kiops\s*:\s*TOTAL\s+([\d.]+)')
HEAD = re.compile(r'=== IO500 mode=(\S+) nodes=(\d+) ppn=(\d+)')
CAPS = re.compile(r'=== caps: (.*)')
CELL = re.compile(r'IO500CELL .* rc=(\d+)')


def parse(path):
  """Read one job output.

  @param path the PBS output file of a pbs_io500_aurora.sh job
  @return dict: name, mode, nodes, ppn, caps, rc, phases {name: (value,
          unit, seconds)}, score (bw, iops, total) or None
  """
  run = {'name': os.path.splitext(os.path.basename(path))[0], 'mode': '?',
         'nodes': '?', 'ppn': '?', 'caps': '', 'rc': None, 'phases': {},
         'score': None}
  with open(path, errors='replace') as f:
    for line in f:
      m = HEAD.search(line)
      if m:
        run['mode'], run['nodes'], run['ppn'] = m.groups()
      m = CAPS.search(line)
      if m:
        run['caps'] = m.group(1).strip()
      m = RESULT.search(line)
      if m:
        run['phases'][m.group(1)] = (float(m.group(2)), m.group(3),
                                     float(m.group(4)))
      m = SCORE.search(line)
      if m:
        run['score'] = tuple(float(x) for x in m.groups())
      m = CELL.search(line)
      if m:
        run['rc'] = int(m.group(1))
  return run


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('paths', nargs='+')
  ap.add_argument('--base', default='daos',
                  help='mode or file name of the run the ratios are over')
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
  # A file named like --base wins over the first run of that mode.
  base = next((r for r in runs if r['name'] == a.base), None) or \
      next((r for r in runs if r['mode'] == a.base), None)

  phases = []
  for r in runs:
    phases += [p for p in r['phases'] if p not in phases]
  hdr = ['phase', 'unit'] + [r['name'] for r in runs]
  others = [r for r in runs if base is not None and r is not base]
  hdr += [f'{r["name"]} / {base["name"]}' for r in others]
  lines = ['# IO500', '']
  for r in runs:
    state = 'no score' if r['score'] is None else 'scored'
    lines.append(f'- **{r["name"]}**: mode {r["mode"]}, {r["nodes"]} nodes x '
                 f'{r["ppn"]} ranks, {r["caps"] or "caps unknown"}; '
                 f'rc={r["rc"]}, {state}')
  lines += ['', 'Work per rank is capped, so the scores are not valid IO500 '
            'submissions; compare runs with the same caps only.', '',
            '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]

  def row(label, unit, get):
    cells = [label, unit]
    vals = {id(r): get(r) for r in runs}
    cells += ['n/a' if vals[id(r)] is None else f'{vals[id(r)]:.2f}'
              for r in runs]
    for r in others:
      v, b = vals[id(r)], vals[id(base)]
      cells.append('n/a' if v is None or not b else f'{v / b:.2f}x')
    lines.append('| ' + ' | '.join(cells) + ' |')

  for p in phases:
    unit = next(r['phases'][p][1] for r in runs if p in r['phases'])
    row(p, unit, lambda r, p=p: r['phases'].get(p, (None,))[0])
  for i, (label, unit) in enumerate((('SCORE bandwidth', 'GiB/s'),
                                     ('SCORE IOPS', 'kiops'),
                                     ('SCORE total', ''))):
    row(f'**{label}**', unit,
        lambda r, i=i: r['score'][i] if r['score'] else None)
  text = '\n'.join(lines) + '\n'
  if a.out:
    with open(a.out, 'w') as f:
      f.write(text)
    print(f'io500 table: {a.out}')
  else:
    print(text)
  return 0 if all(r['score'] for r in runs) else 1


if __name__ == '__main__':
  sys.exit(main())
