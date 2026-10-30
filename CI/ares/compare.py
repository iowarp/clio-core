#!/usr/bin/env python3
"""Compare an Ares run against a baseline run and flag regressions.

  compare.py RUN_DIR [--baseline BASE_DIR] [--threshold 0.20] [--out report.md]

RUN_DIR (and BASE_DIR) are CI/ares/run.sh output directories: one
sub-directory per suite leg, each holding the clio_fs_suite results.jsonl.

Reliability: every test must be PASS or UNSUP. A test that passed in the
baseline and does not now is a REGRESSION; one that also failed in the
baseline is reported as STILL FAILING.

Performance: every numeric metric is compared with the baseline's value for
the same leg/test/metric. Direction comes from the name: *_mb_s, *_iops,
*_ops_s are higher-is-better; *_us, *_ms, *_secs, *_s (not *_mb_s/_ops_s) are
lower-is-better; anything else is reported, never judged. A metric worse by
more than --threshold (fraction) is a PERF REGRESSION.

Exit 0 when nothing regressed (and every test passed), 1 otherwise.
"""

import argparse
import glob
import json
import os
import sys

HIGHER = ('_mb_s', '_iops', '_ops_s')
LOWER = ('_us', '_ms', '_secs', '_s')


def direction(name):
  """+1 higher is better, -1 lower is better, 0 unknown."""
  if name.endswith(HIGHER):
    return 1
  if name.endswith(LOWER) or '_p99_' in name or '_p50_' in name:
    return -1
  return 0


def load(run_dir):
  """{(leg, test): record} from every leg's results.jsonl (last record of a
  test wins, so a resumed leg reports its final attempt)."""
  out = {}
  for path in sorted(glob.glob(os.path.join(run_dir, '*', 'results.jsonl'))):
    leg = os.path.basename(os.path.dirname(path))
    with open(path) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          rec = json.loads(line)
        except ValueError:
          continue
        out[(leg, rec.get('name', '?'))] = rec
  return out


def numeric(metrics):
  """Flatten numeric metrics (dict values one level deep are skipped: they
  are descriptive, e.g. tier_usage)."""
  return {k: v for k, v in (metrics or {}).items()
          if isinstance(v, (int, float)) and not isinstance(v, bool)}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('run')
  ap.add_argument('--baseline', default='')
  ap.add_argument('--threshold', type=float, default=0.20)
  ap.add_argument('--out', default='')
  a = ap.parse_args()

  cur = load(a.run)
  base = load(a.baseline) if a.baseline else {}
  if not cur:
    print(f'no results.jsonl under {a.run}', file=sys.stderr)
    return 1

  rel_regress, still_fail, perf_regress, perf_better, rows = [], [], [], [], []
  for key in sorted(cur):
    leg, name = key
    rec = cur[key]
    st = rec.get('status')
    bst = base.get(key, {}).get('status')
    if st not in ('PASS', 'UNSUP'):
      (still_fail if bst and bst not in ('PASS', 'UNSUP') else
       rel_regress).append((leg, name, st, bst,
                            (rec.get('error') or '')[:300]))
    if key not in base:
      continue
    m0 = numeric(base[key].get('metrics'))
    for mk, v in numeric(rec.get('metrics')).items():
      if mk not in m0 or m0[mk] == 0:
        continue
      d = direction(mk)
      change = (v - m0[mk]) / abs(m0[mk])
      rows.append((leg, name, mk, m0[mk], v, change, d))
      if d and change * d < -a.threshold:
        perf_regress.append((leg, name, mk, m0[mk], v, change))
      elif d and change * d > a.threshold:
        perf_better.append((leg, name, mk, m0[mk], v, change))

  lines = [f'# Ares run report: `{a.run}`', '',
           f'baseline: `{a.baseline or "(none)"}`, '
           f'perf threshold {int(a.threshold * 100)}%', '']
  counts = {}
  for rec in cur.values():
    counts[rec.get('status')] = counts.get(rec.get('status'), 0) + 1
  lines.append('**tests:** ' + ', '.join(f'{k} {v}' for k, v in
                                          sorted(counts.items())))
  lines.append('')

  def table(title, hdr, items, fmt):
    lines.append(f'## {title} ({len(items)})')
    lines.append('')
    if items:
      lines.append('| ' + ' | '.join(hdr) + ' |')
      lines.append('|' + '---|' * len(hdr))
      for it in items:
        lines.append('| ' + ' | '.join(fmt(it)) + ' |')
    else:
      lines.append('none')
    lines.append('')

  table('Reliability regressions', ['leg', 'test', 'now', 'baseline',
                                    'error'], rel_regress,
        lambda r: [r[0], r[1], str(r[2]), str(r[3] or 'n/a'),
                   r[4].replace('|', '/').replace('\n', ' ')])
  table('Still failing (also failed in baseline)', ['leg', 'test', 'now',
                                                    'error'], still_fail,
        lambda r: [r[0], r[1], str(r[2]),
                   r[4].replace('|', '/').replace('\n', ' ')])
  pf = lambda r: [r[0], r[1], r[2], f'{r[3]:g}', f'{r[4]:g}',
                  f'{r[5] * 100:+.1f}%']
  table('Performance regressions', ['leg', 'test', 'metric', 'baseline',
                                    'now', 'change'], perf_regress, pf)
  table('Performance improvements', ['leg', 'test', 'metric', 'baseline',
                                     'now', 'change'], perf_better, pf)
  lines.append('## All compared metrics')
  lines.append('')
  lines.append('| leg | test | metric | baseline | now | change | judged |')
  lines.append('|---|---|---|---|---|---|---|')
  for leg, name, mk, b, v, c, d in rows:
    lines.append(f'| {leg} | {name} | {mk} | {b:g} | {v:g} | '
                 f'{c * 100:+.1f}% | {"yes" if d else "no"} |')
  report = '\n'.join(lines) + '\n'
  out = a.out or os.path.join(a.run, 'report.md')
  with open(out, 'w') as f:
    f.write(report)
  print(f'report: {out}')
  print(f'reliability regressions: {len(rel_regress)}, still failing: '
        f'{len(still_fail)}, perf regressions: {len(perf_regress)}')
  return 1 if (rel_regress or still_fail or perf_regress) else 0


if __name__ == '__main__':
  sys.exit(main())
