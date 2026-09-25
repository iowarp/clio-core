#!/usr/bin/env python3
"""Summarize the E1 scaling ladder logs (scale_<N>_<group>.log).

Per workload, per substrate: time per iteration/step/pass, ratio to MPI, and
communication (slowest rank's COMM time over the run and its share).
Baselines' communication = wall time inside the substrate's exchanges;
Eternia's = GPU time inside Fetch/Hold/Flush (grayscott: resolve+release).
"""
import glob, os, re, sys

D = os.path.dirname(os.path.abspath(__file__))
ITERS = None
rows = []
# Reruns (rerun_<N>_<group>.log) come after the ladder logs and replace
# their cells.
LOGS = sorted(glob.glob(os.path.join(D, 'scale_*_*.log')),
              key=lambda p: int(re.search(r'_(\d+)_', os.path.basename(p)).group(1)))
LOGS += sorted(glob.glob(os.path.join(D, 'rerun_*_*.log')))
for log in LOGS:
    n = int(re.search(r'_(\d+)_', os.path.basename(log)).group(1))
    txt = open(log, errors='replace').read()
    commmax = {m.group(1): float(m.group(2)) for m in re.finditer(
        r'^COMMMAX e1/(\w+?)_b\d+_(\w+?)x\d+: ([\d.]+) ms', txt, re.M)
        for _ in [0]} if False else {}
    for m in re.finditer(r'^COMMMAX e1/(\w+)_b\d+_(\w+)x\d+: ([\d.]+) ms', txt, re.M):
        commmax[(m.group(1), m.group(2))] = float(m.group(3))
    result = {(m.group(1), m.group(2)): m.group(3) for m in re.finditer(
        r'^RESULT e1/(\w+)_b\d+_(\w+)x\d+: (\S+)', txt, re.M)}
    t = {}
    for sub, key in (('mpi', 'MPI'), ('ccl', 'ONECCL'), ('ishmem', 'ISHMEM')):
        m = re.search(r'KMEANS %s:.*?ms_per_iter=([\d.]+)' % key, txt)
        if m: t[('kmeans', sub)] = float(m.group(1))
        m = re.search(r'GRAYSCOTT %s:.*?ms_per_step=([\d.]+)' % key, txt)
        if m: t[('grayscott', sub)] = float(m.group(1))
        m = re.search(r'GMX %s:.*?passes=(\d+).*?spread_ms=([\d.]+) gather_ms=([\d.]+)' % key, txt)
        if m: t[('gmx', sub)] = (float(m.group(2)) + float(m.group(3))) / int(m.group(1))
        m = re.search(r'LBANN %s:.*?ms_per_step=([\d.]+)' % key, txt)
        if m: t[('lbann', sub)] = float(m.group(1))
    # Eternia: slowest rank printed in the rank tails
    km = [float(ms) / int(it) for it, ms in re.findall(
        r'KMEANS mode=paged.*? iters=(\d+).*? ms=([\d.]+)', txt)]
    if km: t[('kmeans', 'eternia')] = max(km)
    gs = [float(ms) / int(st) for st, ms in re.findall(
        r'GRAYSCOTT mode=paged.*? steps=(\d+).*? ms=([\d.]+)', txt)]
    if gs: t[('grayscott', 'eternia')] = max(gs)
    gx = [float(a) + float(b) for a, b in re.findall(
        r'spread ([\d.]+) ms \(dense [\d.]+\)\s+gather\+sum ([\d.]+) ms', txt)]
    if gx: t[('gmx', 'eternia')] = max(gx)
    lb = [float(x) for x in re.findall(r'paged ([\d.]+) ms/step', txt)]
    if lb: t[('lbann', 'eternia')] = max(lb)
    for (wl, sub), res in result.items():
        rows.append((n, wl, sub, t.get((wl, sub)), commmax.get((wl, sub)), res))

print('%-4s %-10s %-8s %12s %8s %14s  %s' % ('N', 'workload', 'subst', 'ms/it', 'xMPI', 'comm max(ms)', 'result'))
by = {}
for r in rows:
    if r[5] is not None and (r[3] is not None or (r[0], r[1], r[2]) not in by):
        by[(r[0], r[1], r[2])] = r
for r in sorted(by.values(), key=lambda r: (r[0], r[1], ['mpi', 'ccl', 'ishmem', 'eternia'].index(r[2]))):
    n, wl, sub, ms, cm, res = r
    mpi = by.get((n, wl, 'mpi'))
    ratio = (ms / mpi[3]) if (ms and mpi and mpi[3]) else None
    print('%-4d %-10s %-8s %12s %8s %14s  %s' % (
        n, wl, sub, '%.1f' % ms if ms else '-', '%.2f' % ratio if ratio else '-',
        '%.1f' % cm if cm is not None else '-', res))
