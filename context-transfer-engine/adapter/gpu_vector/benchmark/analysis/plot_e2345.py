#!/usr/bin/env python3
"""Figures for E2 (page size), E3 (checkpointing), E4 (tier composition) and
E5 (HBM budget) from the batch/cell logs.

  e2_page.pdf   kmeans time vs page size, DRAM-only vs 70% Lustre
  e3_ckpt.pdf   grayscott time: no checkpoint / async / sync (drain), per tier
  e4_tiers.pdf  time per tier composition, normalised to DRAM-only, per workload
  e5_hbm.pdf    time and energy vs HBM budget, normalised to 32 GB, per
                workload and node count

Run with a Python that has matplotlib:
  /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3 plot_e2345.py
"""
import glob, os, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/home/llogan/clio-core/.claude/worktrees/gpu-coro/build-spike/pbs'
FL = '/lus/flare/projects/IOWarp/llogan_e1'
OUT = os.path.join(FL, 'figs'); os.makedirs(OUT, exist_ok=True)
COMPS = ['dram100', 'dram75', 'bal25', 'daos70', 'lustre70']
CLAB = ['DRAM', '75% DRAM', 'balanced', '70% DAOS', '70% Lustre']


def ms_of(path, wl):
    """Slowest rank's timed ms from one cell log (None if missing/failed)."""
    if not os.path.exists(path):
        return None
    t = open(path, errors='replace').read()
    if not re.search(r'^RESULT \S+: OK', t, re.M):
        return None
    pats = {'kmeans': r'KMEANS mode=paged.*? ms=([\d.]+)',
            'grayscott': r'GRAYSCOTT mode=paged.*? ms=([\d.]+)',
            'lbann': r'paged ([\d.]+) ms/step',
            'lammps_md': r'\d+ steps in ([\d.]+) ms'}
    if wl == 'gmx':
        v = [float(a) + float(b) for a, b in re.findall(
            r'spread ([\d.]+) ms.*?gather\+sum ([\d.]+) ms', t)]
    else:
        v = [float(x) for x in re.findall(pats[wl], t)]
    return max(v) if v else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=110)


def e2():
    fig, ax = plt.subplots(figsize=(4.2, 3))
    pk = [64, 256, 1024, 4096, 16384]
    for comp, lab in (('dram100', 'DRAM only'), ('lustre70', '70% Lustre')):
        ys = [ms_of(os.path.join(D, 'kmeans_e4_%s_p%d_s%d.log' % (comp, p, max(1, 8192 // p))),
                    'kmeans') for p in pk]
        xs = [p for p, y in zip(pk, ys) if y]
        ax.plot(xs, [y / 1000 for y in ys if y], marker='o', label=lab)
    ax.set_xscale('log', base=2); ax.set_xticks(pk)
    ax.set_xticklabels(['64K', '256K', '1M', '4M', '16M'])
    ax.set_xlabel('page size'); ax.set_ylabel('time (s)'); ax.set_ylim(bottom=0)
    ax.set_title('E2: kmeans, cache held constant'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    save(fig, 'e2_page')


def e3():
    fig, ax = plt.subplots(figsize=(4.8, 3))
    arms = [('nockpt', 'no checkpoint'), ('ckpt', 'async checkpoint'), ('ckptdrain', 'sync (drain)')]
    comps = [('dram100', 'DRAM'), ('bal25', 'balanced'), ('lustre70', '70% Lustre')]
    w = 0.27
    for i, (a, lab) in enumerate(arms):
        ys = [ms_of(os.path.join(D, 'grayscott_e4_%s_%s_p1024_s16.log' % (a, c)), 'grayscott')
              for c, _ in comps]
        ax.bar([j + (i - 1) * w for j in range(len(comps))], [(y or 0) / 1000 for y in ys], w, label=lab)
    ax.set_xticks(range(len(comps))); ax.set_xticklabels([l for _, l in comps])
    ax.set_ylabel('time (s)'); ax.set_title('E3: grayscott, 8 steps, ckpt every 2')
    ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=7)
    save(fig, 'e3_ckpt')


def e4():
    fig, ax = plt.subplots(figsize=(6.5, 3))
    wls = ['kmeans', 'gmx', 'grayscott', 'lbann', 'lammps_md']
    w = 0.16
    for i, wl in enumerate(wls):
        f = lambda c: os.path.join(D, ('%s_e4_%s_p1024_s16.log' if wl == 'grayscott' else '%s_e4_%s.log') % (wl, c))
        ys = [ms_of(f(c), wl) for c in COMPS]
        base = ys[0]
        ax.bar([j + (i - 2) * w for j in range(len(COMPS))],
               [(y / base) if (y and base) else 0 for y in ys], w, label=wl)
    ax.set_xticks(range(len(COMPS))); ax.set_xticklabels(CLAB, fontsize=8)
    ax.set_ylabel('time / DRAM-only'); ax.set_title('E4: tier composition')
    ax.grid(alpha=.3, axis='y'); ax.legend(fontsize=7, ncol=3)
    save(fig, 'e4_tiers')


def e5():
    cells = {}
    logs = glob.glob(os.path.join(D, 'e5_*.log')) + glob.glob(os.path.join(FL, 'e5_*.log'))
    for log in logs:
        for wl, gb, n, ms, ej, rc in re.findall(
                r'^E5CELL (\w+) hbm_gb=(\d+) nodes=(\d+) ms=([\d.]+) energy_J=(\d+) rc=(\d+)',
                open(log, errors='replace').read(), re.M):
            if rc == '0' and float(ms) > 0:
                cells[(wl, int(n), int(gb))] = (float(ms), float(ej))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for (wl, n) in sorted({(k[0], k[1]) for k in cells}):
        gbs = sorted(g for (w, nn, g) in cells if w == wl and nn == n)
        if 32 not in gbs:
            continue
        t0, e0 = cells[(wl, n, 32)]
        lab = '%s %dn' % (wl, n)
        axes[0].plot(gbs, [cells[(wl, n, g)][0] / t0 for g in gbs], marker='o', label=lab)
        axes[1].plot(gbs, [cells[(wl, n, g)][1] / e0 for g in gbs], marker='o', label=lab)
    for ax, yl in zip(axes, ('time / 32 GB', 'energy / 32 GB')):
        ax.set_xscale('log', base=2); ax.set_xticks([4, 8, 16, 24, 32])
        ax.set_xticklabels(['4', '8', '16', '24', '32']); ax.invert_xaxis()
        ax.axhline(1.1, ls='--', c='gray', lw=1); ax.set_xlabel('HBM budget (GB)')
        ax.set_ylabel(yl); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7); axes[0].set_title('E5: 64 GB/node deck')
    save(fig, 'e5_hbm')
    return cells


if __name__ == '__main__':
    e2(); e3(); e4(); c = e5()
    print('wrote', OUT, 'e5 cells', len(c))
