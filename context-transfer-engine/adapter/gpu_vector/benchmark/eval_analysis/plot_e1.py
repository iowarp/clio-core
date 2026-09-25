#!/usr/bin/env python3
"""E1 figures from the scaling-ladder logs (scale_<N>_<group>.log).

  e1_scaling.pdf  time per iteration vs node count, one panel per workload,
                  one line per substrate (weak scaling: flat is ideal)
  e1_comm.pdf     communication as a share of the run, slowest rank, per
                  substrate and node count (baselines: wall time inside
                  their exchanges; Eternia: GPU time in Fetch/Hold/Flush,
                  grayscott: resolve+release+publish)

Run with a Python that has matplotlib, e.g.
  /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3 plot_e1.py [outdir]
"""
import glob, os, re, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'figs')
os.makedirs(OUT, exist_ok=True)

WLS = [('kmeans', 'ms/iteration'), ('grayscott', 'ms/step'),
       ('gmx', 'ms/pass'), ('lbann', 'ms/step')]
SUBS = [('mpi', 'MPI'), ('ccl', 'oneCCL'), ('ishmem', 'Intel SHMEM'),
        ('eternia', 'Eternia')]
STYLE = {'mpi': dict(marker='o'), 'ccl': dict(marker='s'),
         'ishmem': dict(marker='^'), 'eternia': dict(marker='D', lw=2.5)}


def parse():
    """{(nodes, wl, sub): (ms_per_it, iters, comm_max_ms)}"""
    res = {}
    logs = sorted(glob.glob(os.path.join(D, 'scale_*_*.log')))
    logs += sorted(glob.glob(os.path.join(D, 'rerun_*_*.log')))  # override
    for log in logs:
        n = int(re.search(r'_(\d+)_', os.path.basename(log)).group(1))
        txt = open(log, errors='replace').read()
        ok = {(m.group(1), m.group(2)) for m in re.finditer(
            r'^RESULT e1/(\w+)_b\d+_(\w+)x\d+: OK', txt, re.M)}
        cm = {(m.group(1), m.group(2)): float(m.group(3)) for m in re.finditer(
            r'^COMMMAX e1/(\w+)_b\d+_(\w+)x\d+: ([\d.]+) ms', txt, re.M)}
        t = {}
        for sub, key in (('mpi', 'MPI'), ('ccl', 'ONECCL'), ('ishmem', 'ISHMEM')):
            m = re.search(r'KMEANS %s:.*?iters=(\d+).*?ms_per_iter=([\d.]+)' % key, txt)
            if m: t[('kmeans', sub)] = (float(m.group(2)), int(m.group(1)))
            m = re.search(r'GRAYSCOTT %s:.*?steps=(\d+).*?ms_per_step=([\d.]+)' % key, txt)
            if m: t[('grayscott', sub)] = (float(m.group(2)), int(m.group(1)))
            m = re.search(r'GMX %s:.*?passes=(\d+).*?spread_ms=([\d.]+) gather_ms=([\d.]+)' % key, txt)
            if m:
                p = int(m.group(1))
                t[('gmx', sub)] = ((float(m.group(2)) + float(m.group(3))) / p, p)
            m = re.search(r'LBANN %s:.*?steps=(\d+).*?ms_per_step=([\d.]+)' % key, txt)
            if m: t[('lbann', sub)] = (float(m.group(2)), int(m.group(1)))
        km = [(float(ms) / int(it), int(it)) for it, ms in re.findall(
            r'KMEANS mode=paged.*? iters=(\d+).*? ms=([\d.]+)', txt)]
        if km: t[('kmeans', 'eternia')] = max(km)
        gs = [(float(ms) / int(st), int(st)) for st, ms in re.findall(
            r'GRAYSCOTT mode=paged.*? steps=(\d+).*? ms=([\d.]+)', txt)]
        if gs: t[('grayscott', 'eternia')] = max(gs)
        gx = [float(a) + float(b) for a, b in re.findall(
            r'spread ([\d.]+) ms \(dense [\d.]+\)\s+gather\+sum ([\d.]+) ms', txt)]
        np_ = re.search(r'gmx: (\d+) iterations', txt)
        if gx: t[('gmx', 'eternia')] = (max(gx), int(np_.group(1)) if np_ else 1)
        lb = [float(x) for x in re.findall(r'paged ([\d.]+) ms/step', txt)]
        ns = re.search(r'lbann: (\d+) iterations', txt)
        if lb: t[('lbann', 'eternia')] = (max(lb), int(ns.group(1)) if ns else 1)
        for (wl, sub), (ms, it) in t.items():
            if (wl, sub) in ok:
                res[(n, wl, sub)] = (ms, it, cm.get((wl, sub)))
    return res


def main():
    res = parse()
    nodes = sorted({k[0] for k in res})
    if not nodes:
        print('no results yet'); return
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, (wl, unit) in zip(axes, WLS):
        for sub, label in SUBS:
            xs = [n for n in nodes if (n, wl, sub) in res]
            ys = [res[(n, wl, sub)][0] for n in xs]
            if xs: ax.plot(xs, ys, label=label, **STYLE[sub])
        ax.set_title(wl); ax.set_xlabel('nodes'); ax.set_ylabel(unit)
        ax.set_xscale('log', base=2); ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes]); ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'e1_scaling.pdf')); fig.savefig(os.path.join(OUT, 'e1_scaling.png'), dpi=110)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    w = 0.2
    for ax, (wl, _) in zip(axes, WLS):
        for i, (sub, label) in enumerate(SUBS):
            xs, ys = [], []
            for j, n in enumerate(nodes):
                r = res.get((n, wl, sub))
                if r and r[2] is not None:
                    xs.append(j + (i - 1.5) * w)
                    ys.append(100.0 * r[2] / (r[0] * r[1]))
            if xs: ax.bar(xs, ys, w, label=label)
        ax.set_title(wl); ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels([str(n) for n in nodes]); ax.set_xlabel('nodes')
        ax.set_ylabel('communication (% of run)'); ax.grid(alpha=0.3, axis='y')
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'e1_comm.pdf')); fig.savefig(os.path.join(OUT, 'e1_comm.png'), dpi=110)
    print('wrote', OUT, 'nodes', nodes, 'cells', len(res))


if __name__ == '__main__':
    main()
