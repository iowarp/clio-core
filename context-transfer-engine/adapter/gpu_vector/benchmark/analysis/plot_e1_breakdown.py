#!/usr/bin/env python3
"""E1 weak scaling as stacked bar charts (e1_breakdown.pdf).

One panel per workload, one group of bars per node count, one bar per
substrate. Each bar is absolute time per iteration split into communication
(hatched, slowest rank) and compute (the rest), so the panels show how each
workload actually scales. Panels have their own y-axis since per-iteration
times differ by two orders of magnitude across workloads.

Reads e1_cells.csv (from e1_extract.py). Run with a Python that has
matplotlib, e.g.
  /opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python3 \
      plot_e1_breakdown.py [out.pdf]
"""
import csv, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

D = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(D, 'e1_breakdown.pdf')

WLS = [('kmeans', 'kmeans'), ('grayscott', 'Gray-Scott'), ('gmx', 'gmx'),
       ('lbann', 'lbann')]
SUBS = [('mpi', 'MPI', '#2a78d6'), ('ccl', 'oneCCL', '#eb6834'),
        ('ishmem', 'Intel SHMEM', '#1baf7a'), ('eternia', 'Eternia', '#4a3aa7')]


def tint(c, a):
    """Blend color c toward white; a=1 keeps c, a=0 is white."""
    return tuple(1 - a * (1 - x) for x in to_rgb(c))


def load():
    """{(wl, nodes, sub): (comm_ms, compute_ms)} per iteration, OK runs only."""
    res = {}
    for r in csv.DictReader(open(os.path.join(D, 'e1_cells.csv'))):
        if r['status'] != 'OK':
            continue
        it = int(r['iters'])
        t = float(r['ms_per_iter'])
        comm = float(r['comm_ms_max']) / it
        res[(r['workload'], int(r['nodes']), r['substrate'])] = (comm, t - comm)
    return res


def main():
    res = load()
    nodes = sorted({k[1] for k in res})
    plt.rcParams.update({'font.size': 8, 'hatch.linewidth': 0.6})
    fig, axes = plt.subplots(1, len(WLS), figsize=(7.0, 2.2))
    w = 0.2
    for ax, (wl, wl_label) in zip(axes, WLS):
        scale, unit = (1000.0, 's') if wl == 'lbann' else (1.0, 'ms')
        top = max(sum(v) for k, v in res.items() if k[0] == wl) / scale
        for j, n in enumerate(nodes):
            for i, (sub, _, col) in enumerate(SUBS):
                r = res.get((wl, n, sub))
                xi = j + (i - 1.5) * w
                if r is None:
                    ax.text(xi, top * 0.02, 'n/a', rotation=90, ha='center',
                            va='bottom', fontsize=5, color='0.45')
                    continue
                comm, comp = r[0] / scale, r[1] / scale
                ax.bar(xi, comp, w * 0.9, color=tint(col, 0.35),
                       edgecolor=col, lw=0.4)
                ax.bar(xi, comm, w * 0.9, bottom=comp, color=col,
                       edgecolor='white', lw=0.4, hatch='/////')
        ax.set_title(wl_label, fontsize=8, fontweight='bold')
        ax.set_xticks(range(len(nodes)))
        ax.set_xticklabels([str(n) for n in nodes], fontsize=7)
        ax.set_xlabel('nodes', fontsize=7)
        ax.set_ylabel('%s per iteration' % unit, fontsize=7)
        ax.set_ylim(0, top * 1.08)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(axis='y', alpha=0.3, lw=0.5)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for _, l, c in SUBS]
    handles += [Patch(facecolor='0.85', edgecolor='0.4', lw=0.4,
                      label='compute'),
                Patch(facecolor='0.4', edgecolor='white', hatch='/////',
                      label='communication')]
    fig.legend(handles=handles, ncol=6, fontsize=7, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 1.0),
               handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT)
    fig.savefig(os.path.splitext(OUT)[0] + '.png', dpi=200)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
