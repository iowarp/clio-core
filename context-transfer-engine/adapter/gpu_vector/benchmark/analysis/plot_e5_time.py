"""E5 time-only figure: one panel per workload, time normalised to the
row's largest budget. Values are the slowest-rank times behind tab:e5 in
5-evaluation.tex (sources in EVALUATION.md, 'E5 collected as-is'):
kmeans/Gray-Scott 4n e5_4.log; kmeans 16n E5_16a/E5_16/E5_16c; Gray-Scott 16n
E5_16gA/E5_16gB (8 GB from E5_16b's first timed run); gmx 4n e5_e4deck
(8866821); lbann 4n E4-batch cap sweep lbann_e4_dram75_p1024_s*."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = '/lus/flare/projects/IOWarp/llogan_e1/figs'
os.makedirs(OUT, exist_ok=True)

# workload -> list of (label, [(budget GB/node, runtime s)]). Runtime = wall
# time of the timed region on the slowest rank: kmeans 8 iterations,
# Gray-Scott 8 steps, gmx 1 pass, lbann 1 step.
DATA = {
    'kmeans': [('4 nodes',  [(32, 73.6), (24, 89.4), (16, 103.4), (8, 106.0), (4, 98.4)]),
               ('16 nodes', [(32, 93.5), (24, 114.5), (16, 137.3), (8, 134.4), (4, 140.4)])],
    'Gray-Scott': [('4 nodes',  [(32, 218.3), (24, 231.2), (16, 231.0), (8, 235.2), (4, 193.2)]),
                   ('16 nodes', [(32, 299.5), (24, 323.4), (16, 355.1), (8, 324.1), (4, 301.5)])],
    'gmx': [('4 nodes', [(8.2, 7.31), (4.1, 6.79), (2.05, 6.66), (1.37, 6.55)])],
    'lbann': [('4 nodes', [(8, 22.4), (4, 25.1), (2, 21.9), (1, 19.7)])],
}
DECK = {'kmeans': 64, 'Gray-Scott': 64, 'gmx': 8, 'lbann': 8}
STYLE = {'4 nodes': dict(marker='o', color='#1f77b4'),
         '16 nodes': dict(marker='s', color='#d62728', ls='--')}

fig, axes = plt.subplots(1, 4, figsize=(3.5, 1.55))
axes = axes.reshape(1, 4)
for ax, (wl, series) in zip(axes.flat, DATA.items()):
    for lab, pts in series:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=lab, lw=1.0, ms=2.5, **STYLE[lab])
    xs = sorted({p[0] for _, pts in series for p in pts})
    ax.set_xscale('log', base=2); ax.set_xticks(xs)
    ax.set_xticklabels(['%g' % x for x in xs], fontsize=4.8, rotation=90)
    ax.minorticks_off(); ax.invert_xaxis()
    ax.set_ylim(0, 1.18 * max(p[1] for _, pts in series for p in pts)); ax.tick_params(axis='y', labelsize=5.5, pad=1); ax.tick_params(axis='x', pad=1, length=2)
    ax.set_title('%s\n(%d GB/node)' % (wl, DECK[wl]), fontsize=6, pad=2)
    ax.grid(alpha=.3)
fig.supxlabel('GPU memory budget (GB/node)', fontsize=6, y=0.0)
axes[0, 0].set_ylabel('runtime (s)', fontsize=6, labelpad=1)
axes[0, 0].legend(fontsize=5, loc='lower center', frameon=False, handlelength=1.5)
fig.tight_layout(pad=0.2, w_pad=0.3, rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(OUT, 'e5_runtime.pdf'))
fig.savefig(os.path.join(OUT, 'e5_runtime.png'), dpi=200)
print('wrote', os.path.join(OUT, 'e5_runtime.pdf'))
