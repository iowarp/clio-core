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

# workload -> list of (label, [(budget GB, normalised time)], deck GB/node)
DATA = {
    'kmeans': [('4 nodes',  [(32, 1.00), (24, 1.21), (16, 1.40), (8, 1.44), (4, 1.34)]),
               ('16 nodes', [(32, 1.00), (24, 1.22), (16, 1.47), (8, 1.44), (4, 1.35)])],
    'Gray-Scott': [('4 nodes',  [(32, 1.00), (24, 1.06), (16, 1.06), (8, 1.08), (4, 0.88)]),
                   ('16 nodes', [(32, 1.00), (24, 1.08), (16, 1.19), (8, 0.94), (4, 1.01)])],
    'gmx': [('4 nodes', [(8.2, 1.00), (4.1, 0.93), (2.05, 0.91), (1.37, 0.90)])],
    'lbann': [('4 nodes', [(8, 1.00), (4, 1.12), (2, 0.98), (1, 0.88)])],
}
DECK = {'kmeans': 64, 'Gray-Scott': 64, 'gmx': 8, 'lbann': 8}
STYLE = {'4 nodes': dict(marker='o', color='#1f77b4'),
         '16 nodes': dict(marker='s', color='#d62728', ls='--')}

fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.1), sharey=True)
for ax, (wl, series) in zip(axes.flat, DATA.items()):
    for lab, pts in series:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=lab, lw=1.2, ms=3.5, **STYLE[lab])
    xs = sorted({p[0] for _, pts in series for p in pts})
    ax.set_xscale('log', base=2); ax.set_xticks(xs)
    ax.set_xticklabels(['%g' % x for x in xs], fontsize=6.5)
    ax.minorticks_off(); ax.invert_xaxis()
    ax.axhline(1.0, c='gray', lw=0.6)
    ax.set_ylim(0.8, 1.6); ax.tick_params(axis='y', labelsize=7)
    ax.set_title('%s (%d GB/node)' % (wl, DECK[wl]), fontsize=7.5, pad=2)
    ax.grid(alpha=.3)
fig.supxlabel('GPU memory budget (GB/node)', fontsize=7, y=0.01)
for ax in axes[:, 0]:
    ax.set_ylabel('normalised time', fontsize=7)
axes[0, 0].legend(fontsize=6.5, loc='lower right', frameon=False)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4, rect=(0, 0.03, 1, 1))
fig.savefig(os.path.join(OUT, 'e5_time.pdf'))
fig.savefig(os.path.join(OUT, 'e5_time.png'), dpi=200)
print('wrote', os.path.join(OUT, 'e5_time.pdf'))
