"""E4 tier composition as absolute runtime, one panel per workload (2 nodes).
Same cells and parsing as plot_e2345.e4(); slowest rank's timed region."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_e2345 import D, OUT, COMPS, ms_of

CLAB = ['D', 'DH', 'B', 'DA', 'L']
WLS = [('kmeans', 'kmeans', 's'), ('grayscott', 'Gray-Scott', 's'),
       ('gmx', 'gmx', 's'), ('lbann', 'lbann', 's'), ('lammps_md', 'LAMMPS-MD', 's')]
fig, axg = plt.subplots(2, 3, figsize=(3.5, 2.9))
axes = list(axg.flat)
axes[5].axis('off')
axes[5].text(0.02, 0.95, 'DRAM/DAOS/Lustre\n(% of capacity)\n\nD   100/0/0\nDH  75/25/0\nB   25/50/25\nDA  10/70/20\nL   10/20/70',
             transform=axes[5].transAxes, va='top', fontsize=6.3, family='monospace')
for ax, (wl, name, unit) in zip(axes, WLS):
    f = lambda c: os.path.join(D, ('%s_e4_%s_p1024_s16.log' if wl == 'grayscott' else '%s_e4_%s.log') % (wl, c))
    ys = [ms_of(f(c), wl) for c in COMPS]
    print(wl, ys)
    xs = range(len(COMPS))
    ax.bar(xs, [(y or 0) / 1000.0 for y in ys], 0.7, color='#1f77b4')
    ax.set_xticks(list(xs)); ax.set_xticklabels(CLAB, fontsize=6.5)
    ax.set_title(name, fontsize=7.5, pad=2); ax.tick_params(axis='y', labelsize=6.5)
    ax.grid(alpha=.3, axis='y')
axes[0].set_ylabel('runtime (s)', fontsize=7); axes[3].set_ylabel('runtime (s)', fontsize=7)
fig.tight_layout(pad=0.3, w_pad=0.4, h_pad=0.5)
fig.savefig(os.path.join(OUT, 'e4_runtime.pdf')); fig.savefig(os.path.join(OUT, 'e4_runtime.png'), dpi=200)
print('wrote', os.path.join(OUT, 'e4_runtime.pdf'))
