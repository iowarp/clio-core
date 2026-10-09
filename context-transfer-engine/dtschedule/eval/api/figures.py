#!/usr/bin/env python3
"""Paper figures from measured DTSchedule runs (sweep.py result JSONs + decision traces).

Usage: figures.py --root ~/jarvis-runs/dtschedule-results --out <paper>/figures/evals/measured [names...]
Every number comes from a run directory; nothing is hand-entered. Runs whose consumer
saw bad files or whose status is not ok are excluded and counted in the console log.
"""
import argparse
import glob
import json
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order (validated reference palette, light mode).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
    'legend.frameon': False, 'pdf.fonttype': 42, 'figure.dpi': 150})
COL_W = 3.33  # inches, one ACM column


def save(fig, out, name):
    """Write the PDF for the paper and a PNG preview next to it (preview/)."""
    fig.savefig(os.path.join(out, name))
    os.makedirs(os.path.join(out, 'preview'), exist_ok=True)
    fig.savefig(os.path.join(out, 'preview', name.replace('.pdf', '.png')), dpi=200)
    plt.close(fig)

LABEL = {'raw_local': 'Hermes', 'baseline': 'Hermes', 's1': 'DT-PP',
         's1_local_blosc': 'DT-PP', 'local_blosc': 'DT-PP',
         's2': 'DT-PC', 'cons_raw': 'DT-XC', 'local_raw': 'Hermes',
         's3': 'DT-CC', 's3_cons_blosc': 'DT-CC', 'cons_blosc': 'DT-CC',
         'auto': 'DT-AUTO', 'full': 'DT-AUTO', 'no_dag': 'DT-AUTO w/o DAG',
         'hcompress': 'HCompress'}


V2 = False  # --v2: use the re-runs after the 4 KiB padding / S1 score fixes (<exp>_v2)
V3 = False  # --v3: additionally let <exp>_v3 runs (final build, screened exclusive nodes) replace same-named runs


def resolve(root, exp):
    """Experiment directory pattern to read: with --v2, <exp>_v2 replaces an experiment and
    its older repetitions (<exp>_r*) when it exists."""
    if not V2:
        return exp
    if exp == 'e5async_r*':
        return 'e5async_v2'
    if exp.endswith('_r*'):
        return exp if not glob.glob(os.path.join(root, exp[:-3] + '_v2')) else None
    return exp + '_v2' if glob.glob(os.path.join(root, exp + '_v2')) else exp


def load(root, exp, allow_bad=False, newest=True):
    """Valid runs of one experiment: {run: [result dict, ...]} (repetitions merged by run name).
    With --v3, runs of <exp>_v3 replace the same-named runs of the resolved experiment.
    allow_bad keeps runs whose consumer saw a few failed reads (reported on the console)."""
    if V3 and newest and not exp.endswith('_r*') and not exp.endswith('_v3'):
        out = load(root, exp, allow_bad, newest=False)
        out.update(load(root, exp + '_v3', allow_bad))
        return out
    exp = resolve(root, exp) if not exp.endswith('_v3') else exp
    if exp is None:
        return {}
    out, skipped = {}, 0
    for e in sorted(glob.glob(os.path.join(root, exp))):
        for f in sorted(glob.glob(os.path.join(e, '*.json'))):
            with open(f) as fh:
                d = json.load(fh)
            m = d.get('makespan') or {}
            bad = (m.get('consumer') or {}).get('bad', 0) or m.get('errors', 0) or 0
            if bad and allow_bad and m.get('status', 'ok') == 'ok' and not d.get('rc') and m.get('total_ms'):
                print(f"{exp}/{d['run']}: kept with {bad} failed reads")
            elif m.get('status', 'ok') != 'ok' or bad or d.get('rc') or not m.get('total_ms'):
                skipped += 1
                continue
            out.setdefault(d['run'], []).append(d)
    if skipped:
        print(f'{exp}: skipped {skipped} invalid runs')
    return out


def reps(runs):
    """Merge repetitions named <cfg>_r<k> into {cfg: [result, ...]}."""
    out = {}
    for k, v in runs.items():
        out.setdefault(k.rsplit('_r', 1)[0] if '_r' in k and k.rsplit('_r', 1)[1].isdigit() else k, []).extend(v)
    return out


def ms(d):
    """Makespan in seconds."""
    return d['makespan']['total_ms'] / 1000.0


def stat(runs):
    """(mean, min, max, n) of makespans."""
    v = [ms(d) for d in runs]
    return statistics.mean(v), min(v), max(v), len(v)


def bar_err(ax, names, runs, color=C[0]):
    """Horizontal makespan bars with min-max whiskers and value labels."""
    ys = range(len(names))
    for y, n in zip(ys, names):
        mean, lo, hi, k = stat(runs[n])
        ax.barh(y, mean, height=0.62, color=color if n not in ('auto', 'full') else C[1])
        if k > 1:
            ax.plot([lo, hi], [y, y], color=INK, lw=0.8)
        ax.text(hi + 3, y, f'{mean:.0f} s' + (f' (n={k})' if k > 1 else ''), va='center',
                fontsize=7, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABEL.get(n, n) for n in names])
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)


def fig_prodcons(root, out):
    """Producer-consumer (CTE API, 40 GbE, idle consumer): makespan per placement/compression
    config, merged over repetitions, and the producer's compute/write split."""
    runs = {}
    for e in ('e5_40g_idle', 'e5_40g_idle_r*'):
        for k, v in load(root, e).items():
            runs.setdefault(k, []).extend(v)
    names = [n for n in ('raw_local', 's1', 's2', 's3', 'auto') if n in runs]
    if not names:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL_W, 1.6), gridspec_kw={'width_ratios': [1.25, 1]})
    bar_err(a, names, runs)
    a.set_xlabel('Makespan (s)')
    ys = range(len(names))
    comp = [statistics.mean(d['makespan']['producer']['compute_s'] for d in runs[n]) for n in names]
    wr = [statistics.mean(d['makespan']['producer']['write_s'] + (d['makespan']['producer'].get('flush_s') or 0)
                          for d in runs[n]) for n in names]
    b.barh(ys, comp, height=0.62, color=C[2], label='solver compute', edgecolor='white', lw=1)
    b.barh(ys, wr, left=comp, height=0.62, color=C[3], label='write (stored)', edgecolor='white', lw=1)
    b.set_yticks(list(ys)); b.set_yticklabels([]); b.invert_yaxis(); b.grid(axis='y', visible=False)
    b.set_xlabel('Producer time (s)')
    b.legend(loc='lower right', fontsize=7)
    fig.tight_layout()
    save(fig, out, 'prodcons_configs.pdf')
    print('prodcons_configs.pdf', {n: round(stat(runs[n])[0], 1) for n in names})


def fig_motivation_flip(root, out):
    """Motivation: the best FIXED policy per condition (E5 rows without DTSchedule's auto),
    normalised to the best fixed policy in each column."""
    fig_e5_grid(root, out, rows=['raw_local', 's1', 's2', 's3'], name='motivation_flip.pdf',
                height=1.55)


def fig_e5_grid(root, out, rows=None, name='e5_grid.pdf', height=1.9):
    """Scenario ablation heatmap: rows = config, cols = (network, consumer load);
    cell = makespan / best config in that column."""
    cols = [('40g', 'idle'), ('40g', 'cload'), ('40g', 'async')]
    rows = rows or ['raw_local', 's1', 's2', 's3', 'auto']
    data = {}
    for net, load_ in cols:
        if load_ == 'async':  # overlapped writes, idle consumer (Section "Where to compress")
            r = {}
            for k, v in load(root, 'e5async_r*').items():
                r.setdefault(k, []).extend(v)
            if V3:  # repetitions on the final build
                for k, v in reps(load(root, 'e5async_v3')).items():
                    r.setdefault(k, []).extend(v)
            if V3 and reps(load(root, 'abl_async_v3')).get('full'):
                r['auto'] = reps(load(root, 'abl_async_v3'))['full']
            if V3:  # Hermes with overlapped writes and one consumer = the fan-out study's k=1
                h = reps(load(root, 'fanout_v3', allow_bad=True)).get('hermes_k1')
                if h:
                    r['raw_local'] = h
        else:
            r = load(root, f'e5_{net}_{load_}')
            for k, v in load(root, f'e5_{net}_{load_}_r*').items():  # repetitions
                r.setdefault(k, []).extend(v)
        r = reps(r)
        if r.get('auto_wb'):  # blocking writes: DTSchedule with the writers-block hint
            r['auto'] = r['auto_wb']
        data[(net, load_)] = {n: stat(r[n])[0] for n in rows if n in r}
    present = [c for c in cols if data[c]]
    if not present:
        return
    import numpy as np
    m = np.full((len(rows), len(present)), np.nan)
    for j, c in enumerate(present):
        best = min(data[c].values())
        for i, n in enumerate(rows):
            if n in data[c]:
                m[i, j] = data[c][n] / best
    fig, ax = plt.subplots(figsize=(COL_W, height))
    im = ax.imshow(np.nan_to_num(m, nan=1.0), cmap='Blues', vmin=1.0, vmax=max(2.2, np.nanmax(m)), aspect='auto')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('makespan / best in column', fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    for i in range(len(rows)):
        for j in range(len(present)):
            if not np.isnan(m[i, j]):
                s = data[present[j]][rows[i]]
                ax.text(j, i, f'{m[i, j]:.2f}x\n{s:.0f}s', ha='center', va='center', fontsize=6.5,
                        color='white' if m[i, j] > 1.6 else INK, fontweight='bold' if m[i, j] == 1.0 else None)
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([{'idle': 'idle', 'cload': 'loaded', 'async': 'overlapped'}[l]
                        for n, l in present], fontsize=6.5)
    ax.set_xlabel('consumer idle / loaded; writes overlapped', fontsize=6.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([LABEL[n] for n in rows], fontsize=7)
    ax.grid(False)
    fig.tight_layout()
    save(fig, out, name)
    print(name, {f'{n}/{l}': data[(n, l)] for n, l in present})


def fig_ablation(root, out, exp='e6', order=('baseline', 'no_placement', 'fixed_ccm', 'no_load', 'no_dag', 'full'),
                 name='e6_ablation.pdf', labels=None):
    """Bar chart of makespan for a set of named runs of one experiment."""
    runs = load(root, exp)
    names = [n for n in order if n in runs]
    if not names:
        return
    lab = labels or {'baseline': 'Hermes', 'no_placement': 'w/o placement (local)',
                     'fixed_ccm': 'zstd as the only codec', 'no_load': 'w/o load awareness',
                     'no_dag': 'w/o workflow knowledge', 'full': 'DT-AUTO'}
    fig, ax = plt.subplots(figsize=(COL_W, 0.32 * len(names) + 0.5))
    bar_err(ax, names, runs)
    ax.set_yticklabels([lab.get(n, LABEL.get(n, n)) for n in names], fontsize=7)
    ax.set_xlabel('Makespan (s)')
    fig.tight_layout(); save(fig, out, name)
    print(name, {n: round(stat(runs[n])[0], 1) for n in names})


def fig_workflows(root, out):
    """WfCommons recipes over the CTE API: makespan normalised to the local/raw baseline."""
    recipes = ['montage', 'seismology', 'genome', 'epigenomics', 'blast', 'blastio', 'bwa', 'cycles', 'soykb', 'srasearch', 'rnaseq']
    cfgs = ['baseline', 'hcompress', 's1_local_blosc', 'cons_raw', 's3_cons_blosc', 'full']
    res = {r: (load(root, 'wf_blastio_v4') if r == 'blastio' else load(root, f'wf_{r}')) for r in recipes}
    recipes = [r for r in recipes if all(res[r].get(c) for c in cfgs)]
    # Left to right: by how much DT-AUTO beats the next best policy on that workflow.
    def margin(r):
        others = min(stat(res[r][c])[0] for c in cfgs if c != 'full')
        return others / stat(res[r]['full'])[0]
    recipes.sort(key=margin, reverse=True)
    print('workflow order (next best / DT-AUTO):', [(r, round(margin(r), 2)) for r in recipes])
    if not recipes:
        return
    fig, ax = plt.subplots(figsize=(2 * COL_W, 1.3))
    w = 0.8 / len(cfgs)
    for k, cfg in enumerate(cfgs):
        xs, ys = [], []
        for i, r in enumerate(recipes):
            if cfg in res[r]:
                xs.append(i + (k - (len(cfgs) - 1) / 2) * w)
                ys.append(stat(res[r][cfg])[0] / stat(res[r]['baseline'])[0])
        col = {'baseline': C[0], 'hcompress': C[2], 's1_local_blosc': C[6], 'cons_raw': C[3],
               's3_cons_blosc': C[4], 'full': C[1]}.get(cfg, C[k])
        ax.bar(xs, ys, width=w * 0.92, color=col, label=LABEL.get(cfg, cfg))
    ax.axhline(1.0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(recipes))); ax.set_xticklabels(recipes)
    ax.set_ylabel('Makespan / Hermes'); ax.set_yticks([0, 0.5, 1.0, 1.5])
    names = {'genome': '1000Gen.', 'epigenomics': 'Epigen.', 'blast': 'BLAST', 'blastio': 'BLAST (I/O)', 'bwa': 'BWA', 'soykb': 'SoyKB',
             'srasearch': 'SRASearch', 'rnaseq': 'RNA-Seq'}
    ax.set_xticklabels([names.get(r, r.capitalize()) for r in recipes], fontsize=6)
    ax.legend(ncol=6, fontsize=6.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)
    ax.grid(axis='x', visible=False)
    fig.tight_layout(); save(fig, out, 'workflows.pdf')
    print('workflows.pdf', {r: {c: round(stat(res[r][c])[0], 1) for c in cfgs if c in res[r]} for r in recipes})


def fig_motivation(root, out):
    """Interference microbenchmark: a fixed-work MPI job on all 40 hw threads alone vs with
    4 compression threads for half its runtime (left), and the codecs' own throughput idle vs
    under the MPI job (right). Source: micro/interference_4min.jsonl."""
    path = os.path.join(root, 'micro', 'interference_4min.jsonl')
    if not os.path.exists(path):
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.45))
    names = [r['codec'] for r in rows]
    xs = range(len(rows))
    w = 0.38
    alone = [statistics.mean(r['alone']) for r in rows]
    withc = [statistics.mean(r['with']) for r in rows]
    a.bar([x - w / 2 for x in xs], alone, w, color=C[0], label='MPI alone')
    a.bar([x + w / 2 for x in xs], withc, w, color=C[1], label='+4 compress thr.')
    for x, (u, v) in enumerate(zip(alone, withc)):
        a.text(x + w / 2, v + 4, f'+{100 * (v / u - 1):.0f}%', ha='center', fontsize=6.5, color=INK)
    a.set_xticks(list(xs)); a.set_xticklabels(names); a.set_ylabel('MPI job time (s)')
    a.set_ylim(0, max(withc) * 1.25); a.grid(axis='x', visible=False)
    a.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
    ca = [statistics.mean(r['comp_alone']) for r in rows]
    cw = [statistics.mean(r['comp_with']) for r in rows]
    b.bar([x - w / 2 for x in xs], ca, w, color=C[0], label='idle node')
    b.bar([x + w / 2 for x in xs], cw, w, color=C[1], label='under MPI')
    for x, (u, v) in enumerate(zip(ca, cw)):
        b.text(x + w / 2, v + 8, f'{100 * (v / u - 1):.0f}%', ha='center', fontsize=6.5, color=INK)
    b.set_xticks(list(xs)); b.set_xticklabels(names); b.set_ylabel('Codec MB/s (4 thr.)')
    b.set_ylim(0, max(ca) * 1.25); b.grid(axis='x', visible=False)
    b.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
    fig.tight_layout(); save(fig, out, 'motivation_interference.pdf')
    print('motivation_interference.pdf', list(zip(names, alone, withc, ca, cw)))


def tab_codecs(root, out):
    """LaTeX table: per-codec throughput and ratio on montage-like data at several chunk
    sizes (one core). Source: micro/codec_sizes_c08.txt (last row per codec/preset/size)."""
    path = os.path.join(root, 'micro', 'codec_sizes_c08.txt')
    if not os.path.exists(path):
        return
    cell = {}
    for line in open(path):
        kv = dict(t.split('=') for t in line.split())
        cell[(kv['codec'], kv['preset'], int(kv['chunk_kb']))] = (float(kv['mb_s']), float(kv['ratio']))
    sizes = [4, 64, 1024]
    order = [('blosc2', '0'), ('lz4', '0'), ('zstd', '0'), ('zstd', '1'), ('snappy', '1'), ('brotli', '0'),
             ('zlib', '0'), ('lzma', '0')]
    pname = {'0': 'fast', '1': 'bal.'}
    lines = [r'\begin{tabular}{l' + 'rr' * len(sizes) + '}', r'\toprule',
             'Codec & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{s if s < 1024 else 1}\,{"KiB" if s < 1024 else "MiB"}}}' for s in sizes) + r' \\',
             ' & ' + ' & '.join(['MB/s & ratio'] * len(sizes)) + r' \\', r'\midrule']
    for c, p in order:
        if not all((c, p, s) in cell for s in sizes):
            continue
        vals = ' & '.join(f'{cell[(c, p, s)][0]:.0f} & {cell[(c, p, s)][1]:.2f}' for s in sizes)
        lines.append(f'{c} ({pname.get(p, p)}) & {vals} ' + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    with open(os.path.join(out, 'tab_codecs.tex'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('tab_codecs.tex', len(lines) - 7, 'codecs')


def fig_e10(root, out):
    """Load-response timeline (E10): consumer loaded 60-120 s after launch. Rows: producer
    and consumer CPU from dtschedule's load samples; share of chunks per scenario and share
    compressed, per 5 s bin, from the decision trace."""
    import pandas as pd
    tr = sorted(glob.glob(os.path.join(root, 'e10', 'traces', 'timeline', 'dtschedule_trace.0.csv')))  # writer's decisions only (the consumer logs the chunks it stores again)
    frames = [pd.read_csv(f) for f in tr]
    frames = [f for f in frames if len(f)]
    if not frames:
        return
    d = pd.concat(frames)
    d['t'] = (d.ts_ms - d.ts_ms.min()) / 1000.0
    d['bin'] = (d.t // 5 * 5).astype(int)
    g = d.groupby('bin')
    cpu = g[['producer_cpu', 'consumer_cpu']].median()
    sc = d.groupby(['bin', 'chosen_scenario']).size().unstack(fill_value=0)
    sc = sc.div(sc.sum(axis=1), axis=0)
    z = d.chosen_lib != 'raw'
    at_c = (z & (d.chosen_scenario == 3)).groupby(d['bin']).mean() * 100
    at_p = (z & (d.chosen_scenario != 3)).groupby(d['bin']).mean() * 100
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 2.5), sharex=True,
                               gridspec_kw={'height_ratios': [1, 1.15]})
    # Shade the injected-load window: bins where the consumer's CPU sample
    # stays above 70% (the 20-rank MPI job; the consumer alone stays below).
    hot = cpu.index[cpu.consumer_cpu >= 70]
    for ax in (a, b):
        if len(hot):
            ax.axvspan(hot.min() - 2.5, hot.max() + 2.5, color='#d9d8d4', alpha=0.5, lw=0, zorder=0)
        ax.grid(axis='x', visible=False)
        ax.set_ylim(0, 105); ax.set_yticks([0, 50, 100])
    a.plot(cpu.index, cpu.producer_cpu, color=C[0], lw=1.5)
    a.plot(cpu.index, cpu.consumer_cpu, color=C[1], lw=1.5)
    a.set_ylabel('CPU (%)')
    # Direct labels instead of a legend box.
    a.text(cpu.index.min(), 104, 'producer', color=C[0], fontsize=6.5, va='bottom')
    a.text(cpu.index.min() + 22, 25, 'consumer', color=C[1], fontsize=6.5)
    if len(hot):
        a.text((hot.min() + hot.max()) / 2, 104, 'consumer loaded', ha='center', va='bottom',
               fontsize=6.5, color=INK2)
    b.plot(at_c.index, at_c, color=C[2], lw=1.5, marker='o', ms=2.5, label='at the consumer (DT-CC)')
    b.plot(at_p.index, at_p, color=C[4], lw=1.5, marker='s', ms=2.5, label='at the producer (DT-PP/PC)')
    b.set_ylabel('chunks\ncompressed (%)'); b.set_xlabel('time since first write (s)')
    b.legend(fontsize=6.5, ncol=1, loc='upper left', handlelength=1.4)
    fig.tight_layout(); save(fig, out, 'e10_timeline.pdf')
    print('e10_timeline.pdf bins', len(cpu))


def fig_e8_e9(root, out):
    """Predictor ablation (E8: qtable/ema/oracle) and ratio-noise sensitivity (E9)."""
    # e8_r2 ran in a quiet window; the first e8 pass coincided with a slow
    # period on the producer node (both predictors ~2x slower), see the text.
    e8, e9 = load(root, 'e8_r2') or load(root, 'e8'), load(root, 'e9')
    if not e8 and not e9:
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.4))
    names = [n for n in ('qtable', 'ema') if n in e8]
    for i, n in enumerate(names):
        a.bar(i, stat(e8[n])[0], color=C[i], width=0.6)
        a.text(i, stat(e8[n])[0] + 3, f'{stat(e8[n])[0]:.0f}', ha='center', fontsize=6.5)
    a.set_xticks(range(len(names))); a.set_xticklabels(['Q-table', 'EMA'][:len(names)])
    a.set_ylabel('Makespan (s)'); a.grid(axis='x', visible=False)
    sig = sorted((float(n[5:]), n) for n in e9 if n.startswith('sigma'))
    base = e8.get('qtable')
    xs = ([0.0] if base else []) + [x for x, _ in sig]
    ys = ([stat(base)[0]] if base else []) + [stat(e9[n])[0] for _, n in sig]
    b.plot(xs, ys, marker='o', color=C[0], lw=1.5, ms=4)
    b.set_xlabel('ratio noise $\\sigma$'); b.set_ylim(0, max(ys + [1]) * 1.2 if ys else 1)
    fig.tight_layout(); save(fig, out, 'e8_e9.pdf')
    print('e8_e9.pdf', {n: round(stat(e8[n])[0], 1) for n in names}, list(zip(xs, [round(y, 1) for y in ys])))


def fig_e7(root, out):
    """QoS (E7) on the float32 solver field: makespan and bytes stored per QoS setting
    (objective x max_error). Stored bytes are summed from the decision trace
    (size / observed ratio for compressed chunks, size for raw ones)."""
    import pandas as pd
    r = load(root, 'e7f')
    order = [('err0', 'perf, lossless'), ('err0.001', 'perf, 1e-3'), ('err0.05', 'perf, 5e-2'),
             ('ratio_err0.001', 'ratio, 1e-3'), ('ratio_err0.05', 'ratio, 5e-2')]
    order = [(n, l) for n, l in order if n in r]
    if not order:
        return
    gb = []
    for n, _ in order:
        tot = 0.0
        frames = [pd.read_csv(f) for f in
                  glob.glob(os.path.join(root, 'e7f', 'traces', n, 'dtschedule_trace.[0-9].csv'))]
        frames = [f for f in frames if len(f)]
        # A chunk stored on the consumer also appears in the consumer's trace
        # (without a ratio): keep one row per (file, chunk), the writer's.
        d_all = pd.concat(frames) if frames else pd.DataFrame()
        if len(d_all):
            d_all = (d_all.assign(_has=d_all.obs_ratio.notna())
                     .sort_values('_has', ascending=False)
                     .drop_duplicates(['tag', 'blob']))
        for d in ([d_all] if len(d_all) else []):
            comp = d.chosen_lib != 'raw'
            # Chunks compressed on the consumer (S3) have no observed ratio in the
            # writer's trace: use the run's median for the same codec and preset.
            med = d[comp & (d.obs_ratio > 0)].groupby('chosen_lib').obs_ratio.median()
            fill = d.chosen_lib.map(med).fillna(1.0)
            ratio = d.obs_ratio.where(d.obs_ratio > 0, fill).where(comp, 1.0).fillna(1.0)
            tot += float((d['size'] / ratio).sum())
        gb.append(tot / 1e9)
    ms_ = [stat(r[n])[0] for n, _ in order]
    fig, (a, b) = plt.subplots(1, 2, figsize=(COL_W, 1.6), gridspec_kw={'width_ratios': [1.15, 1]})
    ys = range(len(order))
    cols = [C[0] if n.startswith('err') else C[1] for n, _ in order]
    a.barh(ys, ms_, color=cols, height=0.6); a.set_yticks(list(ys)); a.set_yticklabels([l for _, l in order], fontsize=6.5)
    a.invert_yaxis(); a.set_xlabel('Makespan (s)'); a.grid(axis='y', visible=False)
    b.barh(ys, gb, color=cols, height=0.6); b.set_yticks(list(ys)); b.set_yticklabels([]); b.invert_yaxis()
    b.set_xscale('log'); b.set_xlabel('Stored (GB, log)'); b.grid(axis='y', visible=False)
    from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter
    b.xaxis.set_major_locator(LogLocator(base=10)); b.xaxis.set_minor_formatter(NullFormatter())
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    b.set_xlim(min(gb) / 2, max(gb) * 4)
    for y, v in zip(ys, gb):
        b.text(v * 1.15, y, f'{v:.2g}', va='center', fontsize=6.5)
    fig.tight_layout(); save(fig, out, 'e7_qos.pdf')
    print('e7_qos.pdf', list(zip([l for _, l in order], [round(x, 1) for x in ms_], [round(x, 3) for x in gb])))

def fig_async(root, out):
    """Where to compress when the producer computes: each fixed policy and DTSchedule with
    blocking writes (data moves while the solver waits) and with overlapped writes (2 files in
    flight, data moves while the solver computes). Bars: makespan; markers: solver compute."""
    sync, asy = {}, {}
    for e in ('e5_40g_idle', 'e5_40g_idle_r*'):
        for k, v in load(root, e).items():
            sync.setdefault(k, []).extend(v)
    for k, v in load(root, 'e5async_r*').items():
        asy.setdefault(k, []).extend(v)
    if V3 and reps(load(root, 'abl_async_v3')).get('full'):
        asy['auto'] = reps(load(root, 'abl_async_v3'))['full']
    names = [n for n in ('s1', 's2', 's3', 'auto') if n in sync and n in asy]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    w = 0.38
    for k, (lab, dd, col) in enumerate((('blocking writes', sync, C[0]), ('overlapped writes', asy, C[1]))):
        xs = [i + (k - 0.5) * w for i in range(len(names))]
        ys = [stat(dd[n])[0] for n in names]
        comp = [statistics.mean(d['makespan']['producer']['compute_s'] for d in dd[n]) for n in names]
        ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        ax.scatter(xs, comp, marker='_', s=120, color=INK, zorder=3, label='solver compute' if k else None)
        for x, y in zip(xs, ys):
            ax.text(x, y + 4, f'{y:.0f}', ha='center', fontsize=6.5)
    short = {'s1': 'DT-PP', 's2': 'DT-PC', 's3': 'DT-CC', 'auto': 'DT-AUTO'}
    ax.set_xticks(range(len(names))); ax.set_xticklabels([short[n] for n in names], fontsize=6.5)
    ax.set_ylabel('Makespan (s)'); ax.grid(axis='x', visible=False)
    ax.set_ylim(0, max(stat(asy[n])[0] for n in names) * 1.12)
    ax.legend(fontsize=6, ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, 'async.pdf')
    print('async.pdf', {n: (round(stat(sync[n])[0], 1), round(stat(asy[n])[0], 1), stat(asy[n])[3]) for n in names})


def fig_scaling(root, out, exp='scale'):
    """Baselines across producer scale: Hermes (the runtime alone: raw, writer-local, tiered),
    HCompress (codec chosen per tier from chunk size only, writer-local, no load or workflow
    signal) and DTSchedule, with 10/20/40 producer ranks (5/10/20 consumer ranks) on the same
    two nodes. Each rank writes 32 MB per step, so data grows with the rank count."""
    runs = load(root, exp)
    sysn = [('hermes', 'Hermes', C[0]), ('hcompress', 'HCompress', C[2]),
            ('dtsched', 'DT-AUTO', C[1])]
    ranks = sorted({int(k.rsplit('_p', 1)[1]) for k in runs if '_p' in k})
    if not ranks:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 1.75))
    w = 0.27
    for k, (key, lab, col) in enumerate(sysn):
        xs, ys = [], []
        for i, r in enumerate(ranks):
            n = f'{key}_p{r}'
            if n not in runs:
                continue
            xs.append(i + (k - 1) * w); ys.append(stat(runs[n])[0])
        ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 3, f'{y:.0f}', ha='center', fontsize=6)
    gb = {r: runs[f'hermes_p{r}'][0]['makespan']['producer']['step_mb'] * runs[f'hermes_p{r}'][0]['makespan']['producer']['steps'] / 1000
          for r in ranks if f'hermes_p{r}' in runs}
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels([f'{r} ranks' + (f'\n{gb[r]:.1f} GB' if r in gb else '') for r in ranks], fontsize=6.5)
    ax.set_ylabel('Makespan (s)'); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6, ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, 'scaling.pdf')
    print('scaling.pdf', {n: round(stat(v)[0], 1) for n, v in sorted(runs.items())})


def fig_stacked(root, out):
    """Stacked ablation: DTSchedule restricted to the writer's node (local), + workflow
    knowledge (consumer placement and compression), + load awareness (full), in the
    compute-heavy cell (overlapped writes) and with a loaded consumer. Bars: mean makespan,
    whiskers: min-max; markers: the solver's compute time."""
    cells = [('abl_async_v3', 'Producer computing\n(overlapped writes)'), ('abl_cload_v3', 'Consumer loaded\n(blocking writes)')]
    steps = [('local', 'DT-AUTO, local only', C[0]), ('wf', '+ workflow knowledge', C[2]),
             ('full', '+ load awareness (full)', C[1])]
    data = {e: reps(load(root, e)) for e, _ in cells}
    for e in data:  # blocking writes: the full system carries the writers_block hint
        if data[e].get('full_wb'):
            data[e]['full'] = data[e].pop('full_wb')
    cells = [(e, l) for e, l in cells if data[e]]
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    w = 0.27
    for k, (key, lab, col) in enumerate(steps):
        for i, (e, _) in enumerate(cells):
            if key not in data[e]:
                continue
            mean, lo, hi, n = stat(data[e][key])
            x = i + (k - 1) * w
            ax.bar(x, mean, w * 0.92, color=col, label=lab if i == 0 else None)
            if n > 1:
                ax.plot([x, x], [lo, hi], color=INK, lw=0.8)
            comp = statistics.mean(d['makespan']['producer']['compute_s'] for d in data[e][key])
            ax.scatter([x], [comp], marker='_', s=90, color=INK, zorder=3,
                       label='solver compute' if (i == 0 and k == 0) else None)
            ax.text(x, 12, f'{mean:.0f}', ha='center', fontsize=6.5, color='white', fontweight='bold')
    ax.set_xticks(range(len(cells))); ax.set_xticklabels([l for _, l in cells], fontsize=6.5)
    ax.set_ylabel('Makespan (s)'); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, 'stacked_ablation.pdf')
    print('stacked_ablation.pdf', {e: {k: (round(stat(v)[0], 1), stat(v)[3]) for k, v in data[e].items()} for e, _ in cells})


def fig_fanout(root, out, exp='fanout_v3'):
    """Hermes, HCompress and DTSchedule with one 40-rank producer node and k consumer nodes
    (20 ranks each), each consumer reading every file (40 GbE, overlapped writes); DTSchedule
    with fan-out replication as a fourth bar. Bars: mean; whiskers: min-max over runs."""
    runs = reps(load(root, exp, allow_bad=True))
    for k, v in reps(load(root, 'fanout_rep_v4')).items():  # rep_on_k<k>
        runs[k.replace('rep_on_', 'rep_')] = v
    sysn = [('hermes', 'Hermes', C[0]), ('hcompress', 'HCompress', C[2]),
            ('dtsched', 'DT-AUTO', C[1]), ('rep', 'DT-AUTO + replication', C[3])]
    ks = sorted({int(k.rsplit('_k', 1)[1]) for k in runs if '_k' in k})
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 1.8))
    w = 0.2
    for j, (key, lab, col) in enumerate(sysn):
        for i, k in enumerate(ks):
            n = f'{key}_k{k}'
            if n not in runs:
                continue
            mean, lo, hi, cnt = stat(runs[n])
            x = i + (j - 1.5) * w
            ax.bar(x, mean, w * 0.92, color=col, label=lab if i == next(
                ii for ii, kk in enumerate(ks) if f'{key}_k{kk}' in runs) else None)
            if cnt > 1:
                ax.plot([x, x], [lo, hi], color=INK, lw=0.8)
            ax.text(x, hi + 8, f'{mean:.0f}', ha='center', fontsize=5.5)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f'{k} consumer' + ('s' if k > 1 else '') for k in ks], fontsize=6.5)
    ax.set_ylabel('Makespan (s)'); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6, ncol=2, loc='lower center', bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, 'fanout.pdf')
    print('fanout.pdf', {n: (round(stat(v)[0], 1), round(stat(v)[1]), round(stat(v)[2]), stat(v)[3]) for n, v in sorted(runs.items())})


def fig_fanin(root, out, exp='fanin_seis_x4_v4'):
    """Fan-in: Seismology's gather (4x data, 1/4 compute) on 2, 4 and 8 nodes, round-robin
    tasks, under DTSchedule with the DAG, without consumer knowledge, writer-local, hash
    placement and Hermes."""
    runs = load(root, exp)
    for n in (2, 4, 8):  # final build: fan-in degree counts the reducer itself
        if runs.get(f'full2_n{n}'):
            runs[f'full_n{n}'] = runs.pop(f'full2_n{n}')
    runs.pop('fullfix_n8', None)
    arms = [('hermes', 'Hermes', C[0]), ('owner', 'hash placement', C[6]), ('local', 'writer-local', C[3]),
            ('none', 'DT-AUTO w/o DAG', C[2]), ('full', 'DT-AUTO', C[1])]
    ns = sorted({int(k.rsplit('_n', 1)[1]) for k in runs if '_n' in k})
    if not ns:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 1.7))
    w = 0.16
    for j, (key, lab, col) in enumerate(arms):
        xs, ys = [], []
        for i, n in enumerate(ns):
            if f'{key}_n{n}' in runs:
                xs.append(i + (j - 2) * w); ys.append(stat(runs[f'{key}_n{n}'])[0])
        ax.bar(xs, ys, w * 0.92, color=col, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x, y + 3, f'{y:.0f}', ha='center', fontsize=5.5)
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f'{n} nodes' for n in ns], fontsize=6.5)
    ax.set_ylabel('Makespan (s)'); ax.grid(axis='x', visible=False)
    ax.legend(fontsize=6, ncol=3, loc='lower center', bbox_to_anchor=(0.45, 1.0), handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, out, 'fanin.pdf')
    print('fanin.pdf', {k: round(stat(v)[0], 1) for k, v in sorted(runs.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=os.path.expanduser('~/jarvis-runs/dtschedule-results'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--v2', action='store_true', help='use the <exp>_v2 re-runs where present')
    ap.add_argument('--v3', action='store_true', help='also let <exp>_v3 runs replace same-named runs')
    ap.add_argument('names', nargs='*')
    a = ap.parse_args()
    global V2, V3
    V2, V3 = a.v2 or a.v3, a.v3
    os.makedirs(a.out, exist_ok=True)
    figs = {'motivation': fig_motivation, 'flip': fig_motivation_flip, 'codecs': tab_codecs, 'prodcons': fig_prodcons, 'e5': fig_e5_grid, 'e6': fig_ablation, 'workflows': fig_workflows, 'e10': fig_e10, 'e8e9': fig_e8_e9, 'e7': fig_e7, 'async': fig_async, 'scaling': fig_scaling, 'stacked': fig_stacked, 'fanout': fig_fanout, 'fanin': fig_fanin}
    for n in (a.names or figs):
        figs[n](a.root, a.out)


if __name__ == '__main__':
    main()
