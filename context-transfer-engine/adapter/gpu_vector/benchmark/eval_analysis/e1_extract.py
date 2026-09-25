#!/usr/bin/env python3
"""Persist every E1 number to CSV: one row per (nodes, workload, substrate,
rank) with total/comm/compute ms, plus Eternia's fetch/hold/flush shares and
host-exchange ms, plus lbann's per-phase ms; and a per-cell summary
(slowest rank, mean, and the time per iteration).

Reads the ladder logs (scale_<N>_<group>.log) and reruns (rerun_*), which
override. Writes e1_ranks.csv and e1_cells.csv next to this script.
"""
import csv, glob, os, re

D = os.path.dirname(os.path.abspath(__file__))
SUB = {'MPI': 'mpi', 'ONECCL': 'ccl', 'ISHMEM': 'ishmem'}
F = r'([\d.]+)'


def cells_of(txt):
    """Split a rung log into its cells: {(wl, sub): text}."""
    out = {}
    parts = re.split(r'^--- (\w+?)_b(\d+)_(\w+): ', txt, flags=re.M)
    for i in range(1, len(parts), 4):
        wl, blocks, sub, body = parts[i], parts[i + 1], parts[i + 2], parts[i + 3]
        out[(wl, sub)] = (int(blocks), body)
    return out


def main():
    logs = sorted(glob.glob(os.path.join(D, 'scale_*_*.log')))
    logs += sorted(glob.glob(os.path.join(D, 'rerun_*_*.log')))
    ranks, cells = {}, {}
    for log in logs:
        n = int(re.search(r'_(\d+)_', os.path.basename(log)).group(1))
        txt = open(log, errors='replace').read()
        for (wl, sub), (blocks, body) in cells_of(txt).items():
            res = re.search(r'^RESULT e1/\S+: (\S+)', body, re.M)
            status = res.group(1) if res else 'NONE'
            rows = {}
            if sub != 'eternia':
                for m in re.finditer(r'^\s+COMM \w+ (\w+): rank (\d+) comm_ms=%s of %s ms'
                                     % (F, F), body, re.M):
                    r = int(m.group(2)); c, t = float(m.group(3)), float(m.group(4))
                    rows[r] = dict(total_ms=t, comm_ms=c, compute_ms=t - c)
                it = re.search(r'(?:iters|steps|passes)=(\d+)', body)
            else:
                # Eternia prints no rank id on its COMM line; they are echoed
                # in rank order (one per rank per repeat -- keep the last).
                gen = [m for m in re.finditer(
                    r'^\s+COMM \w+: comm=%s%% \(fetch %s%% hold %s%% flush %s%% of GPU '
                    r'busy; host xchg %s ms\) \| est comm_ms=%s compute_ms=%s of %s ms'
                    % ((F,) * 8), body, re.M)]
                gs = [m for m in re.finditer(
                    r'^\s+COMM grayscott two-phase: comm_ms=%s .*? compute_ms=%s .*? of %s ms'
                    % (F, F, F), body, re.M)]
                seq = []
                for m in gen:
                    seq.append(dict(total_ms=float(m.group(8)), comm_ms=float(m.group(6)),
                                    compute_ms=float(m.group(7)),
                                    fetch_pct=float(m.group(2)), hold_pct=float(m.group(3)),
                                    flush_pct=float(m.group(4)),
                                    host_xchg_ms=float(m.group(5))))
                # Older builds (the 4- and 8-node rungs): no host-exchange
                # field, so comm_ms is GPU Fetch/Hold/Flush only and the
                # host-side exchange is NOT counted -- flagged undercounted.
                for m in re.finditer(
                        r'^\s+COMM \w+: comm=%s%% \(fetch %s%% hold %s%% flush %s%%\) '
                        r'compute=[\d.]+%% of GPU busy \| est comm_ms=%s compute_ms=%s of %s ms'
                        % ((F,) * 7), body, re.M):
                    seq.append(dict(total_ms=float(m.group(7)), comm_ms=float(m.group(5)),
                                    compute_ms=float(m.group(6)),
                                    fetch_pct=float(m.group(2)), hold_pct=float(m.group(3)),
                                    flush_pct=float(m.group(4)), undercounted=1))
                for m in gs:
                    t = float(m.group(3)); c = float(m.group(1))
                    seq.append(dict(total_ms=t, comm_ms=c, compute_ms=float(m.group(2))))
                per = len(seq) // n if n and len(seq) >= n else 1
                for r in range(min(n, len(seq))):
                    rows[r] = seq[r * per + per - 1] if per > 1 else seq[r]
                it = re.search(r'(?:--iters|--steps|--repeat) (\d+)', body)
            iters = int(it.group(1)) if it else None
            ph = re.search(r'phases per step \(ms\): fwd1=%s fwd2=%s bwd1=%s upd2=%s upd1=%s'
                           % ((F,) * 5), body)
            for r, row in rows.items():
                row.update(nodes=n, workload=wl, substrate=sub, rank=r, blocks=blocks,
                           iters=iters, log=os.path.basename(log))
                if ph and r == 0:
                    for k, v in zip(('fwd1', 'fwd2', 'bwd1', 'upd2', 'upd1'), ph.groups()):
                        row['phase_' + k + '_ms'] = float(v)
                ranks[(n, wl, sub, r)] = row
            if rows:
                tot = [x['total_ms'] for x in rows.values()]
                com = [x['comm_ms'] for x in rows.values()]
                cells[(n, wl, sub)] = dict(
                    nodes=n, workload=wl, substrate=sub, blocks=blocks, iters=iters,
                    status=status, ranks=len(rows), total_ms_max=max(tot),
                    ms_per_iter=(max(tot) / iters) if iters else None,
                    comm_ms_max=max(com), comm_ms_mean=sum(com) / len(com),
                    compute_ms_mean=sum(t - c for t, c in zip(tot, com)) / len(com),
                    comm_pct_mean=100.0 * sum(com) / sum(tot),
                    comm_undercounted=int(any(x.get('undercounted') for x in rows.values())),
                    log=os.path.basename(log))
    rk = sorted(ranks.values(), key=lambda x: (x['nodes'], x['workload'], x['substrate'], x['rank']))
    keys = sorted({k for r in rk for k in r}, key=lambda k: (
        ['nodes', 'workload', 'substrate', 'rank', 'blocks', 'iters', 'total_ms', 'comm_ms',
         'compute_ms'] + [k]).index(k))
    with open(os.path.join(D, 'e1_ranks.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rk)
    cl = sorted(cells.values(), key=lambda x: (x['nodes'], x['workload'], x['substrate']))
    with open(os.path.join(D, 'e1_cells.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, list(cl[0].keys())); w.writeheader(); w.writerows(cl)
    print('e1_ranks.csv rows', len(rk), '| e1_cells.csv rows', len(cl))


if __name__ == '__main__':
    main()
