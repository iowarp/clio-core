#!/usr/bin/env python3
"""
Empirically find, per workload, the MPI configuration whose PEAK VRAM is
~6 GB on the target GPU -- the anchor problem size for the memory-pressure
pipelines. Prints a table; does not guess: every row is a real run with
nvidia-smi sampled at 50 ms.

Usage:  PATH=<build>/bin:$PATH python3 calibrate_6gb.py [--target-mb 6144]

The candidate ladders below were seeded from the workload_understanding
sweeps' measured scalings (delta ~= 138 MB CUDA context + the workload's
analytic footprint) and bracket the target; the run PICKS the closest.
"""
import argparse
import subprocess
import threading
import time

CANDIDATES = {
    # workload: (argv-template, [candidate values])
    'kmeans': ('clio_kmeans_mpi_bench --iters 2 --data-mb {v}',
               [5600, 5800, 6000]),
    'weights': ('clio_weights_mpi_bench --passes 1 --data-mb {v}',
                [5600, 5800, 6000]),
    'grayscott': ('clio_grayscott_mpi_bench --steps 4 --data-mb {v}',
                  [5600, 5800, 6000]),
    'gmx': ('clio_gmx_mpi_bench --k {v} --atoms 200000',
            [880, 900, 920]),
    'lbann': ('clio_lbann_mpi_bench --in 4096 --hidden {v} --out 64 '
              '--batch 256 --steps 2',
              [131072, 163840, 196608]),
    'lammps_md': ('clio_lammps_md_mpi_bench --lattice {v} --steps 10 --md',
                  [100, 110, 120]),
}


def sample_peak(proc):
    peak = [0]
    stop = threading.Event()

    def run():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=10)
                v = int(out.stdout.strip().splitlines()[0])
                peak[0] = max(peak[0], v)
            except Exception:
                pass
            stop.wait(0.05)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    proc.wait()
    stop.set()
    t.join(timeout=5)
    return peak[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target-mb', type=int, default=6144)
    ap.add_argument('--timeout', type=int, default=600)
    ap.add_argument('--only', help='single workload')
    args = ap.parse_args()

    chosen = {}
    for wl, (tpl, values) in CANDIDATES.items():
        if args.only and wl != args.only:
            continue
        rows = []
        for v in values:
            cmd = 'timeout -k 30 %d mpirun -n 1 --oversubscribe %s' \
                  % (args.timeout, tpl.format(v=v))
            print('== %s  %s' % (wl, cmd), flush=True)
            proc = subprocess.Popen(cmd, shell=True,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True)
            peak = sample_peak(proc)
            out = proc.stdout.read()
            ok = ('ALL GATES PASS' in out) or ('GATE: PASS' in out)
            print('   peak=%d MB  gates=%s' % (peak, ok), flush=True)
            rows.append((v, peak, ok))
        good = [(v, p) for v, p, ok in rows if ok and p > 0]
        if good:
            v, p = min(good, key=lambda r: abs(r[1] - args.target_mb))
            chosen[wl] = (v, p)
    print('\n=== chosen (target %d MB) ===' % args.target_mb)
    for wl, (v, p) in chosen.items():
        print('%-12s knob=%-8s peak=%d MB (%.1f%% of target)'
              % (wl, v, p, 100.0 * p / args.target_mb))


if __name__ == '__main__':
    main()
