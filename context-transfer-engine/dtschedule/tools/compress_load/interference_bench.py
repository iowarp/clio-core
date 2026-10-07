#!/usr/bin/env python3
"""
Interference microbenchmark: how much does background compression slow a
CPU-saturating MPI job (and how fast is compression under that load)?

On one node: an MPI job (dtschedule_mpi_stress, fixed work, one rank per
hardware thread) runs alone to get T_alone; then the same job runs while
dtschedule_compress_load compresses for 0.5 * T_alone on the same node.
Solo and contended runs alternate (--repeats) so node drift cancels.

Output: one JSON line per run in --out, then a summary table:
  threads codec  T_alone  T_with  slowdown%  comp_MB/s_alone  comp_MB/s_with

Run from the head node inside an allocation that holds --node:
  python3 interference_bench.py --node ares-comp-04 --out ~/x.jsonl
"""
import argparse
import json
import re
import shlex
import statistics
import subprocess
import threading
import time

MPI_RE = re.compile(r'mpi_stress ranks=(\d+) iters=(\d+)\.\.(\d+) '
                    r'iter_ms_p50=([0-9.]+) wall_s=([0-9.]+)')
COMP_RE = re.compile(r'compress_load .*seconds=([0-9.]+) mb_s=([0-9.]+) '
                     r'ratio=([0-9.]+)')


def parse_args():
    """Command-line options."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--node', required=True)
    p.add_argument('--ppn', type=int, default=40)
    p.add_argument('--bin', default='/mnt/common/llogan/iowarp/dtschedule/'
                   'build/bin')
    p.add_argument('--mpirun', required=True)
    p.add_argument('--launch-agent', default='')
    p.add_argument('--input', default='')
    p.add_argument('--target-s', type=float, default=60.0,
                   help='calibrate the MPI job to about this long alone')
    p.add_argument('--threads', default='4,20,40')
    p.add_argument('--codecs', default='zstd:1,blosc2:1')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--out', required=True)
    return p.parse_args()


def run(cmd):
    """Run a shell command, return stdout+stderr text."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.stdout + res.stderr


def mpi_cmd(a, iters, work_m):
    """mpirun line for the fixed-work MPI job on the node."""
    agent = f'--launch-agent {a.launch_agent} ' if a.launch_agent else ''
    return (f'{a.mpirun} {agent}-np {a.ppn} --host {a.node}:{a.ppn} '
            f'--map-by ppr:{a.ppn}:node:HWTCPUS --bind-to hwthread '
            f'{a.bin}/dtschedule_mpi_stress --mem-mb 64 --halo-kb 64 '
            f'--iters {iters} --work-m {work_m}')


def comp_cmd(a, threads, codec, preset, seconds):
    """ssh line for the compression load on the node."""
    inner = (f'LD_LIBRARY_PATH={a.bin} {a.bin}/dtschedule_compress_load '
             f'--threads {threads} --codec {codec} --preset {preset} '
             f'--seconds {seconds:.1f}')
    if a.input:
        inner += f' --input {a.input}'
    return f'ssh {a.node} {shlex.quote(inner)}'


def mpi_once(a, iters, work_m):
    """Run the MPI job; return its wall seconds (rank 0's summary)."""
    out = run(mpi_cmd(a, iters, work_m))
    m = MPI_RE.search(out)
    if m is None:
        raise RuntimeError(f'mpi_stress failed:\n{out[-2000:]}')
    return float(m.group(5))


def comp_once(a, threads, codec, preset, seconds):
    """Run the compression load; return (MB/s, ratio)."""
    out = run(comp_cmd(a, threads, codec, preset, seconds))
    m = COMP_RE.search(out)
    if m is None:
        raise RuntimeError(f'compress_load failed:\n{out[-2000:]}')
    return float(m.group(2)), float(m.group(3))


def calibrate(a):
    """Pick (iters, work_m) so the job runs about target_s alone."""
    work_m, iters = 2.0, 60
    mpi_once(a, iters, work_m)  # warm-up: page faults, first-touch, turbo
    walls = [mpi_once(a, iters, work_m) for _ in range(2)]
    wall = min(walls)
    iters = max(20, int(iters * a.target_s / max(wall, 0.1)))
    return iters, work_m


def contended(a, iters, work_m, threads, codec, preset, seconds):
    """MPI job with compression for `seconds` from its start."""
    comp = {}

    def bg():
        comp['mb_s'], comp['ratio'] = comp_once(a, threads, codec, preset,
                                                seconds)
    t = threading.Thread(target=bg)
    t.start()
    wall = mpi_once(a, iters, work_m)
    t.join()
    return wall, comp['mb_s'], comp['ratio']


def main():
    """Calibrate, then measure every (threads, codec) cell."""
    a = parse_args()
    iters, work_m = calibrate(a)
    rows = []
    with open(a.out, 'a') as log:
        for codec_spec in a.codecs.split(','):
            codec, preset = codec_spec.split(':')
            for threads in [int(x) for x in a.threads.split(',')]:
                cell = {'codec': codec, 'preset': int(preset),
                        'threads': threads, 'iters': iters, 'work_m': work_m,
                        'alone': [], 'with': [], 'comp_alone': [],
                        'comp_with': []}
                for _ in range(a.repeats):
                    t_alone = mpi_once(a, iters, work_m)
                    secs = 0.5 * t_alone
                    t_with, mb_s, ratio = contended(a, iters, work_m, threads,
                                                    codec, preset, secs)
                    mb_alone, _ = comp_once(a, threads, codec, preset,
                                            min(secs, 15.0))
                    cell['alone'].append(t_alone)
                    cell['with'].append(t_with)
                    cell['comp_with'].append(mb_s)
                    cell['comp_alone'].append(mb_alone)
                    cell['ratio'] = ratio
                log.write(json.dumps(cell) + '\n')
                log.flush()
                rows.append(cell)
                summarize([cell])
    print('==== summary')
    summarize(rows)


def summarize(rows):
    """Print one line per cell."""
    for c in rows:
        ta, tw = statistics.mean(c['alone']), statistics.mean(c['with'])
        print(f"threads={c['threads']:>2} codec={c['codec']}:{c['preset']} "
              f"T_alone={ta:.1f}s T_with={tw:.1f}s "
              f"slowdown={100 * (tw - ta) / ta:+.1f}% "
              f"comp_alone={statistics.mean(c['comp_alone']):.0f}MB/s "
              f"comp_with={statistics.mean(c['comp_with']):.0f}MB/s "
              f"ratio={c.get('ratio', 0):.2f}", flush=True)


if __name__ == '__main__':
    main()
