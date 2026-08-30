"""
k-means across every substrate -- ONE benchmark, one package.

WHY THIS EXISTS SEPARATELY FROM clio_gv_workload
------------------------------------------------
clio_gv_workload drives all six workloads from one class, and paid for it
with a comparison that was not a comparison. Two specific defects, both
observed on Delta:

  1. It emitted --blocks / --threads ONLY when they were non-zero
     ("0 = binary default"). A pipeline that left them unset therefore ran
     each substrate at ITS OWN default -- paged kmeans at 32x256, the MPI
     edition at its own 64x256 default, and paged weights at 32x32. The
     resulting "paged is 18.9x slower than MPI" was, to within a factor of
     ~1.2, just a 16x thread deficit on one side of the comparison.

  2. Nothing checked that the binary actually USED the geometry it was
     handed, so a silently-ignored flag was indistinguishable from a slow
     substrate.

This package fixes both by construction:

  * blocks and threads are REQUIRED and are passed to EVERY variant. There
    is no "unset means default" path, because that path is how an
    uncontrolled experiment gets published.
  * after each run the echoed geometry is parsed back out of the benchmark's
    own output and compared against what was requested. A mismatch sets
    geometry_ok=0 and FAILS the cell rather than reporting a number.
  * every timing is also reported per iteration (ms_per_iter). Comparing a
    4-iteration paged run against a 2-iteration MPI run produced a bogus 2x
    once already; normalising removes the whole class of error.
  * the substrates can be pinned to the same answer, not just the same
    inputs: the mpi/nvshmem editions accept --check-csum/--check-tol, so a
    reference centroid checksum makes numerical agreement a gate.

WHAT MUST BE IDENTICAL ACROSS VARIANTS FOR A VALID COMPARISON
  problem   : data_mb, dims, clusters, iters
  geometry  : blocks, threads
Everything else (cache size, page size, tiers) is paged-only and has no
counterpart on a resident substrate -- that asymmetry is the thing being
measured, so it is the only asymmetry allowed.

SIZING blocks FOR THE ACTUAL GPU
The gpu_vector block ladders were written against an RTX 5080 (84 SMs) and
top out at 67. On a 108-SM A100 that under-subscribes the device even at
the top rung, and both substrates then measure launch width rather than
anything interesting. Give `blocks` directly, or give `blocks_per_sm` and
let it resolve against `sm_count` (auto-detected from the GPU name when it
is a model this package knows). Both the request and the resolution land in
results.csv.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
import os
import re
import subprocess
import threading
import time

#: SM counts for the GPUs this study runs on. Used only to resolve
#: blocks_per_sm -> blocks; an unknown GPU is an error, never a guess.
_SM_COUNTS = {
    'A100': 108, 'A40': 84, 'A10': 72,
    'H100': 132, 'H200': 132,
    'V100': 80,
    'RTX 4070': 46, 'RTX 4090': 128, 'RTX 5080': 84,
}


def _detect_gpu_name():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=15)
        return out.stdout.strip().splitlines()[0].strip()
    except Exception:
        return ''


def _detect_sm_count():
    """SM count for the GPU in front of us, or 0 if we cannot say.

    Deliberately a lookup rather than a probe: nvidia-smi has no
    multiprocessor-count query field, and silently guessing the wrong SM
    count would silently mis-size every run.
    """
    name = _detect_gpu_name()
    for key, sms in _SM_COUNTS.items():
        if key.replace(' ', '').lower() in name.replace(' ', '').lower():
            return sms
    return 0


class _VramSampler:
    """Peak nvidia-smi memory.used while the benchmark runs, persisted to a
    file because _get_stat runs on a FRESH pkg instance that shares no
    memory with start()."""

    def __init__(self, path, period_s=0.05):
        self.path = path
        self.period_s = period_s
        self.peak = 0
        self.baseline = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip().splitlines()[0])
        except Exception:
            return None

    def _run(self):
        while not self._stop.is_set():
            v = self._sample()
            if v is not None and v > self.peak:
                self.peak = v
            self._stop.wait(self.period_s)

    def start(self):
        self.baseline = self._sample()
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            with open(self.path, 'w') as f:
                f.write('%d %d\n' % (self.peak, self.baseline or 0))
        except OSError:
            pass


def _wait_for_gpu_idle(log, max_used_mb=700, timeout_s=240):
    """Block until the PREVIOUS cell's VRAM is actually released.

    A finished process frees VRAM lazily, and at 32 GB footprints a
    back-to-back cell can otherwise start against a GPU that still holds
    most of the last one. Measured previously as cells running 2x slow AND
    failing physics gates mid-sweep, then passing standalone.
    """
    deadline = time.time() + timeout_s
    used = 0
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=15)
            used = int(out.stdout.strip().splitlines()[0])
        except Exception:
            return
        if used <= max_used_mb:
            return
        log('  waiting for GPU to drain: %d MB still held' % used)
        time.sleep(5)
    log('  WARNING: GPU still holds %d MB after %ds' % (used, timeout_s))


class ClioGvKmeans(Application):
    """One cell = one k-means run on one substrate, at a geometry that is
    stated rather than inherited."""

    VARIANTS = ('paged', 'mpi', 'nvshmem')

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            # ---- substrate ------------------------------------------
            {'name': 'variant', 'msg': 'paged | mpi | nvshmem',
             'type': str, 'default': 'paged'},

            # ---- THE PROBLEM (identical across variants) -------------
            {'name': 'data_mb', 'msg': 'dataset size in MB (the point set)',
             'type': int, 'default': 0},
            {'name': 'dims', 'msg': 'point dimensionality',
             'type': int, 'default': 32},
            {'name': 'clusters', 'msg': 'k', 'type': int, 'default': 16},
            {'name': 'iters', 'msg': 'Lloyd iterations (timings are also '
                                     'reported per iteration)',
             'type': int, 'default': 4},

            # ---- THE GEOMETRY (identical across variants, REQUIRED) --
            {'name': 'blocks',
             'msg': 'CUDA blocks. REQUIRED unless blocks_per_sm is set; '
                    'there is deliberately no "binary default" path',
             'type': int, 'default': 0},
            {'name': 'blocks_per_sm',
             'msg': 'resolve blocks as blocks_per_sm * sm_count, so a '
                    'sweep is portable across GPUs instead of pinned to '
                    'the SM count of the card it was written on',
             'type': int, 'default': 0},
            {'name': 'sm_count',
             'msg': 'SMs on the target GPU (0 = detect from the GPU name; '
                    'an unrecognised GPU is an error, not a guess)',
             'type': int, 'default': 0},
            {'name': 'threads', 'msg': 'threads per block. REQUIRED',
             'type': int, 'default': 256},

            # ---- paged-only: the cache and tier stack ----------------
            {'name': 'cache_mb',
             'msg': 'paged: TOTAL page-cache MB. Resolved to per-block '
                    'slots as cache_mb/blocks/page_kb -- so changing '
                    'blocks at fixed cache_mb holds the VRAM budget '
                    'constant, which is what makes a blocks sweep mean '
                    'anything',
             'type': int, 'default': 0},
            {'name': 'slots',
             'msg': 'paged: per-block slots directly (ignored when '
                    'cache_mb is set)', 'type': int, 'default': 0},
            {'name': 'page_kb', 'msg': 'paged: page size KB',
             'type': int, 'default': 1024},
            {'name': 'hbm_mb', 'msg': 'paged: kHBM CTE tier MB',
             'type': int, 'default': 256},
            {'name': 'nvme_mb',
             'msg': 'paged: file tier MB (>0 gives the full three-tier '
                    'stack hbm+ram+file)', 'type': int, 'default': 0},
            {'name': 'nvme_path', 'msg': 'paged: file tier path',
             'type': str, 'default': '/tmp/gv_kmeans_tier.dat'},

            # ---- cross-substrate numerical agreement ----------------
            {'name': 'check_csum',
             'msg': 'reference centroid checksum. mpi/nvshmem gate on it '
                    'via --check-csum, so the substrates are pinned to the '
                    'same ANSWER and not merely the same inputs. Empty = '
                    'no gate', 'type': str, 'default': ''},
            {'name': 'check_tol', 'msg': 'relative tolerance for check_csum',
             'type': str, 'default': ''},

            # ---- harness --------------------------------------------
            {'name': 'nprocs', 'msg': 'ranks/PEs for mpi and nvshmem',
             'type': int, 'default': 1},
            {'name': 'mpi_launcher',
             'msg': 'launcher for mpi/nvshmem; "{n}" becomes nprocs. The '
                    'default is OpenMPI. On a Cray site with cray-mpich '
                    'use "srun -n {n}"; under Slurm with OpenMPI you also '
                    'need --bind-to none or mpirun fails to bind inside '
                    'the cgroup cpuset',
             'type': str,
             'default': 'mpirun -n {n} --oversubscribe --bind-to none'},
            {'name': 'prefault',
             'msg': 'CLIO_PREFAULT ("0" pre-faults the whole RAM tier at '
                    'compose, so timings exclude first-touch population)',
             'type': str, 'default': '0'},
            {'name': 'timeout_sec', 'msg': 'kill the cell after N seconds',
             'type': int, 'default': 900},
            {'name': 'output_dir', 'msg': 'log directory', 'type': str,
             'default': '/tmp/clio_gv_kmeans'},
        ]

    # ------------------------------------------------------------------
    # geometry resolution -- the part that has to be airtight
    # ------------------------------------------------------------------

    def _sm_count(self):
        c = self.config
        if c['sm_count']:
            return int(c['sm_count'])
        sms = _detect_sm_count()
        if not sms:
            raise Exception(
                'cannot determine the SM count for GPU %r. Set sm_count '
                'explicitly, or set blocks directly instead of '
                'blocks_per_sm.' % (_detect_gpu_name() or 'unknown'))
        return sms

    def _blocks(self):
        """The block count, from an explicit value or from blocks_per_sm.

        Refuses rather than defaulting: an unset geometry is exactly how
        the previous comparison ended up measuring launch width.
        """
        c = self.config
        if c['blocks'] and c['blocks_per_sm']:
            raise Exception('set blocks OR blocks_per_sm, not both '
                            '(blocks=%s, blocks_per_sm=%s)'
                            % (c['blocks'], c['blocks_per_sm']))
        if c['blocks']:
            return int(c['blocks'])
        if c['blocks_per_sm']:
            return int(c['blocks_per_sm']) * self._sm_count()
        raise Exception(
            'blocks is REQUIRED: set blocks, or blocks_per_sm. This '
            'package has no "binary default" path on purpose -- letting '
            'each substrate pick its own is how a substrate comparison '
            'silently becomes a launch-geometry comparison.')

    def _threads(self):
        t = int(self.config['threads'])
        if t <= 0:
            raise Exception('threads is REQUIRED and must be > 0')
        return t

    def _slots(self):
        """Per-block cache slots from the TOTAL cache budget.

        Total is the swept quantity because total is what costs VRAM. A
        share below one page is refused rather than rounded up, which
        would silently hand the run more cache than the sweep asked for.
        """
        c = self.config
        if not c['cache_mb']:
            return int(c['slots'])
        blocks, page_kb = self._blocks(), int(c['page_kb'])
        slots = int(c['cache_mb']) * 1024 // (blocks * page_kb)
        if slots < 1:
            raise Exception(
                'cache_mb=%d over blocks=%d x page_kb=%d leaves under one '
                'page per block' % (c['cache_mb'], blocks, page_kb))
        return slots

    # ------------------------------------------------------------------
    # command construction
    # ------------------------------------------------------------------

    def _binary(self):
        v = self.config['variant']
        if v not in self.VARIANTS:
            raise Exception('unknown variant %r (expected one of %s)'
                            % (v, ', '.join(self.VARIANTS)))
        return 'clio_kmeans_%s_bench' % v

    def _args(self):
        c = self.config
        v = c['variant']
        a = []

        # THE CONTROLLED SET -- always emitted, for every substrate.
        a += ['--blocks %d' % self._blocks(),
              '--threads %d' % self._threads(),
              '--dims %d' % int(c['dims']),
              '--clusters %d' % int(c['clusters']),
              '--iters %d' % int(c['iters'])]
        if not c['data_mb']:
            raise Exception('data_mb is REQUIRED (it is the problem size, '
                            'and it must match across substrates)')
        a.append('--data-mb %d' % int(c['data_mb']))

        if v == 'paged':
            slots = self._slots()
            if slots:
                a.append('--slots %d' % slots)
            a.append('--page-kb %d' % int(c['page_kb']))
            if c['hbm_mb']:
                a.append('--hbm-mb %d' % int(c['hbm_mb']))
            if c['nvme_mb']:
                a += ['--nvme-mb %d' % int(c['nvme_mb']),
                      '--nvme-path %s' % c['nvme_path']]
            # --repeat 1, ALWAYS AND EXPLICITLY. The paged bench defaults
            # to `int repeat = 3` and reports the BEST of them, while the
            # resident substrates get exactly one shot -- so omitting the
            # flag does not mean "no repeats", it means a silent best-of-3
            # bias in paged's favour. There is deliberately no knob: run
            # more ITERATIONS if you want a steadier number, which costs
            # every substrate equally instead of only helping one.
            a.append('--repeat 1')
        else:
            # Only the resident substrates can be gated on a reference
            # answer; the paged run is the one that PRODUCES it.
            if c['check_csum']:
                a.append('--check-csum %s' % c['check_csum'])
                if c['check_tol']:
                    a.append('--check-tol %s' % c['check_tol'])
        return a

    def _cmd(self):
        c = self.config
        parts = []
        if c['timeout_sec'] > 0:
            # --kill-after: a wedged GPU cell can ignore SIGTERM and hold
            # the runtime port against every later cell.
            parts.append('timeout -k 30 %d' % int(c['timeout_sec']))
        if c['variant'] in ('mpi', 'nvshmem'):
            parts.append(c['mpi_launcher'].format(n=int(c['nprocs'])))
        parts.append(self._binary())
        parts += self._args()
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # naming: every controlled parameter lands in the filename, or two
    # cells overwrite each other and the sweep still looks complete
    # ------------------------------------------------------------------

    def _tag(self):
        c = self.config
        return ('kmeans_%s_np%d_b%d_t%d_d%d_k%d_i%d_mb%d_pg%d_ca%d'
                % (c['variant'], int(c['nprocs']), self._blocks(),
                   self._threads(), int(c['dims']), int(c['clusters']),
                   int(c['iters']), int(c['data_mb']), int(c['page_kb']),
                   int(c['cache_mb'])))

    def _output_file(self):
        return os.path.join(self.config['output_dir'], self._tag() + '.log')

    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.log('kmeans cell: %s' % self._cmd())

    def start(self):
        c = self.config
        _wait_for_gpu_idle(self.log)
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        if c['prefault'] != '':
            self.setenv('CLIO_PREFAULT', str(c['prefault']))
        cmd = self._cmd()
        self.log('Running: %s' % cmd)
        sampler = _VramSampler(out + '.vram')
        sampler.start()
        try:
            # Summary lines go to stderr on some editions: capture both.
            Exec('%s 2>&1 | tee %s' % (cmd, out),
                 LocalExecInfo(env=self.mod_env, cwd=c['output_dir'])).run()
        finally:
            sampler.stop()
        self.log('cell done -> %s' % out)

    def stop(self):
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            os.remove(os.path.join(out, name))

    # ------------------------------------------------------------------
    # stats -- including the checks that make the row trustworthy
    # ------------------------------------------------------------------

    def _get_stat(self, stats):
        c = self.config
        blocks, threads = self._blocks(), self._threads()

        # The controlled variables, recorded so results.csv PROVES the
        # comparison was matched instead of asserting it in a comment.
        stats['variant'] = c['variant']
        stats['binary'] = self._binary()
        stats['blocks_req'] = blocks
        stats['threads_req'] = threads
        stats['threads_total'] = blocks * threads
        stats['dims'] = int(c['dims'])
        stats['clusters'] = int(c['clusters'])
        stats['iters'] = int(c['iters'])
        stats['data_mb'] = int(c['data_mb'])
        stats['nprocs'] = int(c['nprocs'])
        if c['variant'] == 'paged':
            stats['cache_mb'] = int(c['cache_mb'])
            stats['page_kb'] = int(c['page_kb'])
            stats['slots'] = self._slots()
        try:
            sms = self._sm_count()
            stats['sm_count'] = sms
            stats['blocks_per_sm_eff'] = round(blocks / float(sms), 3)
        except Exception:
            pass

        out = self._output_file()
        try:
            with open(out + '.vram') as f:
                peak, base = f.read().split()
                stats['vram_peak_mb'] = int(peak)
                stats['vram_delta_mb'] = int(peak) - int(base)
        except (OSError, ValueError):
            pass

        if not os.path.exists(out):
            stats['completed'] = 0
            return
        try:
            with open(out, errors='replace') as f:
                text = re.sub(r'\x1b\[[0-9;]*m', '', f.read())
        except OSError:
            stats['completed'] = 0
            return

        def grab(pattern, key, cast=float):
            m = re.findall(pattern, text)
            if m:
                try:
                    stats[key] = cast(m[-1])
                except ValueError:
                    pass

        # ---- GEOMETRY VERIFICATION -------------------------------------
        # The paged edition echoes "blocks=N thr=M". If a binary ever
        # ignores or clamps what we asked for, this is what catches it --
        # the failure mode that made the previous comparison meaningless
        # was invisible precisely because nothing read the echo back.
        got_b = re.search(r'\bblocks=(\d+)', text)
        got_t = re.search(r'\bthr(?:eads)?=(\d+)', text)
        if got_b or got_t:
            ok = True
            if got_b:
                stats['blocks_used'] = int(got_b.group(1))
                ok = ok and int(got_b.group(1)) == blocks
            if got_t:
                stats['threads_used'] = int(got_t.group(1))
                ok = ok and int(got_t.group(1)) == threads
            stats['geometry_ok'] = int(ok)
        else:
            # mpi/nvshmem do not echo their launch geometry. The flags are
            # accepted and used (AssignKernel<<<blocks, threads>>>), but we
            # cannot confirm it from the output, so say so rather than
            # imply a check happened.
            stats['geometry_ok'] = -1

        # ---- timing ----------------------------------------------------
        grab(r'(\d+) iters in', 'iters_run', int)
        grab(r'iters in ([0-9.]+) ms', 'bench_ms')
        grab(r'(?:^|\s)ms=([0-9.]+)', 'bench_ms')
        grab(r'(?:^|\s)GB/s=([0-9.]+)', 'gbps')
        grab(r'centroid_checksum=([0-9.]+)', 'centroid_checksum')
        for key in ('faults', 'evicts', 'puts', 'get_errors', 'put_errors'):
            grab(r'(?:^|\s)%s=(\d+)' % key, key, int)

        # PER-ITERATION is the comparable quantity. Comparing a 4-iteration
        # run against a 2-iteration one produced a bogus 2x once already.
        if 'bench_ms' in stats and int(c['iters']) > 0:
            stats['ms_per_iter'] = round(stats['bench_ms'] / int(c['iters']),
                                         4)

        # ---- correctness ------------------------------------------------
        # The resident editions print an explicit gate line; the paged
        # edition's contract is clean counters plus a checksum.
        if 'ALL GATES PASS' in text:
            stats['gates_pass'] = 1
        elif c['variant'] == 'paged':
            clean = (stats.get('get_errors', 0) == 0 and
                     stats.get('put_errors', 0) == 0)
            stats['gates_pass'] = int(clean and 'bench_ms' in stats)
        else:
            stats['gates_pass'] = 0
        if 'CSUM' in text:
            stats['csum_match'] = int('CSUM GATE: PASS' in text or
                                      'csum ok' in text.lower())

        stats['completed'] = int('bench_ms' in stats)
        # A row whose geometry did not match what was requested is not a
        # measurement of anything, so refuse to call it a pass.
        if stats.get('geometry_ok') == 0:
            stats['gates_pass'] = 0
