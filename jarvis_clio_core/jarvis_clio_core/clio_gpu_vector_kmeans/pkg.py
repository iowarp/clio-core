"""
GPU Vector K-Means Benchmark Package

Drives clio_gpu_vector_kmeans_bench: a STREAMING READ over a point set that
does not fit on the device. Each Lloyd iteration walks the whole dataset once,
so a page is touched once per pass and never revisited within it -- the access
pattern neither the flush (write) nor weights (re-read) benchmark exercises.

The binary is self-contained: it writes its own CLIO_SERVER_CONF and starts the
runtime, so no clio_runtime or clio_cte package belongs in the pipeline.

Axes worth sweeping:
- page_kb    16 KB to 8 MB. The dominant axis: measured 612.9 / 219.8 /
             151.9 ms at 16 / 64 / 1024 KB on the same problem.
- blocks     concurrent fault streams
- cache_mb   TOTAL page cache across all blocks, in MB. Prefer this over
             `slots` whenever `blocks` also varies: `slots` is per block, so a
             fixed `slots` makes the cache scale with the block count and the
             two axes cannot be read apart.
- slots      per-block page cache, in pages (ignored when cache_mb is set)
- data_mb    dataset size; keep it at least 2x VRAM to stay out-of-core

Assumes clio_gpu_vector_kmeans_bench is on PATH (it lives in <build>/bin).
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re

def _wait_for_free_vram(log, need_gb, timeout_s=180):
    """Block until the GPU actually has `need_gb` free.

    jarvis starts cells back to back and a finished process releases its VRAM
    LAZILY, so the next cell can begin while the previous one still holds
    memory. Measured: a cell that needs 4 GB of tier saw "GPU free=2908MiB"
    and died -- three cells of a 10-cell GNN sweep were lost this way, and the
    failures did NOT correlate with their own footprint (one needed only 32
    MiB of cache). Without this the sweep silently loses whichever cells
    happen to land in the teardown window.
    """
    import subprocess, time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.free',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=15)
            free_gb = int(out.stdout.strip().splitlines()[0]) / 1024.0
        except Exception:
            return  # no nvidia-smi: nothing to wait on
        if free_gb >= need_gb:
            return
        log(f'  waiting for GPU: {free_gb:.1f} GB free, need {need_gb:.1f} GB')
        time.sleep(10)
    log(f'  WARNING: proceeding with less than {need_gb:.1f} GB free after '
        f'{timeout_s}s -- the cell may fail for lack of device memory')



def _reap_stale_runtime(log, binary, port=9441, timeout_s=60):
    """Kill an orphaned benchmark left behind by a PREVIOUS cell.

    These benchmarks host the runtime in-process, so an orphan keeps the
    runtime's TCP port bound. `timeout` sends SIGTERM, which a WEDGED GPU cell
    does not always honour -- the process survives its own kill, and every
    LATER cell then dies at startup with "Could not start TCP server on any
    host from hostfile", a FATAL that has nothing to do with that cell's own
    settings.

    One hang therefore invalidates the ENTIRE REMAINDER of a sweep rather than
    costing a single cell. Measured: a Gray-Scott cache sweep wedged at 256
    pages/block, and the next cell died instantly on the orphan rather than on
    its own merits. This is also how a sweep can report "36 successful, 0
    failed" while cells were being killed.

    Processes are matched by the exact target of /proc/<pid>/exe, NOT by name:
    pgrep -x compares against a 15-character comm field, which every one of
    these binaries overflows, and pkill -f would also match the shell wrapper
    and the tee. The exe match means an unrelated clio process belonging to
    the user is never touched.
    """
    import os, signal, time

    def orphans():
        found = []
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            try:
                exe = os.path.basename(os.readlink('/proc/%s/exe' % entry))
            except OSError:
                continue  # not ours to see, or already gone
            if exe == binary and int(entry) != os.getpid():
                found.append(int(entry))
        return found

    pids = orphans()
    if not pids:
        return
    log('  REAPING %d orphaned %s process(es) from a previous cell: %s'
        % (len(pids), binary, ' '.join(str(p) for p in pids)))
    log('  (a wedged cell that ignored SIGTERM holds port %d and would fail '
        'every remaining cell of this sweep)' % port)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not orphans():
            log('  reaped; port %d released' % port)
            return
        time.sleep(2)
    log('  WARNING: %s survived SIGKILL for %ds -- this cell will probably '
        'fail to bind port %d' % (binary, timeout_s, port))

class ClioGpuVectorKmeans(Application):
    """K-means streaming-read benchmark over a GPU vector."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'blocks', 'msg': 'CUDA blocks', 'type': int,
             'default': 64},
            {'name': 'threads', 'msg': 'Threads per block', 'type': int,
             'default': 256},
            {'name': 'dims', 'msg': 'Point dimensionality', 'type': int,
             'default': 32},
            {'name': 'clusters', 'msg': 'k (number of centroids)', 'type': int,
             'default': 16},
            {'name': 'slots', 'msg': 'Per-block page cache slots (pages)',
             'type': int, 'default': 8},
            {'name': 'cache_frac',
             'msg': 'TOTAL page cache as a FRACTION OF THE DATASET (e.g. 0.75 '
                    '= three quarters of it). Takes precedence over cache_mb. '
                    'This is the axis that actually moves the hit rate: misses '
                    'scale with 1 - cache/working_set, so what matters is '
                    'COVERAGE, not an absolute byte count. Measured on weights '
                    'at 16GB, taking the cache from 0.05%% to 6.25%% of the '
                    'dataset (128x more slots) removed only 2.3%% of faults -- '
                    'the benefit is a step function that turns on as coverage '
                    'approaches 1, not a smooth 1/cache curve',
             'type': float, 'default': 0.0},
            {'name': 'cache_mb',
             'msg': 'TOTAL page cache across all blocks, in MB. When set '
                    '(non-zero) it overrides `slots`, which is derived as '
                    'cache_mb/blocks/page_kb. This is the axis worth sweeping: '
                    '`slots` is per block, so holding it fixed while blocks '
                    'varies changes the cache by the same factor and confounds '
                    'the two axes',
             'type': int, 'default': 0},
            {'name': 'iters', 'msg': 'Lloyd iterations per timed run',
             'type': int, 'default': 4},
            {'name': 'page_kb',
             'msg': 'Page size in KB. Must divide evenly by dims*4 or the '
                    'benchmark refuses to run rather than round it',
             'type': int, 'default': 1024},
            {'name': 'data_mb',
             'msg': 'Dataset size in MB. Keep >= 2x VRAM so the run is '
                    'genuinely out-of-core',
             'type': int, 'default': 16384},
            {'name': 'hbm_mb', 'msg': 'kHBM tier capacity in MB', 'type': int,
             'default': 4096},
            {'name': 'hbm_only',
             'msg': 'Omit the host spill tier, so a dataset that does not fit '
                    'fails loudly instead of quietly spilling',
             'type': bool, 'default': False},
            {'name': 'repeat', 'msg': 'Timed repetitions (best is reported)',
             'type': int, 'default': 3},
            {'name': 'timeout_sec',
             'msg': 'Kill a run after this many seconds (0 = no limit). A GPU '
                    'kernel can wedge and never return; without a limit one '
                    'stuck cell stalls the rest of a sweep',
             'type': int, 'default': 1800},
            {'name': 'vram_budget_gb',
             'msg': 'Device-memory budget in GB for the page cache '
                    'plus the kHBM tier. A cell that exceeds it is '
                    'refused rather than left to die after printing '
                    'its header',
             'type': float, 'default': 7.0},
            {'name': 'output_dir', 'msg': 'Output directory', 'type': str,
             'default': '/tmp/clio_gpu_vector_kmeans'},
        ]


    def _dataset_mb(self):
        """Size of the workload's dataset in MB."""
        return float(self.config['data_mb'])

    def _cache_mb(self):
        """Total device page cache in MB, from the fraction when one is set.

        `cache_frac` wins over the absolute `cache_mb` because hit rate is
        governed by COVERAGE -- cache divided by working set -- not by an
        absolute size. A 1 GB cache is nearly useless against 16 GB (6.25%%
        coverage, measured 2.3%% fewer faults) and nearly total against 1.3 GB.
        Expressing the axis as a fraction makes the sweep comparable across
        dataset sizes and puts its points where the knee actually is.
        """
        c = self.config
        frac = c.get('cache_frac') or 0.0
        if frac:
            return self._dataset_mb() * frac
        return c.get('cache_mb') or 0

    def _slots(self):
        """Pages per block, derived from the total cache budget when one is set.

        `cache_mb` is the TOTAL device page cache; `slots` is per block. The
        total is what costs VRAM and what a cache-size question is about, so it
        is the swept quantity and `slots` follows from it.

        A cell whose per-block share is under one page is REFUSED, not rounded
        up to one slot: rounding would silently run a larger cache than the one
        asked for (64 MB requested, 256 MB actually allocated at 256 blocks x
        1 MB pages) and report it under the small-cache label.
        """
        c = self.config
        cache_mb = self._cache_mb()
        if not cache_mb:
            return c['slots']
        per_block_kb = cache_mb * 1024.0 / c['blocks']
        slots = int(cache_mb * 1024 // (c['blocks'] * c['page_kb']))
        if slots < 1:
            raise Exception(
                f"cache_mb={cache_mb:.0f} over blocks={c['blocks']} leaves "
                f"{per_block_kb:.1f} KB per block, under one {c['page_kb']}KB "
                f"page. Refused rather than rounded up to "
                f"{c['blocks'] * c['page_kb'] / 1024.0:.0f} MB, which would "
                f"report a larger cache under this cell's label.")
        return slots

    def _actual_cache_mb(self):
        """Cache the run really allocates: blocks x slots x page_kb."""
        c = self.config
        return self._slots() * c['blocks'] * c['page_kb'] / 1024.0

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        slots = self._slots()
        page_floats = c['page_kb'] * 1024 // 4
        if page_floats % c['dims'] != 0:
            # The benchmark rejects this too; failing here names the cell
            # before a run is spent on it.
            raise Exception(
                f"page_kb={c['page_kb']} gives {page_floats} floats per page, "
                f"which is not a multiple of dims={c['dims']}. A point would "
                f"straddle a page boundary.")
        # DEVICE-MEMORY GUARD. The page cache costs blocks * slots *
        # page_kb of VRAM, which explodes on the large-page/many-block corner:
        # a 16 GB k-means sweep lost three cells to this (8192KB x 256 slots=8
        # wants 16 GB of cache on an 8 GB GPU). They printed a header and died,
        # and only the post-processing trust check caught them. Fail here with
        # the arithmetic instead.
        cache_gb = (c['blocks'] * slots * c['page_kb']) / (1024.0 * 1024.0)
        budget_gb = c.get('vram_budget_gb', 7.0)
        if cache_gb + c['hbm_mb'] / 1024.0 > budget_gb:
            raise Exception(
                f"page cache needs {cache_gb:.1f} GB (blocks={c['blocks']} x "
                f"slots={slots} x {c['page_kb']}KB) plus a "
                f"{c['hbm_mb'] / 1024.0:.1f} GB kHBM tier, over the "
                f"{budget_gb:.1f} GB budget. Reduce blocks, slots or page_kb.")
        if c.get('cache_mb'):
            # Exact for power-of-two settings; reported when it is not, because
            # a cache axis whose points are not the requested sizes is not the
            # axis it claims to be.
            got = self._actual_cache_mb()
            if abs(got - self._cache_mb()) > 0.01:
                self.log(f'  NOTE: requested {self._cache_mb():.0f}MB of total cache, '
                         f'allocating {got:.1f}MB ({slots} slots x '
                         f'{c["blocks"]} blocks x {c["page_kb"]}KB)')
        self.log('GPU vector k-means configured')
        self.log(f'  dataset:   {c["data_mb"]}MB, dims={c["dims"]}, '
                 f'k={c["clusters"]}, iters={c["iters"]}')
        self.log(f'  paging:    page={c["page_kb"]}KB '
                 f'({page_floats // c["dims"]} points/page), '
                 f'cache={slots} pages/block '
                 f'({cache_gb * 1024:.0f}MB total over {c["blocks"]} blocks)')
        self.log(f'  parallel:  {c["blocks"]} blocks x {c["threads"]} threads')
        self.log(f'  kHBM tier: {c["hbm_mb"]}MB'
                 f'{" (HBM ONLY)" if c["hbm_only"] else ""}')
        if c['data_mb'] < 2 * c['hbm_mb']:
            self.log(f'  WARNING: dataset {c["data_mb"]}MB is less than 2x the '
                     f'kHBM tier ({c["hbm_mb"]}MB) -- this is not an '
                     f'out-of-core run')

    def _output_file(self):
        """Path of this configuration's log.

        Every swept parameter must appear, or two cells of a sweep write to the
        same file and one silently overwrites the other -- which still looks
        like a completed sweep.
        """
        c = self.config
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{self._slots()}'
               f'_d{c["data_mb"]}mb_hbm{c["hbm_mb"]}_k{c["clusters"]}'
               f'_dim{c["dims"]}_it{c["iters"]}')
        return os.path.join(c['output_dir'], f'kmeans_{tag}.log')

    def _get_stat(self, stats):
        """Harvest the benchmark's numbers into the pipeline results.

        jarvis calls this on a FRESHLY LOADED package instance after the run,
        so nothing survives from start(): the log is re-read from disk and the
        path rebuilt from self.config.

        `ms` is the measured kernel time (best of `repeat`) and excludes setup
        and seeding; jarvis's `runtime` column is whole-process wall clock and
        must not be used as the benchmark's time.

        centroid_checksum is reported for comparison ACROSS cells, but must be
        compared with a relative tolerance: the sums use atomicAdd, so the
        float summation order follows the page and block layout and is not
        associative. Bit-equality is not expected even when a run is correct.
        """
        # Record what the cache axis ACTUALLY was, not what was requested: the
        # requested cache_mb is in the sweep variables already, so carrying the
        # realised slots and total lets the post-processing check the two agree
        # instead of assuming it.
        try:
            stats['slots'] = self._slots()
            stats['cache_mb_actual'] = round(self._actual_cache_mb(), 1)
        except Exception:
            return

        path = self._output_file()
        if not os.path.exists(path):
            return
        ansi = re.compile(r'\x1b\[[0-9;]*m')
        try:
            with open(path, 'r', errors='replace') as f:
                text = ansi.sub('', f.read())
        except OSError:
            return

        line = None
        for ln in text.splitlines():
            if ln.startswith('KMEANS ') and 'ms=' in ln:
                line = ln
        if line is None:
            # No summary line: the binary never got far enough to print one.
            # Emitted so a blank in the CSV means "produced no output" rather
            # than "passed".
            stats['completed'] = 0
            return
        stats['completed'] = 1

        for key, out in (('ms', 'kernel_ms'), ('GB/s', 'gbps'),
                         ('centroid_checksum', 'centroid_checksum'),
                         ('faults', 'faults'), ('evicts', 'evicts'),
                         ('puts', 'puts'), ('points', 'points'),
                         ('get_errors', 'get_errors'),
                         ('put_errors', 'put_errors'),
                         # Achievable H2D bandwidth at THIS page size.
                         ('memcpy_pin_gbps', 'memcpy_pin_gbps'),
                         ('memcpy_page_gbps', 'memcpy_page_gbps')):
            # The key must start at a WORD BOUNDARY. Without it,
            # searching for 'ms=' matches the 'ms=' inside 'dims=32'
            # and silently reports 32 instead of 5044.1 -- a wrong
            # number, not a missing one, which no downstream check
            # would catch.
            m = re.search(r'(?:^|\s)' + re.escape(key) + r'=([0-9.]+)',
                          line)
            if m:
                v = float(m.group(1))
                stats[out] = int(v) if v.is_integer() else v
        # Bytes actually paged per timed run, which is the number the page-size
        # axis moves. faults * page_kb, in MB.
        if 'faults' in stats:
            stats['paged_mb'] = round(
                stats['faults'] * self.config['page_kb'] / 1024.0, 1)

    def start(self):
        # Clear a previous cell's orphan BEFORE anything else: it
        # holds the runtime port and would fail this cell at startup.
        _reap_stale_runtime(self.log, 'clio_gpu_vector_kmeans_bench')
        Which('clio_gpu_vector_kmeans_bench',
              LocalExecInfo(env=self.mod_env)).run()
        slots = self._slots()
        _wait_for_free_vram(self.log,
                            self.config['hbm_mb'] / 1024.0 +
                            (self.config['blocks'] * slots *
                             self.config['page_kb']) / (1024.0 * 1024.0) + 1.0)
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()

        cmd = ['clio_gpu_vector_kmeans_bench',
               f'--blocks {c["blocks"]}', f'--threads {c["threads"]}',
               f'--dims {c["dims"]}', f'--clusters {c["clusters"]}',
               f'--slots {slots}', f'--iters {c["iters"]}',
               f'--page-kb {c["page_kb"]}', f'--data-mb {c["data_mb"]}',
               f'--hbm-mb {c["hbm_mb"]}', f'--repeat {c["repeat"]}']
        if c['hbm_only']:
            cmd.append('--hbm-only')
        if c['timeout_sec'] > 0:
            cmd.insert(0, f'timeout {c["timeout_sec"]}')

        self.log(f'Running: {" ".join(cmd)}')
        # The summary line goes to STDERR; without the redirect the log is
        # empty and every stat silently disappears.
        Exec(f'{" ".join(cmd)} 2>&1 | tee {out}',
             LocalExecInfo(env=self.mod_env, cwd=c['output_dir'])).run()
        self.log(f'Benchmark completed - results saved to {out}')

    def stop(self):
        """The benchmark runs to completion on its own."""
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.startswith('kmeans_') or name.endswith('.yaml'):
                os.remove(os.path.join(out, name))
        try:
            os.rmdir(out)
        except OSError:
            pass
