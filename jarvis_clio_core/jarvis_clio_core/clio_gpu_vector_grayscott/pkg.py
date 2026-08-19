"""
GPU Vector Gray-Scott Benchmark Package

Drives clio_gpu_vector_grayscott_bench: a 3D reaction-diffusion stencil over a
grid that does not fit on the device. The distinguishing feature is a SLIDING
WINDOW -- computing plane z needs z-1, z, z+1, and computing z+1 then needs z,
z+1, z+2 -- so two of every three planes are immediately reused. None of the
other GPU-vector benchmarks has that reuse distance.

ONE PAGE IS ONE XY PLANE, so page_kb sets the plane dimensions (16 KB ->
64x64, 8 MB -> 2048x1024) rather than chunking a fixed grid, which makes page
size a first-class axis.

slots must be >= 8: the kernel holds 6 input planes (z-1,z,z+1 of u and v) and
2 output planes at once. The binary refuses a smaller cache rather than
running, because a plane still being read could otherwise be evicted under it
-- which would not crash, it would silently read whatever replaced it. This
package validates the same rule at configure time so the cell is named before
a run is spent on it.

Assumes clio_gpu_vector_grayscott_bench is on PATH (it lives in <build>/bin).
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re

# The stencil holds z-1, z, z+1 of both fields plus both outputs.
MIN_SLOTS = 8

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

class ClioGpuVectorGrayscott(Application):
    """Gray-Scott stencil benchmark over a GPU vector."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'blocks', 'msg': 'CUDA blocks (each owns a z-slab)',
             'type': int, 'default': 64},
            {'name': 'threads', 'msg': 'Threads per block', 'type': int,
             'default': 256},
            {'name': 'slots',
             'msg': f'Per-block page cache in planes. Must be >= {MIN_SLOTS}: '
                    f'the stencil holds 6 input planes and 2 output planes at '
                    f'once',
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
                    f'varies changes the cache by the same factor and confounds '
                    f'the two axes. The derived value must still clear the '
                    f'{MIN_SLOTS}-plane floor',
             'type': int, 'default': 0},
            {'name': 'steps', 'msg': 'Gray-Scott steps per timed run',
             'type': int, 'default': 4},
            {'name': 'page_kb',
             'msg': 'Page size in KB; ALSO sets the XY plane dimensions, since '
                    'one page is one plane',
             'type': int, 'default': 1024},
            {'name': 'data_mb',
             'msg': 'Total grid size in MB across all four fields (u, v, '
                    'u_next, v_next). Keep >= 2x VRAM to stay out-of-core',
             'type': int, 'default': 16384},
            {'name': 'hbm_mb', 'msg': 'kHBM tier capacity in MB', 'type': int,
             'default': 4096},
            {'name': 'hbm_only',
             'msg': 'Omit the host spill tier so a grid that does not fit '
                    'fails loudly instead of quietly spilling',
             'type': bool, 'default': False},
            {'name': 'repeat', 'msg': 'Timed repetitions (best is reported)',
             'type': int, 'default': 3},
            {'name': 'Du', 'msg': 'u diffusion rate', 'type': float,
             'default': 0.2},
            {'name': 'Dv', 'msg': 'v diffusion rate', 'type': float,
             'default': 0.1},
            {'name': 'F', 'msg': 'feed rate', 'type': float, 'default': 0.02},
            {'name': 'K', 'msg': 'kill rate', 'type': float, 'default': 0.048},
            {'name': 'dt', 'msg': 'timestep', 'type': float, 'default': 1.0},
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
             'default': '/tmp/clio_gpu_vector_grayscott'},
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
        """Planes per block, derived from the total cache budget when one is set.

        `cache_mb` is the TOTAL device page cache; `slots` is per block. The
        total is what costs VRAM and what a cache-size question is about, so it
        is the swept quantity and `slots` follows from it.

        A cell whose per-block share is under one plane is REFUSED, not rounded
        up: rounding would silently run a larger cache than the one asked for
        and report it under the small-cache label. The MIN_SLOTS floor is
        checked separately by the caller, since it is a correctness bound
        rather than an arithmetic one.
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
                f"plane. Refused rather than rounded up to "
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
        if slots < MIN_SLOTS:
            # With cache_mb driving the axis this is reachable by asking for a
            # total too small to give every block 8 planes, so the message
            # names the total as well as the per-block result.
            raise Exception(
                f"slots={slots} but the stencil holds {MIN_SLOTS} planes "
                f"at once (z-1,z,z+1 of u and v, plus both outputs). A smaller "
                f"cache would let a plane still being read be evicted."
                + (f" Derived from cache_mb={c['cache_mb']} over "
                   f"{c['blocks']} blocks at {c['page_kb']}KB; this workload "
                   f"needs at least "
                   f"{MIN_SLOTS * c['blocks'] * c['page_kb'] // 1024} MB."
                   if c.get('cache_mb') else ''))
        plane_elems = c['page_kb'] * 1024 // 4
        nz = (c['data_mb'] * 1024 * 1024 // 4) // (4 * plane_elems)
        if nz < 3:
            raise Exception(
                f"data_mb={c['data_mb']} at page_kb={c['page_kb']} leaves "
                f"nz={nz} planes; a stencil needs at least 3.")
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
        self.log('GPU vector Gray-Scott configured')
        self.log(f'  grid:      one {c["page_kb"]}KB plane = {plane_elems} '
                 f'cells, {nz} planes deep')
        self.log(f'  total:     {c["data_mb"]}MB over 4 fields '
                 f'(u, v, u_next, v_next)')
        self.log(f'  parallel:  {c["blocks"]} blocks x {c["threads"]} threads, '
                 f'cache {slots} planes/block '
                 f'({cache_gb * 1024:.0f}MB total over {c["blocks"]} blocks)')
        self.log(f'  kHBM tier: {c["hbm_mb"]}MB'
                 f'{" (HBM ONLY)" if c["hbm_only"] else ""}')
        if c['data_mb'] < 2 * c['hbm_mb']:
            self.log(f'  WARNING: grid {c["data_mb"]}MB is less than 2x the '
                     f'kHBM tier ({c["hbm_mb"]}MB) -- not an out-of-core run')

    def _output_file(self):
        """Path of this configuration's log.

        Every swept parameter must appear, or two cells write to the same file
        and one silently overwrites the other -- which still looks like a
        completed sweep.
        """
        c = self.config
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{self._slots()}'
               f'_d{c["data_mb"]}mb_hbm{c["hbm_mb"]}_st{c["steps"]}')
        return os.path.join(c['output_dir'], f'grayscott_{tag}.log')

    def _get_stat(self, stats):
        """Harvest the benchmark's numbers into the pipeline results.

        jarvis calls this on a FRESHLY LOADED package instance after the run,
        so nothing survives from start(): the log is re-read from disk and the
        path rebuilt from self.config.

        `ms` is the measured step time (best of `repeat`) and excludes seeding
        and setup; jarvis's `runtime` column is whole-process wall clock and
        must not be used as the benchmark's time.

        v_checksum is comparable ACROSS cells only with a RELATIVE TOLERANCE:
        the reduction uses atomicAdd, so the float summation order follows the
        page and block layout and is not associative.
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
            # Matched on the workload name plus `ms=`, NOT on the field that
            # used to follow it: the summary line gained a `mode=` field when
            # the baseline kernel was added, and `GRAYSCOTT blocks=` then
            # matched nothing -- every cell reported completed=0 with no stats,
            # which reads as a failed run rather than a stale parser.
            if ln.startswith('GRAYSCOTT ') and 'ms=' in ln:
                line = ln
        if line is None:
            # No summary line: the binary never got far enough to print one.
            # Emitted so a blank in the CSV means "produced no output" rather
            # than "passed".
            stats['completed'] = 0
            return
        stats['completed'] = 1

        for key, out in (('ms', 'step_ms'), ('GB/s', 'gbps'),
                         ('v_checksum', 'v_checksum'), ('nx', 'nx'),
                         ('ny', 'ny'), ('nz', 'nz'), ('faults', 'faults'),
                         ('evicts', 'evicts'), ('puts', 'puts'),
                         ('get_errors', 'get_errors'),
                         ('put_errors', 'put_errors'),
                         # Achievable H2D bandwidth at THIS page size.
                         ('memcpy_pin_gbps', 'memcpy_pin_gbps'),
                         ('memcpy_page_gbps', 'memcpy_page_gbps')):
            # Anchored at a word boundary: an unanchored 'ms=' also matches the
            # 'ms=' inside another key, which reports a wrong number rather
            # than a missing one and would pass every downstream check.
            m = re.search(r'(?:^|\s)' + re.escape(key) + r'=([0-9.eE+-]+)',
                          line)
            if m:
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                stats[out] = int(v) if v.is_integer() else v
        if 'faults' in stats:
            stats['paged_mb'] = round(
                stats['faults'] * self.config['page_kb'] / 1024.0, 1)
        # Faults per plane touched: the sliding window makes 8 holds per plane,
        # so a value near 8 means the cache captured nothing and a value near 2
        # means it captured the reuse. This is the number the study is about.
        if stats.get('faults') and stats.get('nz'):
            per_step = stats['faults'] / (stats['nz'] * self.config['steps'])
            stats['faults_per_plane_step'] = round(per_step, 2)

    def start(self):
        # Clear a previous cell's orphan BEFORE anything else: it
        # holds the runtime port and would fail this cell at startup.
        _reap_stale_runtime(self.log, 'clio_gpu_vector_grayscott_bench')
        Which('clio_gpu_vector_grayscott_bench',
              LocalExecInfo(env=self.mod_env)).run()
        _wait_for_free_vram(self.log,
                            self.config['hbm_mb'] / 1024.0 +
                            (self.config['blocks'] * self._slots() *
                             self.config['page_kb']) / (1024.0 * 1024.0) + 1.0)
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()

        cmd = ['clio_gpu_vector_grayscott_bench',
               f'--blocks {c["blocks"]}', f'--threads {c["threads"]}',
               f'--slots {self._slots()}', f'--steps {c["steps"]}',
               f'--page-kb {c["page_kb"]}', f'--data-mb {c["data_mb"]}',
               f'--hbm-mb {c["hbm_mb"]}', f'--repeat {c["repeat"]}',
               f'--Du {c["Du"]}', f'--Dv {c["Dv"]}', f'--F {c["F"]}',
               f'--K {c["K"]}', f'--dt {c["dt"]}']
        if c['hbm_only']:
            cmd.append('--hbm-only')
        if c['timeout_sec'] > 0:
            cmd.insert(0, f'timeout {c["timeout_sec"]}')

        self.log(f'Running: {" ".join(cmd)}')
        # The summary goes to STDERR; without the redirect the log is empty and
        # every stat silently disappears.
        Exec(f'{" ".join(cmd)} 2>&1 | tee {out}',
             LocalExecInfo(env=self.mod_env, cwd=c['output_dir'])).run()
        self.log(f'Benchmark completed - results saved to {out}')

    def stop(self):
        """The benchmark runs to completion on its own."""
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.startswith('grayscott_') or name.endswith('.yaml'):
                os.remove(os.path.join(out, name))
        try:
            os.rmdir(out)
        except OSError:
            pass
