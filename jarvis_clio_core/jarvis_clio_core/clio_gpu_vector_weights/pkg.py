"""
GPU Vector Model-Weights Benchmark Package

Drives clio_gpu_vector_weights_bench, which walks a model's weight matrix out
of a GPU vector -- the inference-side access pattern, as opposed to the flush
benchmark's write-back pattern. Weights are read-mostly and are re-read every
forward pass, so the questions it answers are whether the working set fits in
the kHBM tier, and what compression buys when it does not.

The binary is self-contained: it writes its own CLIO_SERVER_CONF and starts the
runtime, so no clio_runtime or clio_cte package belongs in the pipeline
alongside it. The kHBM tier is sized with `hbm_mb`.

The axes worth sweeping:
- hbm_mb        how much of the model fits on the device
- compressed    raw vs a GPU codec, which is the whole point when it does not
- pages         the working set (pages per block)
- cache_mb      TOTAL page cache across all blocks, in MB. Prefer this over
                `slots` whenever `blocks` also varies: `slots` is per block, so
                a fixed `slots` makes the cache scale with the block count and
                the two axes cannot be read apart.
- slots         the per-block page cache (ignored when cache_mb is set)
- flat_pct      fraction of the matrix that is low-entropy, which sets how much
                a compressor can actually win

Assumes clio_gpu_vector_weights_bench is on PATH (it lives in <build>/bin).
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

class ClioGpuVectorWeights(Application):
    """Model-weight paging benchmark over a GPU vector."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'blocks', 'msg': 'CUDA blocks',
             'type': int, 'default': 64},
            {'name': 'threads', 'msg': 'Threads per block',
             'type': int, 'default': 256},
            {'name': 'rt_threads', 'msg': 'Runtime worker threads',
             'type': int, 'default': 8},
            {'name': 'hbm_mb', 'msg': 'kHBM tier capacity in MB',
             'type': int, 'default': 1024},
            {'name': 'hbm_only',
             'msg': 'Configure ONLY the kHBM tier (no host spill), so a run '
                    'that does not fit fails instead of silently spilling',
             'type': bool, 'default': False},
            {'name': 'pages', 'msg': 'Pages of weights per block (working set)',
             'type': int, 'default': 64},
            {'name': 'data_mb',
             'msg': 'Total weight matrix in MB. When set (non-zero) it '
                    'overrides `pages`, which is derived as '
                    'data_mb/blocks/page_kb. Use this whenever blocks or '
                    'page_kb is swept: the dataset is blocks*pages*page_kb, so '
                    'a fixed `pages` silently changes the DATASET SIZE along '
                    'those axes, and a dataset-size effect then reads as a '
                    'page-size one',
             'type': int, 'default': 0},
            {'name': 'page_kb',
             'msg': 'Vector page size in KB. The DATA granule stays 64 KB, so '
                    'the generated bytes are identical at every page size',
             'type': int, 'default': 64},
            {'name': 'slots', 'msg': 'Per-block page cache slots',
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
            {'name': 'vram_budget_gb',
             'msg': 'Device-memory budget in GB for the page cache plus the '
                    'kHBM tier. A cell that exceeds it is refused rather than '
                    'left to die after printing its header',
             'type': float, 'default': 7.0},
            {'name': 'compressed', 'msg': 'Store the weights compressed',
             'type': bool, 'default': False},
            {'name': 'cpu_codec',
             'msg': 'Use a CPU codec instead of the GPU one. Off by default: '
                    'a CPU codec on the fault path is the thing the GPU vector '
                    'exists to avoid, so it is only ever a comparison point',
             'type': bool, 'default': False},
            {'name': 'no_prefetch', 'msg': 'Disable prefetching',
             'type': bool, 'default': False},
            {'name': 'yieldable',
             'msg': 'Use the yieldable (coroutine) kernel form',
             'type': bool, 'default': True},
            {'name': 'flat_pct',
             'msg': 'Percent of the weight matrix that is low-entropy. Sets '
                    'the ceiling on what any compressor can win, so a '
                    'compression result is meaningless without stating it',
             'type': int, 'default': 50},
            {'name': 'repeat', 'msg': 'Timed repetitions (best is reported)',
             'type': int, 'default': 3},
            {'name': 'timeout_sec',
             'msg': 'Kill a run after this many seconds (0 = no limit). A GPU '
                    'kernel can wedge and never return, which without a limit '
                    'stalls every remaining cell of a sweep',
             'type': int, 'default': 900},
            {'name': 'output_dir', 'msg': 'Output directory',
             'type': str, 'default': '/tmp/clio_gpu_vector_weights'},
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
        asked for and report it under the small-cache label.
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

    def _pages(self):
        """Pages of weights per block, derived from the dataset size when set.

        The dataset is blocks * pages * page_kb, so holding `pages` fixed while
        blocks or page_kb is swept changes the DATASET, not just the paging
        granularity. That failure has already produced a result here: a sweep
        that varied page_kb at fixed `pages` shrank the matrix from 16 GB to
        256 MB across the axis and reported a spurious 92x "page-size effect".
        Deriving pages from data_mb makes the dataset an invariant of the sweep
        instead of something each cell has to be trusted to preserve.
        """
        c = self.config
        if not c.get('data_mb'):
            return c['pages']
        denom = c['blocks'] * c['page_kb']
        total_kb = c['data_mb'] * 1024
        if total_kb % denom != 0:
            raise Exception(
                f"data_mb={c['data_mb']} does not divide into blocks="
                f"{c['blocks']} x page_kb={c['page_kb']}. Rounding would make "
                f"this cell's dataset differ from the rest of the sweep, which "
                f"is the one thing a page-size result must not do.")
        pages = total_kb // denom
        if pages < 1:
            raise Exception(
                f"data_mb={c['data_mb']} over blocks={c['blocks']} at "
                f"page_kb={c['page_kb']} leaves under one page per block.")
        return int(pages)

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(self.config['output_dir'], exist_ok=True)
        slots = self._slots()
        pages = self._pages()
        # DEVICE-MEMORY GUARD. The page cache costs blocks * slots * page_kb of
        # VRAM and is charged against the same card as the kHBM tier. Refusing
        # here names the cell; the alternative is a run that prints its header
        # and dies, which reads as a hang.
        cache_gb = (c['blocks'] * slots * c['page_kb']) / (1024.0 * 1024.0)
        budget_gb = c.get('vram_budget_gb', 7.0)
        if cache_gb + c['hbm_mb'] / 1024.0 > budget_gb:
            raise Exception(
                f"page cache needs {cache_gb:.1f} GB (blocks={c['blocks']} x "
                f"slots={slots} x {c['page_kb']}KB) plus a "
                f"{c['hbm_mb'] / 1024.0:.1f} GB kHBM tier, over the "
                f"{budget_gb:.1f} GB budget. Reduce blocks, cache_mb or "
                f"page_kb.")
        if self._cache_mb():
            got = self._actual_cache_mb()
            if abs(got - self._cache_mb()) > 0.01:
                self.log(f'  NOTE: requested {self._cache_mb():.0f}MB of total cache, '
                         f'allocating {got:.1f}MB ({slots} slots x '
                         f'{c["blocks"]} blocks x {c["page_kb"]}KB)')
        self.log('GPU vector weights benchmark configured')
        self.log(f'  blocks x threads: {self.config["blocks"]} x '
                 f'{self.config["threads"]}')
        self.log(f'  kHBM tier:        {self.config["hbm_mb"]}MB'
                 f'{" (HBM ONLY)" if self.config["hbm_only"] else ""}')
        self.log(f'  working set:      {pages} pages/block '
                 f'({pages * c["blocks"] * c["page_kb"] / 1024.0:.0f}MB total '
                 f'at {c["page_kb"]}KB pages), '
                 f'cache {slots} slots ({cache_gb * 1024:.0f}MB total)')
        self.log(f'  codec:            '
                 f'{"compressed" if self.config["compressed"] else "raw"}'
                 f'{" (CPU codec)" if self.config["cpu_codec"] else ""}')
        self.log(f'  flat_pct:         {self.config["flat_pct"]}%')

    def _output_file(self):
        """Path of this configuration's log.

        Every swept parameter belongs in the name. With a partial tag, two
        cells of a sweep write to the same file and one silently overwrites
        the other, which still looks like a completed sweep.
        """
        c = self.config
        tag = (f'b{c["blocks"]}_hbm{c["hbm_mb"]}_pgkb{c["page_kb"]}'
               f'_pg{self._pages()}'
               f'_sl{self._slots()}_flat{c["flat_pct"]}'
               f'_{"cmp" if c["compressed"] else "raw"}'
               f'{"_cpucodec" if c["cpu_codec"] else ""}'
               f'{"_hbmonly" if c["hbm_only"] else ""}')
        return os.path.join(c['output_dir'], f'gvw_{tag}.log')

    def _get_stat(self, stats):
        """Harvest the benchmark's numbers into the pipeline results.

        jarvis calls this on a FRESHLY LOADED package instance after the run,
        so nothing survives from start(): the log is re-read from disk and the
        path rebuilt from self.config.

        The benchmark emits one summary line of key=value pairs:
          GVW mode=nvcomp+yield blocks=64 ... ms=123 GB/s=4.56 checksum=OK ...
        so the parse is a single regex over that line rather than a scrape of
        prose. `ms` is the measured kernel time (best of `repeat`) and excludes
        setup; jarvis's own `runtime` column is whole-process wall clock and
        must not be used as the benchmark's time.
        """
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
            if 'GVW mode=' in ln:
                line = ln          # last one wins; there is normally one
        if line is None:
            # No summary line: the binary never got far enough to print it.
            # Emitted so a blank in the CSV means "produced no output" rather
            # than "passed".
            stats['completed'] = 0
            stats['checksum_ok'] = 0
            return
        stats['completed'] = 1

        num = {'ms': 'kernel_ms', 'GB/s': 'gbps', 'logical': 'logical_mb',
               'stored': 'stored_mb', 'faults': 'faults', 'evicts': 'evicts',
               'put_errors': 'put_errors', 'get_errors': 'get_errors',
               'rounds': 'rounds', 'slots': 'slots', 'pages': 'pages',
               # Achievable H2D bandwidth at THIS page size, probed on an
               # idle device before the runtime starts. The reference the
               # paging rate should be read against.
               'memcpy_pin_gbps': 'memcpy_pin_gbps',
               'memcpy_page_gbps': 'memcpy_page_gbps'}
        for key, out in num.items():
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
        m = re.search(r'mode=(\S+)', line)
        if m:
            stats['mode'] = m.group(1)
        # fits=yes/no is the question the whole benchmark asks, and checksum
        # decides whether any of the timings mean anything.
        stats['fits_in_hbm'] = 1 if re.search(r'fits=yes', line) else 0
        stats['checksum_ok'] = 1 if re.search(r'checksum=OK', line) else 0
        # The cache axis as the binary actually ran it, derived from the slots
        # the BINARY reports rather than from the requested cache_mb, so a
        # disagreement between the two shows up in the results instead of
        # being assumed away.
        if stats.get('slots'):
            stats['cache_mb_actual'] = round(
                stats['slots'] * self.config['blocks'] *
                self.config['page_kb'] / 1024.0, 1)
        if stats.get('logical_mb') and stats.get('stored_mb'):
            stats['compress_ratio'] = round(
                stats['logical_mb'] / stats['stored_mb'], 3)

    def start(self):
        # Clear a previous cell's orphan BEFORE anything else: it
        # holds the runtime port and would fail this cell at startup.
        _reap_stale_runtime(self.log, 'clio_gpu_vector_weights_bench')
        Which('clio_gpu_vector_weights_bench',
              LocalExecInfo(env=self.mod_env)).run()
        slots = self._slots()
        # A cache slot holds one PAGE, so it costs page_kb -- not the 64 KB
        # data granule. The two are equal only at the default page size; at
        # 4 MB pages the old constant under-counted the cache by 64x and the
        # wait let a cell start against a card that could not hold it.
        _wait_for_free_vram(self.log,
                            self.config['hbm_mb'] / 1024.0 +
                            (self.config['blocks'] * slots *
                             self.config['page_kb']) / (1024.0 * 1024.0) + 1.0)
        os.makedirs(self.config['output_dir'], exist_ok=True)
        out = self._output_file()
        c = self.config

        cmd = ['clio_gpu_vector_weights_bench',
               f'--blocks {c["blocks"]}', f'--threads {c["threads"]}',
               f'--rt-threads {c["rt_threads"]}', f'--hbm-mb {c["hbm_mb"]}',
               f'--pages {self._pages()}', f'--slots {slots}',
               # MUST be passed. Without it the binary silently used its
               # 64 KB default while the sweep believed it was varying the
               # page size -- every cell ran a different DATASET SIZE instead
               # (blocks*pages*64KB), and the resulting "92x page-size effect"
               # was really a dataset-size effect. Caught by logical_mb, which
               # is reported precisely so a run that quietly changed size
               # cannot be read as a page-size result.
               f'--page-kb {c["page_kb"]}',
               f'--flat-pct {c["flat_pct"]}', f'--repeat {c["repeat"]}']
        if c['hbm_only']:
            cmd.append('--hbm-only')
        if c['compressed']:
            cmd.append('--compressed')
        if c['cpu_codec']:
            cmd.append('--cpu-codec')
        if c['no_prefetch']:
            cmd.append('--no-prefetch')
        if c['yieldable']:
            cmd.append('--yieldable')
        if c['timeout_sec'] > 0:
            cmd.insert(0, f'timeout {c["timeout_sec"]}')

        self.log(f'Running: {" ".join(cmd)}')
        # The summary line goes to STDERR, so it must be redirected or the
        # log ends up empty and every stat silently disappears.
        Exec(f'{" ".join(cmd)} 2>&1 | tee {out}',
             LocalExecInfo(env=self.mod_env, cwd=c['output_dir'])).run()
        self.log(f'Benchmark completed - results saved to {out}')

    def stop(self):
        """The benchmark runs to completion on its own."""
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.startswith('gvw_') or name.endswith('.yaml'):
                os.remove(os.path.join(out, name))
        try:
            os.rmdir(out)
        except OSError:
            pass
