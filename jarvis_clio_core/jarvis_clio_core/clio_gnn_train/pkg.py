"""
GNN Training Benchmark Package (GPU vector)

Drives test_gpu_vector_gnn_train: GNN training whose aggregated feature matrix
is paged out of a GPU vector. The access pattern is a GATHER -- each training
window pulls scattered node rows -- which is different again from the other
three GPU-vector benchmarks (flush writes, weights re-reads, k-means streams,
Gray-Scott slides a window).

PAGE SIZE IS EXPRESSED IN KB HERE, not in rows. The binary takes
CLIO_GNN_PAGE_ROWS (rows per page) and a row is dim*4 bytes, so this package
converts: page_rows = page_kb * 1024 / (dim * 4). That keeps the page-size
axis directly comparable with the other benchmarks' --page-kb rather than
forcing a reader to do the arithmetic. A page_kb that does not divide into a
whole number of rows is REJECTED rather than rounded, because a rounded page
size makes a page-size sweep report something other than what it swept.

The binary is a ctest-style executable that starts its own runtime and writes
its own CLIO_SERVER_CONF; everything is configured through CLIO_GNN_* env
vars, so no clio_runtime or clio_cte package belongs in the pipeline.

Assumes test_gpu_vector_gnn_train is on PATH (it lives in <build>/bin).
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

class ClioGnnTrain(Application):
    """GNN training over a paged feature matrix."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'nodes',
             'msg': 'Synthetic node count. nodes * dim * 4 bytes is the '
                    'feature matrix; keep it >= 2x VRAM to stay out-of-core',
             'type': int, 'default': 33554432},
            {'name': 'dim', 'msg': 'Feature dimensionality', 'type': int,
             'default': 128},
            {'name': 'classes', 'msg': 'Output classes', 'type': int,
             'default': 40},
            {'name': 'hidden', 'msg': 'Hidden layer width', 'type': int,
             'default': 64},
            {'name': 'epochs', 'msg': 'Training epochs', 'type': int,
             'default': 2},
            {'name': 'page_kb',
             'msg': 'Page size in KB. Converted to CLIO_GNN_PAGE_ROWS as '
                    'page_kb*1024/(dim*4); must divide evenly',
             'type': int, 'default': 1024},
            {'name': 'gather_blocks', 'msg': 'CUDA blocks for the gather',
             'type': int, 'default': 32},
            {'name': 'pages_per_block', 'msg': 'Per-block page cache slots',
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
             'msg': 'TOTAL page cache across all gather blocks, in MB. When '
                    'set (non-zero) it overrides `pages_per_block`, which is '
                    'derived as cache_mb/gather_blocks/page_kb. This is the '
                    'axis worth sweeping: `pages_per_block` is per block, so '
                    'holding it fixed while gather_blocks varies changes the '
                    'cache by the same factor and confounds the two axes',
             'type': int, 'default': 0},
            {'name': 'window_mb',
             'msg': 'Training window in MEGABYTES, converted to pages as '
                    'window_mb*1024/page_kb. Sized in bytes, not pages, '
                    'because a fixed page COUNT makes the staging scratch '
                    'scale with page size: 256 pages is 4 MiB at 16 KB pages '
                    'but 1024 MiB at 4 MB pages (peak 2048 MiB), which '
                    'exhausted the card and lost both 4 MB cells of a sweep',
             'type': int, 'default': 256},
            {'name': 'hbm_mib', 'msg': 'kHBM tier capacity in MiB',
             'type': int, 'default': 4096},
            {'name': 'dram_mib', 'msg': 'Host DRAM tier capacity in MiB',
             'type': int, 'default': 20480},
            {'name': 'dram_type',
             'msg': 'Host tier allocation: "pinned" (the honest opponent) or '
                    '"ram" (pageable, which penalises the tier for its '
                    'allocation type rather than its speed)',
             'type': str, 'default': 'pinned'},
            {'name': 'compress_lib',
             'msg': 'Compressor id (0 = uncompressed, 16 = nvcomp-ans)',
             'type': int, 'default': 0},
            {'name': 'vram_budget_gb',
             'msg': 'Device-memory budget in GB for the page cache plus the '
                    'kHBM tier. A cell that exceeds it is refused rather than '
                    'left to die after printing its header',
             'type': float, 'default': 7.0},
            {'name': 'timeout_sec',
             'msg': 'Kill a run after this many seconds (0 = no limit)',
             'type': int, 'default': 3600},
            {'name': 'output_dir', 'msg': 'Output directory', 'type': str,
             'default': '/tmp/clio_gnn_train'},
        ]

    def _page_rows(self):
        """Page size in ROWS, which is what the binary actually takes.

        A row is dim*4 bytes. Refusing an indivisible page_kb rather than
        rounding it keeps the swept value and the measured value identical.
        """
        c = self.config
        row_bytes = c['dim'] * 4
        page_bytes = c['page_kb'] * 1024
        if page_bytes % row_bytes != 0:
            raise Exception(
                f"page_kb={c['page_kb']} is {page_bytes} bytes, not a multiple "
                f"of a {row_bytes}-byte row (dim={c['dim']}). Choose a page "
                f"size that holds a whole number of rows.")
        return page_bytes // row_bytes


    def _dataset_mb(self):
        """Size of the workload's dataset in MB."""
        c = self.config
        return c['nodes'] * c['dim'] * 4 / (1024.0 * 1024.0)

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

    def _pages_per_block(self):
        """Pages per block, derived from the total cache budget when one is set.

        `cache_mb` is the TOTAL device page cache; `pages_per_block` is per
        block. The total is what costs VRAM and what a cache-size question is
        about, so it is the swept quantity and the per-block count follows.

        A cell whose per-block share is under one page is REFUSED, not rounded
        up to one page: rounding would silently run a larger cache than the one
        asked for and report it under the small-cache label.
        """
        c = self.config
        cache_mb = self._cache_mb()
        if not cache_mb:
            return c['pages_per_block']
        per_block_kb = cache_mb * 1024.0 / c['gather_blocks']
        pages = int(cache_mb * 1024 //
                    (c['gather_blocks'] * c['page_kb']))
        if pages < 1:
            raise Exception(
                f"cache_mb={cache_mb:.0f} over gather_blocks="
                f"{c['gather_blocks']} leaves {per_block_kb:.1f} KB per block, "
                f"under one {c['page_kb']}KB page. Refused rather than rounded "
                f"up to {c['gather_blocks'] * c['page_kb'] / 1024.0:.0f} MB, "
                f"which would report a larger cache under this cell's label.")
        return pages

    def _actual_cache_mb(self):
        """Cache the run really allocates: blocks x pages x page_kb."""
        c = self.config
        return (self._pages_per_block() * c['gather_blocks'] *
                c['page_kb'] / 1024.0)

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        rows = self._page_rows()
        ppb = self._pages_per_block()
        data_mb = c['nodes'] * c['dim'] * 4 / (1024.0 * 1024.0)
        cache_gb = (c['gather_blocks'] * ppb *
                    c['page_kb']) / (1024.0 * 1024.0)
        budget = c.get('vram_budget_gb', 7.0)
        if cache_gb + c['hbm_mib'] / 1024.0 > budget:
            raise Exception(
                f"page cache needs {cache_gb:.1f} GB (blocks="
                f"{c['gather_blocks']} x slots={ppb} x "
                f"{c['page_kb']}KB) plus a {c['hbm_mib'] / 1024.0:.1f} GB kHBM "
                f"tier, over the {budget:.1f} GB budget.")
        if self._cache_mb():
            got = self._actual_cache_mb()
            if abs(got - self._cache_mb()) > 0.01:
                self.log(f'  NOTE: requested {self._cache_mb():.0f}MB of total cache, '
                         f'allocating {got:.1f}MB ({ppb} pages x '
                         f'{c["gather_blocks"]} blocks x {c["page_kb"]}KB)')
        # HOST-MEMORY GUARD. The trainer materialises the whole feature
        # matrix in host RAM before storing it, and the DRAM tier is pinned
        # (unswappable), so the two add. A 16 GiB matrix with a 20 GiB tier
        # was OOM-killed (exit 137) on a 60 GiB machine -- a SIGKILL with no
        # message in the log, which reads exactly like a hang. Fail here with
        # the arithmetic instead.
        try:
            with open('/proc/meminfo') as f:
                avail_gib = next(int(l.split()[1]) for l in f
                                 if l.startswith('MemAvailable')) / (1024.0 ** 2)
        except (OSError, StopIteration):
            avail_gib = None
        need_gib = data_mb / 1024.0 + c['dram_mib'] / 1024.0
        if avail_gib is not None and need_gib > 0.85 * avail_gib:
            raise Exception(
                f"needs ~{need_gib:.0f} GiB of host RAM ({data_mb / 1024.0:.0f} "
                f"GiB matrix + {c['dram_mib'] / 1024.0:.0f} GiB pinned tier) "
                f"against {avail_gib:.0f} GiB available. Reduce dram_mib to "
                f"just above the spill ({(data_mb - c['hbm_mib']) / 1024.0:.0f} "
                f"GiB) or shrink the matrix.")
        self.log('GNN training over a paged feature matrix')
        self.log(f'  matrix:  {data_mb:.0f}MB ({c["nodes"]} nodes x '
                 f'{c["dim"]}-d)')
        self.log(f'  paging:  page={c["page_kb"]}KB = {rows} rows, '
                 f'cache={ppb} pages/block over '
                 f'{c["gather_blocks"]} blocks ({cache_gb * 1024:.0f}MB total)')
        self.log(f'  tiers:   kHBM {c["hbm_mib"]}MiB, host {c["dram_mib"]}MiB '
                 f'({c["dram_type"]})')
        self.log(f'  codec:   {"uncompressed" if c["compress_lib"] == 0 else c["compress_lib"]}')
        if data_mb < 2 * c['hbm_mib']:
            self.log(f'  WARNING: matrix {data_mb:.0f}MB is under 2x the kHBM '
                     f'tier -- not an out-of-core run')

    def _output_file(self):
        c = self.config
        tag = (f'n{c["nodes"]}_d{c["dim"]}_pg{c["page_kb"]}kb'
               f'_b{c["gather_blocks"]}_sl{self._pages_per_block()}'
               f'_hbm{c["hbm_mib"]}_lib{c["compress_lib"]}')
        return os.path.join(c['output_dir'], f'gnn_{tag}.log')

    def _get_stat(self, stats):
        """Harvest the trainer's numbers into the pipeline results.

        jarvis calls this on a FRESHLY LOADED instance, so the log is re-read
        from disk and the path rebuilt from self.config.

        The trainer prints its numbers as prose ([TRAIN] lines), not a single
        key=value record, so each field is matched individually. epoch_s is the
        training time the trainer measures itself; jarvis's `runtime` column is
        whole-process wall clock and includes ingest, which for a 16 GB matrix
        dominates.
        """
        # Record what the cache axis ACTUALLY was, not what was requested: the
        # requested cache_mb is in the sweep variables already, so carrying the
        # realised per-block count and total lets the post-processing check the
        # two agree instead of assuming it.
        try:
            stats['pages_per_block'] = self._pages_per_block()
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

        def grab(pattern, cast=float):
            m = re.search(pattern, text)
            if m is None:
                return None
            try:
                return cast(m.group(1))
            except ValueError:
                return None

        # "ETERNIA: epoch0 loss=.. -> epoch1 loss=.. acc=.. val_acc=.. (5.00s)"
        found = {
            'epoch_s': grab(r'ETERNIA:.*?\(([0-9.]+)s\)'),
            # GREEDY on purpose. The line is
            #   epoch0 loss=A acc=B -> epoch1 loss=C acc=D val_acc=E (Ts)
            # so a NON-greedy match returns epoch0's numbers while calling
            # them final -- wrong values under a right-sounding name, which no
            # downstream check would catch. Greedy takes the last epoch.
            'final_loss': grab(r'ETERNIA:.*epoch\d+ loss=([0-9.]+)'),
            'final_acc': grab(r'ETERNIA:.*epoch\d+ loss=[0-9.]+ acc=([0-9.]+)'),
            'first_loss': grab(r'ETERNIA:.*?epoch\d+ loss=([0-9.]+)'),
            'val_acc': grab(r'ETERNIA:.*?val_acc=([0-9.]+)'),
            'store_s': grab(r'stored A .*? in ([0-9.]+)s'),
            'stored_mib': grab(r'stored A \d+MiB -> (\d+)MiB', int),
            'compress_ratio': grab(r'stored A .*?\(([0-9.]+)x\)'),
            'faults': grab(r'PAGING: faults=(\d+)', int),
            'evicts': grab(r'evicts=(\d+)', int),
            'get_err': grab(r'get_err=(\d+)', int),
            'put_err': grab(r'put_err=(\d+)', int),
            'hbm_used_mib': grab(r'TIER SPLIT: kHBM (\d+)MiB', int),
            'dram_used_mib': grab(r'TIER SPLIT:.*?DRAM (\d+)MiB', int),
            'gather_s': grab(r'KTIME: gather=([0-9.]+)s'),
            # Achievable H2D bandwidth at THIS page size, probed on an
            # idle device before the runtime starts.
            'memcpy_pin_gbps': grab(r'memcpy_pin_gbps=([0-9.]+)'),
            'memcpy_page_gbps': grab(r'memcpy_page_gbps=([0-9.]+)'),
        }
        for k, v in found.items():
            if v is not None:
                stats[k] = v
        # Always emitted, so a blank means "produced no output" rather than
        # "passed".
        stats['completed'] = 1 if 'ETERNIA:' in text else 0
        stats['read_verify_ok'] = 1 if 'read verify: 8/8' in text else 0
        # kHBM receiving nothing is a misconfiguration, not a result: the DPE
        # ranks by a predicted bandwidth model and has been measured leaving a
        # correctly sized HBM tier empty.
        stats['hbm_used_ok'] = 1 if (found.get('hbm_used_mib') or 0) > 0 else 0

    def start(self):
        # Clear a previous cell's orphan BEFORE anything else: it
        # holds the runtime port and would fail this cell at startup.
        _reap_stale_runtime(self.log, 'test_gpu_vector_gnn_train')
        Which('test_gpu_vector_gnn_train', LocalExecInfo(env=self.mod_env)).run()
        _wait_for_free_vram(self.log,
                            2 * self.config['window_mb'] / 1024.0 +
                            self.config['hbm_mib'] / 1024.0 +
                            (self.config['gather_blocks'] *
                             self._pages_per_block() *
                             self.config['page_kb']) / (1024.0 * 1024.0) + 1.0)
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        rows = self._page_rows()

        env = dict(self.mod_env)
        env.update({
            'CLIO_GNN_SYNTH_N': str(c['nodes']),
            'CLIO_GNN_SYNTH_F': str(c['dim']),
            'CLIO_GNN_CLASSES': str(c['classes']),
            'CLIO_GNN_HIDDEN': str(c['hidden']),
            'CLIO_GNN_EPOCHS': str(c['epochs']),
            'CLIO_GNN_PAGE_ROWS': str(rows),
            'CLIO_GNN_GATHER_BLOCKS': str(c['gather_blocks']),
            'CLIO_GNN_PAGES_PER_BLOCK': str(self._pages_per_block()),
            # Pages, derived from the byte budget so the scratch footprint is
            # constant across the page-size axis.
            'CLIO_GNN_WINDOW': str(max(1, c['window_mb'] * 1024 // c['page_kb'])),
            'CLIO_GNN_HBM_MIB': str(c['hbm_mib']),
            'CLIO_GNN_DRAM_MIB': str(c['dram_mib']),
            'CLIO_GNN_DRAM_TYPE': c['dram_type'],
            'CLIO_GNN_COMPRESS_LIB': str(c['compress_lib']),
            'CLIO_FAULT_HIST': '1',
            'CLIO_GNN_KTIME': '1',
        })
        cmd = 'test_gpu_vector_gnn_train'
        if c['timeout_sec'] > 0:
            cmd = f'timeout {c["timeout_sec"]} {cmd}'

        self.log(f'Running: {cmd} (page={c["page_kb"]}KB = {rows} rows)')
        # The trainer writes its [TRAIN] lines to STDERR; without the redirect
        # the log is empty and every stat silently disappears.
        Exec(f'{cmd} 2>&1 | tee {out}',
             LocalExecInfo(env=env, cwd=c['output_dir'])).run()
        self.log(f'Training completed - results saved to {out}')

    def stop(self):
        """The trainer runs to completion on its own."""
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.startswith('gnn_') or name.endswith('.yaml'):
                os.remove(os.path.join(out, name))
        try:
            os.rmdir(out)
        except OSError:
            pass
