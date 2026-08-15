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

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        if c['slots'] < MIN_SLOTS:
            raise Exception(
                f"slots={c['slots']} but the stencil holds {MIN_SLOTS} planes "
                f"at once (z-1,z,z+1 of u and v, plus both outputs). A smaller "
                f"cache would let a plane still being read be evicted.")
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
        cache_gb = (c['blocks'] * c['slots'] * c['page_kb']) / (1024.0 * 1024.0)
        budget_gb = c.get('vram_budget_gb', 7.0)
        if cache_gb + c['hbm_mb'] / 1024.0 > budget_gb:
            raise Exception(
                f"page cache needs {cache_gb:.1f} GB (blocks={c['blocks']} x "
                f"slots={c['slots']} x {c['page_kb']}KB) plus a "
                f"{c['hbm_mb'] / 1024.0:.1f} GB kHBM tier, over the "
                f"{budget_gb:.1f} GB budget. Reduce blocks, slots or page_kb.")
        self.log('GPU vector Gray-Scott configured')
        self.log(f'  grid:      one {c["page_kb"]}KB plane = {plane_elems} '
                 f'cells, {nz} planes deep')
        self.log(f'  total:     {c["data_mb"]}MB over 4 fields '
                 f'(u, v, u_next, v_next)')
        self.log(f'  parallel:  {c["blocks"]} blocks x {c["threads"]} threads, '
                 f'cache {c["slots"]} planes/block')
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
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{c["slots"]}'
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
            if ln.startswith('GRAYSCOTT blocks=') and 'ms=' in ln:
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
                         ('put_errors', 'put_errors')):
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
        Which('clio_gpu_vector_grayscott_bench',
              LocalExecInfo(env=self.mod_env)).run()
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()

        cmd = ['clio_gpu_vector_grayscott_bench',
               f'--blocks {c["blocks"]}', f'--threads {c["threads"]}',
               f'--slots {c["slots"]}', f'--steps {c["steps"]}',
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
