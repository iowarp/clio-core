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
- slots         the per-block page cache
- flat_pct      fraction of the matrix that is low-entropy, which sets how much
                a compressor can actually win

Assumes clio_gpu_vector_weights_bench is on PATH (it lives in <build>/bin).
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re


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
            {'name': 'slots', 'msg': 'Per-block page cache slots',
             'type': int, 'default': 8},
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

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.log('GPU vector weights benchmark configured')
        self.log(f'  blocks x threads: {self.config["blocks"]} x '
                 f'{self.config["threads"]}')
        self.log(f'  kHBM tier:        {self.config["hbm_mb"]}MB'
                 f'{" (HBM ONLY)" if self.config["hbm_only"] else ""}')
        self.log(f'  working set:      {self.config["pages"]} pages/block, '
                 f'cache {self.config["slots"]} slots')
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
        tag = (f'b{c["blocks"]}_hbm{c["hbm_mb"]}_pg{c["pages"]}'
               f'_sl{c["slots"]}_flat{c["flat_pct"]}'
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
               'rounds': 'rounds', 'slots': 'slots', 'pages': 'pages'}
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
        if stats.get('logical_mb') and stats.get('stored_mb'):
            stats['compress_ratio'] = round(
                stats['logical_mb'] / stats['stored_mb'], 3)

    def start(self):
        Which('clio_gpu_vector_weights_bench',
              LocalExecInfo(env=self.mod_env)).run()
        os.makedirs(self.config['output_dir'], exist_ok=True)
        out = self._output_file()
        c = self.config

        cmd = ['clio_gpu_vector_weights_bench',
               f'--blocks {c["blocks"]}', f'--threads {c["threads"]}',
               f'--rt-threads {c["rt_threads"]}', f'--hbm-mb {c["hbm_mb"]}',
               f'--pages {c["pages"]}', f'--slots {c["slots"]}',
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
