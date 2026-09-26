"""
LBANN eternia paged-GEMM package

Drives LBANN with LBANN_ETERNIA_FC: a fully-connected layer whose weight
matrix is held out of core and paged into the GPU by the forward and backward
kernels. The distinguishing access pattern is RE-READ: every pass walks the
whole weight matrix, so every fetched byte is used and page size pays
maximally -- the opposite end of the spectrum from a scattered GNN gather.

The layer width sets the dataset: W is width x width floats, so `width` is
the size knob and data_mb is derived from it rather than set independently,
which keeps the two from disagreeing.

BINARY LOCATION. The workload binaries are large builds from the forks under
external/ and are NOT on PATH by default. `bin_dir` names the build root, and
falls back to the ETERNIA_BIN_DIR environment variable so a pipeline does not
have to hard-code one machine's scratch path.

CORRECTNESS IS A REPORTED STAT, not an assumption. These three workloads have
a known exact answer, so every cell checks its result against it and reports
`correct`. A page-size or block-count sweep that only recorded milliseconds
would happily rank a configuration that computes the wrong answer fastest --
and this project has already shipped paged paths that reported success while
losing data.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
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



class ClioEterniaLbann(Application):
    """LBANN with the paged fully-connected layer."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'bin_dir',
             'msg': 'Root holding the workload builds (lmp-build/, gmx-build/, '
                    'lbann-build/, clio-inst/). Defaults to $ETERNIA_BIN_DIR',
             'type': str, 'default': ''},
            {'name': 'page_kb', 'msg': 'Page size in KB', 'type': int,
             'default': 256},
            {'name': 'blocks', 'msg': 'CUDA blocks driving the paged kernel',
             'type': int, 'default': 32},
            {'name': 'slots',
             'msg': 'Resident pages per block. Overridden when cache_mb or '
                    'cache_frac is set',
             'type': int, 'default': 8},
            {'name': 'cache_mb',
             'msg': 'TOTAL page cache across all blocks, in MB. When set it '
                    'overrides `slots`, which is derived as '
                    'cache_mb/blocks/page_kb. This is the axis worth sweeping: '
                    '`slots` is per block, so holding it fixed while blocks '
                    'varies changes the cache by the same factor and confounds '
                    'the two axes',
             'type': int, 'default': 0},
            {'name': 'cache_frac',
             'msg': 'Page cache as a fraction of the dataset. Takes precedence '
                    'over cache_mb, because hit rate follows COVERAGE rather '
                    'than an absolute byte count',
             'type': float, 'default': 0.0},
            {'name': 'timeout_sec',
             'msg': 'Kill a run after this many seconds (0 = no limit). A GPU '
                    'kernel can wedge and never return; without a limit one '
                    'stuck cell stalls the rest of a sweep',
             'type': int, 'default': 1800},
            {'name': 'vram_budget_gb',
             'msg': 'Device-memory budget in GB for the page cache. A cell '
                    'that exceeds it is refused rather than left to die after '
                    'printing its header',
             'type': float, 'default': 7.0},
            {'name': 'checkpoint_epochs',
             'msg': 'Dump the weights every N epochs (0 = never). This is the '
                    'cost the paged path exists to avoid: the stock path must '
                    'serialise W to disk, while under own_weights the CTE copy '
                    'IS the current W and is already durable if the tier is '
                    'file-backed',
             'type': int, 'default': 0},
            {'name': 'ckpt_via_cte',
             'msg': 'When own_weights is on, treat the CTE copy as the '
                    'checkpoint and skip the file dump. Set false to make both '
                    'paths write the same files, which measures paging '
                    'overhead against an unchanged I/O cost',
             'type': bool, 'default': True},
            {'name': 'baseline',
             'msg': 'Run the application STOCK kernel instead of the paged '
                    'one, via ETERNIA_BASELINE. Same binary, same input, one '
                    'variable -- so a comparison changes only the kernel',
             'type': bool, 'default': False},
            {'name': 'clio_conf',
             'msg': 'Clio server config for the in-process runtime. Each fork '
                    'ships one next to its test harness; the hbm tier size in '
                    'it is a GPU allocation taken at startup, so it counts '
                    'against the same VRAM budget as the page cache',
             'type': str, 'default': '/home/llogan/Documents/Projects/iowarp/core/external/lbann/src/eternia/test/clio.yaml'},
            {'name': 'output_dir', 'msg': 'Output directory', 'type': str,
             'default': '/tmp/clio_eternia_lbann'},
            {'name': 'width',
             'msg': 'Hidden layer width. W is width x width floats, so this '
                    'sets the paged dataset: 8192 -> 256 MiB, 16384 -> 1 GiB',
             'type': int, 'default': 8192},
            {'name': 'epochs', 'msg': 'Training epochs', 'type': int,
             'default': 2},
            {'name': 'mini_batch', 'msg': 'Mini-batch size', 'type': int,
             'default': 256},
            {'name': 'host_weights',
             'msg': 'Keep the linearity off the GPU entirely (Hydrogen holds '
                    'it on the host). Required for the run to be out-of-core '
                    'in storage as well as compute',
             'type': bool, 'default': True},
            {'name': 'own_weights',
             'msg': 'Let the paged store own the weights and drive the SGD '
                    'update, so the gradient never round-trips through a '
                    'resident copy. Plain SGD with zero momentum only -- the '
                    'layer refuses anything stateful',
             'type': bool, 'default': False},
        ]

    def _dataset_mb(self):
        """W in MB. float32, width x width."""
        w = float(self.config['width'])
        return w * w * 4.0 / (1024.0 * 1024.0)

    def _ckpt_is_free(self):
        """True when this configuration's checkpoint costs nothing extra.

        Only the paged path with own_weights qualifies: there the CTE holds the
        current W, so a file dump would be a second copy of something already
        written. Everything else -- the stock path, and the paged path that
        still lets LBANN own the weights -- has to serialise.
        """
        c = self.config
        return (not c['baseline']) and c['own_weights'] and c['ckpt_via_cte']

    def _prototext(self):
        return os.path.join(self.config['output_dir'], 'model.prototext')

    def _write_prototext(self):
        """Generate the model at this cell's width.

        Written per cell rather than shipped as a fixture because `width` is
        the size axis; a fixed prototext would silently run one size for every
        point on the sweep.
        """
        c = self.config
        # The print callback is not optional: without it LBANN trains
        # correctly and reports NOTHING -- no "objective function" line -- so
        # the cell harvests no result and reads as a failed run. The objective
        # is this workload's correctness signal.
        #
        # Comments must not be added to the generated text: prototext takes
        # `#`, and a `//` line fails the whole parse with "Unable to parse
        # prototext from stream", which looks like a model error rather than a
        # comment syntax error.
        # The weights dump is skipped only when the CTE copy really is the
        # live W -- that is own_weights, paged, with ckpt_via_cte left on.
        # Any other combination writes the files, because claiming a
        # checkpoint that nothing persisted would be the whole result invented.
        ckpt = ''
        if c['checkpoint_epochs'] > 0 and not self._ckpt_is_free():
            ckpt = ('\n  callback { dump_weights { directory: "%s" '
                    'epoch_interval: %d } }'
                    % (os.path.join(c['output_dir'], 'ckpt'),
                       c['checkpoint_epochs']))
        with open(self._prototext(), 'w') as f:
            f.write(f'''trainer {{ mini_batch_size: {c["mini_batch"]} }}
data_reader {{
  reader {{
    name: "synthetic"
    role: "train"
    shuffle: false
    num_samples: {c["mini_batch"]}
    num_labels: 4
    synth_dimensions: "{c["width"]}"
    validation_fraction: 0.0
    absolute_sample_count: 0
    fraction_of_data_to_use: 1.0
  }}
}}
model {{
  data_layout: "data_parallel"
  num_epochs: {c["epochs"]}

  layer {{ name: "data" children: "fc1" data_layout: "data_parallel"
          input {{ data_field: "samples" }} }}
  layer {{ name: "lbl" data_layout: "data_parallel"
          input {{ data_field: "labels" }} }}
  layer {{ parents: "data" name: "fc1" data_layout: "data_parallel"
          fully_connected {{ num_neurons: {c["width"]} has_bias: false transpose: true }} }}
  layer {{ parents: "fc1" name: "relu1" data_layout: "data_parallel" relu {{}} }}
  layer {{ parents: "relu1" name: "fc2" data_layout: "data_parallel"
          fully_connected {{ num_neurons: 4 has_bias: false transpose: true }} }}
  layer {{ parents: "fc2" name: "prob" data_layout: "data_parallel" softmax {{}} }}
  layer {{ parents: "prob" parents: "lbl" name: "ce" data_layout: "data_parallel" cross_entropy {{}} }}
  objective_function {{ layer_term {{ layer: "ce" }} }}
  callback {{ print {{ interval: 1 }} }}{ckpt}
}}
optimizer {{ sgd {{ learn_rate: 0.01 momentum: 0.0 }} }}
''')

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        slots = self._slots()
        self._check_vram()
        pages = int(self._dataset_mb() * 1024 // c['page_kb'])
        if pages < c['blocks']:
            raise Exception(
                f"W is {self._dataset_mb():.0f} MB = {pages} pages at "
                f"{c['page_kb']}KB, fewer than the {c['blocks']} blocks. Most "
                f"blocks would own no page and the cell would measure launch "
                f"overhead rather than paging.")
        self._write_prototext()
        self.log('LBANN eternia paged GEMM configured')
        self.log(f'  weights:   {c["width"]}x{c["width"]} = '
                 f'{self._dataset_mb():.0f} MB, {pages} pages of {c["page_kb"]}KB')
        self.log(f'  parallel:  {c["blocks"]} blocks, {slots} pages/block '
                 f'({self._actual_cache_mb():.0f} MB cache, '
                 f'{100.0 * self._actual_cache_mb() / self._dataset_mb():.1f}% '
                 f'of W)')
        self.log(f'  mode:      host_weights={c["host_weights"]} '
                 f'own_weights={c["own_weights"]}')

    def _output_file(self):
        c = self.config
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{self._slots()}'
               f'_w{c["width"]}_ep{c["epochs"]}')
        mode = 'base' if c['baseline'] else 'paged'
        return os.path.join(c['output_dir'], f'lbann_{mode}_{tag}.log')

    def _get_stat(self, stats):
        # Stats are NAMESPACED. The three eternia packages share a
        # pipeline and would otherwise write the same keys -- faults,
        # completed, correct -- into one row, where the last package to
        # report silently overwrites the others and the table reads as
        # though two of the three workloads failed.
        P = 'lb_'
        try:
            stats[P + 'slots'] = self._slots()
            stats[P + 'cache_mb_actual'] = round(self._actual_cache_mb(), 1)
            stats[P + 'dataset_mb'] = round(self._dataset_mb(), 1)
        except Exception:
            return
        text = self._read_log(self._output_file())
        if text is None:
            return
        objs = re.findall(r'objective function : ([0-9.]+)', text)
        if not objs:
            stats[P + 'completed'] = 0
            return
        stats[P + 'completed'] = 1
        stats[P + 'mode'] = 'base' if self.config['baseline'] else 'paged'
        # LBANN's own timing table, printed by both paths. Column 3 is the
        # total over the run; whole-process wall clock would also include
        # data-reader setup and the Clio runtime start, which only the paged
        # path pays for.
        for ln in text.splitlines():
            for tag, out in (('forward prop', 'fwd_s'), ('back prop', 'bwd_s')):
                if ln.strip().startswith(tag):
                    parts = [x.strip() for x in ln.split('|')]
                    if len(parts) >= 3:
                        try:
                            stats[P + out] = float(parts[2])
                        except ValueError:
                            pass
        if P + 'fwd_s' in stats and P + 'bwd_s' in stats:
            stats[P + 'gemm_s'] = round(stats[P + 'fwd_s'] + stats[P + 'bwd_s'], 4)
        # How much this cell actually wrote, and whether it paid for it.
        ck = os.path.join(self.config['output_dir'], 'ckpt')
        total = 0
        for root, _dirs, files in os.walk(ck):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        stats[P + 'ckpt_gb'] = round(total / (1024.0 ** 3), 3)
        stats[P + 'ckpt_free'] = int(self._ckpt_is_free())
        stats[P + 'objective_first'] = float(objs[0])
        stats[P + 'objective_last'] = float(objs[-1])
        stats[P + 'epochs_seen'] = len(objs)
        # Paging counters SUMMED over every layer that reported. Taking the
        # last line instead reports whichever layer ran last, and in this model
        # that is fc2 -- an 8192x4 matrix whose 3 faults stood in for fc1's
        # 3072, making the cache axis look completely flat when it was not.
        totals = {}
        for ln in text.splitlines():
            if not ln.startswith('[eternia] faults='):
                continue
            for key in ('faults', 'evicts', 'get_errors'):
                m = re.search(r'(?:^|\s)' + key + r'=([0-9]+)', ln)
                if m:
                    totals[key] = totals.get(key, 0) + int(m.group(1))
        for key, v in totals.items():
            stats[P + key] = v
        if stats.get(P + 'faults'):
            stats[P + 'paged_mb'] = round(
                stats[P + 'faults'] * self.config['page_kb'] / 1024.0, 1)
        # CORRECTNESS. A paged run must reach the same objective as El::Gemm;
        # get_errors must be zero. Reported rather than asserted so a failing
        # cell still appears in the table with its timing, instead of vanishing.
        stats[P + 'correct'] = int(stats.get(P + 'get_errors', 0) == 0
                               and stats[P + 'epochs_seen'] == self.config['epochs'])

    def start(self):
        c = self.config
        b = self._bin_dir()
        binary = os.path.join(b, 'lbann-build', 'bin', 'lbann')
        _reap_stale_runtime(self.log, 'lbann')
        _wait_for_free_vram(self.log, self._actual_cache_mb() / 1024.0 + 1.0)
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        env = dict(self.mod_env)
        if c['clio_conf']:
            env['CLIO_SERVER_CONF'] = c['clio_conf']
        if c['baseline']:
            env['ETERNIA_BASELINE'] = '1'
        env['LBANN_ETERNIA_FC'] = '1'
        env['LBANN_ETERNIA_STATS'] = '1'
        env['LBANN_ETERNIA_PAGE_KB'] = str(c['page_kb'])
        env['LBANN_ETERNIA_BLOCKS'] = str(c['blocks'])
        env['LBANN_ETERNIA_SLOTS'] = str(self._slots())
        # host_weights and own_weights describe the PAGED path and are
        # meaningless without it. Passing them under ETERNIA_BASELINE puts the
        # linearity on the host while the activations stay on the GPU, and
        # El::Gemm then aborts with "Must call gemm with matrices on same
        # device" -- a baseline that fails outright, which reads as the stock
        # path being broken rather than as a bad flag combination.
        if not c['baseline']:
            if c['host_weights']:
                env['LBANN_ETERNIA_FC_HOST_WEIGHTS'] = '1'
            if c['own_weights']:
                env['LBANN_ETERNIA_FC_OWN_WEIGHTS'] = '1'
        cmd = [binary, f'--prototext={self._prototext()}']
        if c['timeout_sec'] > 0:
            cmd.insert(0, f'timeout {c["timeout_sec"]}')
        self.log(f'Running: {" ".join(cmd)}')
        # Stats go to STDERR; without the redirect the log is empty and every
        # paging counter silently disappears.
        Exec(f'{" ".join(cmd)} 2>&1 | tee {out}',
             LocalExecInfo(env=env, cwd=c['output_dir'])).run()
        self.log(f'Run complete - results in {out}')

    def _bin_dir(self):
        d = self.config.get('bin_dir') or os.environ.get('ETERNIA_BIN_DIR', '')
        if not d:
            raise Exception(
                'bin_dir is unset and ETERNIA_BIN_DIR is not in the '
                'environment; this package cannot find the workload build.')
        return d

    def _cache_mb(self):
        c = self.config
        frac = c.get('cache_frac') or 0.0
        if frac:
            return self._dataset_mb() * frac
        return c.get('cache_mb') or 0

    def _slots(self):
        """Pages per block, derived from the total cache budget when one is set.

        Refused rather than rounded up when the per-block share is under one
        page: rounding would silently run a LARGER cache than the one asked for
        and report it under the small-cache label.
        """
        c = self.config
        cache_mb = self._cache_mb()
        if not cache_mb:
            return c['slots']
        slots = int(cache_mb * 1024 // (c['blocks'] * c['page_kb']))
        if slots < 1:
            raise Exception(
                f"cache_mb={cache_mb:.0f} over blocks={c['blocks']} at "
                f"{c['page_kb']}KB leaves under one page per block. Refused "
                f"rather than rounded up, which would report a larger cache "
                f"under this cell's label.")
        return slots

    def _actual_cache_mb(self):
        c = self.config
        return self._slots() * c['blocks'] * c['page_kb'] / 1024.0

    def _check_vram(self):
        c = self.config
        cache_gb = self._actual_cache_mb() / 1024.0
        budget = c.get('vram_budget_gb', 7.0)
        if cache_gb > budget:
            raise Exception(
                f"page cache needs {cache_gb:.1f} GB (blocks={c['blocks']} x "
                f"slots={self._slots()} x {c['page_kb']}KB), over the "
                f"{budget:.1f} GB budget. Reduce blocks, slots or page_kb.")

    def _read_log(self, path):
        if not os.path.exists(path):
            return None
        ansi = re.compile(r'\x1b\[[0-9;]*m')
        try:
            with open(path, 'r', errors='replace') as f:
                return ansi.sub('', f.read())
        except OSError:
            return None

    def stop(self):
        """The workload runs to completion on its own."""
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.endswith('.log'):
                os.remove(os.path.join(out, name))
        try:
            os.rmdir(out)
        except OSError:
            pass
