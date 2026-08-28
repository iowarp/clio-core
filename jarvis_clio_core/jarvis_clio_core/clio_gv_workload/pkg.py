"""
Generic driver for the gpu_vector science-workload benchmarks.

ONE package covers every (workload, variant) pair under
context-transfer-engine/adapter/gpu_vector/benchmark/:

  workload:  lammps_md | gmx | lbann | grayscott | kmeans | weights
  variant:   mpi | nccl | nvshmem | bam | paged

The binary is clio_<workload>_<variant>_bench in <build>/bin. The mpi /
nccl / nvshmem variants are SELF-CONTAINED (CUDA + launcher, no clio
runtime); they run under `mpirun -n {nprocs}`. The paged variants host the
clio runtime IN-PROCESS and compose their own CTE stack from --hbm-mb /
--nvme-mb (hbm tier + host-RAM tier + optional file tier) -- an external
daemon cannot service another process's in-kernel faults, so the stack is
by construction per-cell and torn down with it.

Every cell:
  - exports CLIO_PREFAULT (default "0" = pre-fault the WHOLE RAM tier at
    compose, so measurements run against warmed memory, not first-touch
    page population);
  - samples nvidia-smi memory.used at 50 ms while the benchmark runs and
    reports the PEAK as vram_peak_mb -- the empirical footprint, not the
    analytic one some benches print;
  - parses the benchmark's gate lines: gates_pass=1 only when the binary
    printed its ALL-GATES-PASS marker. A cell without it is a failed cell
    even if the process exited 0.

Stats exported per workload (into results.csv):
  all:        completed, gates_pass, vram_peak_mb, bench_ms
  paged:      faults, evicts, puts, get_errors, put_errors (+ gbps where
              the bench prints GB/s)
  lammps_md:  ms_per_step, matom_steps_s, ckpt_n, ckpt_ms_each (paged:
              flush+Copy per checkpoint; mpi: stage+durable per checkpoint),
              vram_mb_analytic (mpi's computed per-rank line), halo_mb_step
  kmeans:     centroid_checksum
  grayscott:  v_checksum
  weights:    checksum
  gmx:        spread_ms, gather_ms
  lbann:      ms_per_step (paged also dense_ms_per_step)
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
import hashlib
import os
import re
import subprocess
import threading
import time


def _reap_stale(log, binary):
    """Kill an orphan of `binary` left by a previous cell (see the kmeans
    pkg for the full pathology: a wedged paged cell keeps the runtime port
    bound and every later cell dies at startup with a TCP bind error)."""
    import signal
    def orphans():
        found = []
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            try:
                exe = os.path.basename(os.readlink('/proc/%s/exe' % entry))
            except OSError:
                continue
            if exe == binary and int(entry) != os.getpid():
                found.append(int(entry))
        return found
    pids = orphans()
    if not pids:
        return
    log('  REAPING %d orphaned %s process(es): %s'
        % (len(pids), binary, ' '.join(str(p) for p in pids)))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    deadline = time.time() + 30
    while time.time() < deadline and orphans():
        time.sleep(1)


class _VramSampler:
    """Poll nvidia-smi memory.used while the benchmark runs; persist the
    peak to a file so the FRESH pkg instance jarvis builds for _get_stat
    can read it (nothing survives from start())."""

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


class ClioGvWorkload(Application):
    """One cell of a gpu_vector workload sweep: build the command line for
    (workload, variant), run it under a timeout with VRAM sampling, and
    harvest the summary lines."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'workload',
             'msg': 'lammps_md | gmx | lbann | grayscott | kmeans | weights',
             'type': str, 'default': 'kmeans'},
            {'name': 'variant',
             'msg': 'mpi | nccl | nvshmem | bam | paged',
             'type': str, 'default': 'mpi'},
            {'name': 'nprocs',
             'msg': 'mpirun ranks (mpi/nccl/nvshmem variants only; all '
                    'ranks share this GPU via cudaSetDevice(rank%%ndev))',
             'type': int, 'default': 1},
            # ---- common kernel geometry ------------------------------
            {'name': 'blocks', 'msg': 'CUDA blocks', 'type': int,
             'default': 0},  # 0 = binary default
            {'name': 'threads', 'msg': 'threads per block', 'type': int,
             'default': 0},
            # ---- problem size (per workload; unused ones ignored) ----
            {'name': 'steps', 'msg': 'steps / iterations / passes',
             'type': int, 'default': 0},
            {'name': 'ckpt', 'msg': 'checkpoint every N steps (lammps_md)',
             'type': int, 'default': 0},
            {'name': 'ckpt_dir',
             'msg': 'durable checkpoint dir (lammps_md mpi; empty = DRAM '
                    'staging only)', 'type': str, 'default': ''},
            {'name': 'lattice', 'msg': 'lammps_md lattice cells per edge',
             'type': int, 'default': 0},
            {'name': 'rebin', 'msg': 'lammps_md resort period', 'type': int,
             'default': 0},
            {'name': 'mesh_k', 'msg': 'gmx PME mesh K (K^3 nodes)',
             'type': int, 'default': 0},
            {'name': 'atoms', 'msg': 'gmx atom count', 'type': int,
             'default': 0},
            {'name': 'hidden', 'msg': 'lbann hidden width', 'type': int,
             'default': 0},
            {'name': 'in_dim', 'msg': 'lbann input width', 'type': int,
             'default': 0},
            {'name': 'out_dim', 'msg': 'lbann output width', 'type': int,
             'default': 0},
            {'name': 'batch', 'msg': 'lbann batch size', 'type': int,
             'default': 0},
            {'name': 'data_mb',
             'msg': 'dataset MB (kmeans/weights/grayscott)', 'type': int,
             'default': 0},
            {'name': 'dims', 'msg': 'kmeans point dimensionality',
             'type': int, 'default': 0},
            {'name': 'clusters', 'msg': 'kmeans k', 'type': int,
             'default': 0},
            {'name': 'flat_pct', 'msg': 'weights flat (compressible) %',
             'type': int, 'default': -1},
            # ---- paged-variant cache / tier stack --------------------
            {'name': 'page_kb', 'msg': 'page size KB', 'type': int,
             'default': 0},
            {'name': 'slots', 'msg': 'per-block page-cache slots',
             'type': int, 'default': 0},
            {'name': 'cache_mb',
             'msg': 'TOTAL page cache MB (paged; derives slots as '
                    'cache_mb/blocks/page_kb and overrides `slots`)',
             'type': int, 'default': 0},
            {'name': 'vram_mb',
             'msg': 'lammps_md paged: --vram-mb cache budget across all '
                    'vectors', 'type': int, 'default': 0},
            {'name': 'hbm_mb', 'msg': 'kHBM tier MB (paged)', 'type': int,
             'default': 0},
            {'name': 'nvme_mb',
             'msg': 'file tier MB (paged; >0 composes the FULL CTE stack '
                    'hbm+ram+file)', 'type': int, 'default': 0},
            {'name': 'nvme_path', 'msg': 'file tier path', 'type': str,
             'default': '/tmp/gv_storage_tier.dat'},
            {'name': 'repeat', 'msg': 'timed repetitions (paged benches)',
             'type': int, 'default': 0},
            {'name': 'cap', 'msg': 'per-bin capacity (lammps_md/gmx paged)',
             'type': int, 'default': 0},
            # ---- harness ---------------------------------------------
            # ---- normalized sweep ladders (combined pipelines) -------
            # A COMBINED sweep needs per-workload axis values (lbann's
            # blocks cap at 64, lammps_md sweeps slots not cache_mb), but
            # jarvis vars are zipped or cartesian, never conditional. The
            # ladder is zipped WITH the workload; the LEVEL is the
            # cartesian axis; the pkg indexes ladder[level-1] and applies
            # it. Cache ladder tokens: plain int = cache_mb, 's<N>' =
            # slots (lammps_md's knob).
            {'name': 'cache_ladder',
             'msg': 'comma list of cache settings, indexed by cache_level '
                    '(e.g. "1024,2048,4096,6144" or "s128,s256,s512,s896")',
             'type': str, 'default': ''},
            {'name': 'cache_level',
             'msg': '1-based index into cache_ladder (0 = ladder unused)',
             'type': int, 'default': 0},
            {'name': 'blocks_ladder',
             'msg': 'comma list of block counts, indexed by blocks_level',
             'type': str, 'default': ''},
            {'name': 'blocks_level',
             'msg': '1-based index into blocks_ladder (0 = unused)',
             'type': int, 'default': 0},
            {'name': 'lr',
             'msg': 'lbann learning rate (0 = binary default). At 6GB-class '
                    'H the default 0.01 diverges to NaN losses (the loss '
                    'gate then fails on nan != nan); 0.0001 stays finite',
             'type': float, 'default': 0.0},
            {'name': 'no_dense',
             'msg': 'gmx paged: skip the dense in-VRAM reference (needed '
                    'for 6GB-class meshes; conservation gate still runs)',
             'type': bool, 'default': False},
            {'name': 'prefault',
             'msg': 'CLIO_PREFAULT value ("0" = pre-fault the whole RAM '
                    'tier: warmed memory; "" = leave population lazy)',
             'type': str, 'default': '0'},
            {'name': 'timeout_sec', 'msg': 'kill the cell after this many '
             'seconds', 'type': int, 'default': 900},
            {'name': 'output_dir', 'msg': 'log directory', 'type': str,
             'default': '/tmp/clio_gv_workload'},
        ]

    # ------------------------------------------------------------------
    # command construction
    # ------------------------------------------------------------------

    def _binary(self):
        c = self.config
        return 'clio_%s_%s_bench' % (c['workload'], c['variant'])

    def _apply_ladders(self):
        """Resolve cache_level/blocks_level through their ladders into the
        concrete cache_mb/slots/blocks settings. Idempotent, and called
        from every entry point that reads the config (start, _tag,
        _get_stat run on SEPARATE pkg instances)."""
        c = self.config
        if c.get('_ladders_applied'):
            return
        c['_ladders_applied'] = True
        if c.get('cache_ladder') and c.get('cache_level'):
            toks = [t.strip() for t in str(c['cache_ladder']).split(',')]
            tok = toks[int(c['cache_level']) - 1]
            if tok.startswith('s'):
                c['slots'] = int(tok[1:])
                c['cache_mb'] = 0
            else:
                c['cache_mb'] = int(tok)
        if c.get('blocks_ladder') and c.get('blocks_level'):
            toks = [t.strip() for t in str(c['blocks_ladder']).split(',')]
            c['blocks'] = int(toks[int(c['blocks_level']) - 1])

    def _slots(self):
        """slots from the total-cache budget when one is given (total is
        what costs VRAM, so it is the swept axis; refuse a share under one
        page rather than silently rounding the cache up)."""
        c = self.config
        if not c['cache_mb']:
            return c['slots']
        blocks = c['blocks'] or 64
        page_kb = c['page_kb'] or 64
        slots = int(c['cache_mb'] * 1024 // (blocks * page_kb))
        if slots < 1:
            raise Exception(
                'cache_mb=%d over blocks=%d x page_kb=%d leaves under one '
                'page per block; refused rather than rounded up'
                % (c['cache_mb'], blocks, page_kb))
        return slots

    def _args(self):
        """The benchmark argv for this (workload, variant), from the recon
        of every binary's parser. Options at their 0/'' sentinel are NOT
        passed, so the binary's own defaults hold."""
        self._apply_ladders()
        c = self.config
        wl, var = c['workload'], c['variant']
        a = []

        def opt(flag, key, transform=None):
            v = c.get(key)
            if v in (0, '', None, -1):
                return
            a.append('%s %s' % (flag, transform(v) if transform else v))

        opt('--blocks', 'blocks')
        opt('--threads', 'threads')
        if wl == 'lammps_md':
            opt('--lattice', 'lattice')
            opt('--steps', 'steps')
            opt('--ckpt', 'ckpt')
            opt('--rebin', 'rebin')
            opt('--cap', 'cap')
            a.append('--md')          # the MD deck is the physics under test
            if var == 'mpi':
                opt('--ckpt-dir', 'ckpt_dir')
            if var == 'paged':
                opt('--page-kb', 'page_kb')
                # --vram-mb is GONE (per-block caches no longer exist; one
                # shared cache per vector, sized for residency). The
                # sweepable cache knob is --slots for the x/v caches; the
                # neighbor-list cache is residency-sized by design.
                if c['slots']:
                    a.append('--slots %d' % c['slots'])
        elif wl == 'gmx':
            opt('--k', 'mesh_k')
            opt('--atoms', 'atoms')
            if var == 'paged':
                # --cap is the vector's TOTAL cache in PAGES (0=resident);
                # cache_mb converts through the page size.
                if c['cache_mb'] and not c['cap']:
                    a.append('--cap %d'
                             % max(1, c['cache_mb'] * 1024
                                   // (c['page_kb'] or 64)))
                opt('--cap', 'cap')
                opt('--page-kb', 'page_kb')
                opt('--repeat', 'repeat')
                opt('--nvme-mb', 'nvme_mb')
                opt('--nvme-path', 'nvme_path')
                if c.get('no_dense'):
                    a.append('--no-dense')
        elif wl == 'lbann':
            opt('--in', 'in_dim')
            opt('--hidden', 'hidden')
            opt('--out', 'out_dim')
            opt('--batch', 'batch')
            opt('--steps', 'steps')
            if c.get('lr'):
                a.append('--lr %g' % c['lr'])
            if var == 'paged':
                if c['cache_mb'] and not c['cap']:
                    a.append('--cap %d'
                             % max(1, c['cache_mb'] * 1024
                                   // (c['page_kb'] or 64)))
                opt('--cap', 'cap')
                opt('--page-kb', 'page_kb')
        elif wl == 'grayscott':
            opt('--steps', 'steps')
            opt('--page-kb', 'page_kb')
            opt('--data-mb', 'data_mb')
            if var == 'paged':
                if self._slots():
                    a.append('--slots %d' % self._slots())
                opt('--hbm-mb', 'hbm_mb')
                opt('--nvme-mb', 'nvme_mb')
                opt('--nvme-path', 'nvme_path')
                opt('--repeat', 'repeat')
        elif wl == 'kmeans':
            opt('--dims', 'dims')
            opt('--clusters', 'clusters')
            opt('--iters', 'steps')
            opt('--data-mb', 'data_mb')
            if var == 'paged':
                if self._slots():
                    a.append('--slots %d' % self._slots())
                opt('--page-kb', 'page_kb')
                opt('--hbm-mb', 'hbm_mb')
                opt('--nvme-mb', 'nvme_mb')
                opt('--nvme-path', 'nvme_path')
                opt('--repeat', 'repeat')
        elif wl == 'weights':
            if var == 'paged':
                # The paged edition has no --passes; --repeat is the
                # re-read knob (best-of is reported).
                if c['steps'] and not c['repeat']:
                    a.append('--repeat %d' % c['steps'])
            else:
                opt('--passes', 'steps')
                opt('--data-mb', 'data_mb')
            if c.get('flat_pct', -1) >= 0:
                a.append('--flat-pct %d' % c['flat_pct'])
            if var == 'paged':
                # The paged edition has no --data-mb: the logical array is
                # blocks x pages x page_kb, so the shared problem size maps
                # onto --pages per block.
                if c['data_mb']:
                    blocks = c['blocks'] or 16
                    page_kb = c['page_kb'] or 64
                    a.append('--pages %d'
                             % max(1, c['data_mb'] * 1024
                                   // (blocks * page_kb)))
                if self._slots():
                    a.append('--slots %d' % self._slots())
                opt('--page-kb', 'page_kb')
                opt('--hbm-mb', 'hbm_mb')
                opt('--nvme-mb', 'nvme_mb')
                opt('--repeat', 'repeat')
        else:
            raise Exception('unknown workload %r' % wl)
        return a

    def _cmd(self):
        c = self.config
        parts = []
        if c['timeout_sec'] > 0:
            # --kill-after: a wedged GPU cell can ignore SIGTERM and then
            # hold the port against every later cell of the sweep.
            parts.append('timeout -k 30 %d' % c['timeout_sec'])
        if c['variant'] in ('mpi', 'nccl', 'nvshmem'):
            parts.append('mpirun -n %d --oversubscribe' % c['nprocs'])
        parts.append(self._binary())
        parts += self._args()
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # file naming: EVERY swept parameter must land in the name or two
    # cells overwrite each other and the sweep still looks complete.
    # ------------------------------------------------------------------

    def _tag(self):
        self._apply_ladders()
        c = self.config
        keys = ('workload', 'variant', 'nprocs', 'blocks', 'threads',
                'steps', 'ckpt', 'lattice', 'mesh_k', 'atoms', 'hidden',
                'batch', 'data_mb', 'dims', 'clusters', 'flat_pct',
                'page_kb', 'slots', 'cache_mb', 'vram_mb', 'hbm_mb',
                'nvme_mb', 'repeat', 'cap', 'rebin')
        blob = '|'.join('%s=%s' % (k, c.get(k)) for k in keys)
        h = hashlib.md5(blob.encode()).hexdigest()[:8]
        return '%s_%s_np%s_%s' % (c['workload'], c['variant'],
                                  c['nprocs'], h)

    def _output_file(self):
        return os.path.join(self.config['output_dir'],
                            self._tag() + '.log')

    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        self.log('gv workload cell: %s' % self._cmd())

    def start(self):
        c = self.config
        _reap_stale(self.log, self._binary())
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        # Warmed memory: CLIO_PREFAULT=0 pre-faults the WHOLE RAM tier at
        # compose (mem_bdev_transport.cc), so timings exclude first-touch
        # page population. Exported for every variant; the CTE-free
        # baselines simply ignore it.
        if c['prefault'] != '':
            self.setenv('CLIO_PREFAULT', str(c['prefault']))
        cmd = self._cmd()
        self.log('Running: %s' % cmd)
        sampler = _VramSampler(out + '.vram')
        sampler.start()
        try:
            # Summary lines go to stderr on half the benches: capture both.
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
    # stats harvesting
    # ------------------------------------------------------------------

    _GATE_MARKS = ('ALL GATES PASS',)
    _GATE_FAIL = ('GATE: FAIL', 'FAIL (')

    def _get_stat(self, stats):
        self._apply_ladders()
        c = self.config
        stats['binary'] = self._binary()
        # The resolved axis values, so a combined sweep's CSV carries the
        # concrete settings next to the abstract levels.
        stats['blocks_resolved'] = c.get('blocks') or 0
        if c.get('slots') and not c.get('cache_mb'):
            stats['cache_setting'] = 'slots=%d' % c['slots']
        elif c.get('cache_mb'):
            stats['cache_setting'] = 'cache_mb=%d' % c['cache_mb']
        out = self._output_file()
        # VRAM peak (empirical, nvidia-smi): "<peak> <baseline>".
        try:
            with open(out + '.vram') as f:
                peak, base = f.read().split()
                stats['vram_peak_mb'] = int(peak)
                stats['vram_baseline_mb'] = int(base)
                stats['vram_delta_mb'] = int(peak) - int(base)
        except (OSError, ValueError):
            pass
        if not os.path.exists(out):
            stats['completed'] = 0
            return
        ansi = re.compile(r'\x1b\[[0-9;]*m')
        try:
            with open(out, errors='replace') as f:
                text = ansi.sub('', f.read())
        except OSError:
            stats['completed'] = 0
            return

        wl, var = c['workload'], c['variant']
        # Gates: pass only on the explicit marker. lammps_md paged/mpi
        # prints per-gate lines instead of one ALL-GATES marker.
        if wl == 'lammps_md':
            gates = re.findall(r'(?:NVE|BALLISTIC|STATICS|RESORT) GATE: '
                               r'(PASS|FAIL)', text)
            stats['gates_pass'] = int(bool(gates) and
                                      all(g == 'PASS' for g in gates))
        else:
            stats['gates_pass'] = int(any(m in text
                                          for m in self._GATE_MARKS))
        stats['completed'] = int(stats['gates_pass'] == 1 or
                                 'iters in' in text or 'steps in' in text)
        # The paged kmeans/grayscott/weights editions print no gate lines
        # -- their correctness contract in a sweep is the machine line plus
        # CLEAN COUNTERS: any get/put error is a lost page. (Checksums are
        # cross-checked against the MPI edition by the analysis, not here:
        # atomics make them order-sensitive at bit level.)
        if stats['gates_pass'] == 0 and var == 'paged' and \
           wl in ('kmeans', 'grayscott', 'weights'):
            m = re.search(r'(?:KMEANS|GRAYSCOTT|GVW) .*mode=.*'
                          r'get_errors=(\d+)', text)
            if m:
                pe = re.search(r'put_errors=(\d+)', text)
                clean = (m.group(1) == '0' and
                         (pe is None or pe.group(1) == '0'))
                stats['gates_pass'] = int(clean)
                stats['completed'] = 1

        def grab(pattern, key, cast=float, last=True):
            ms = re.findall(pattern, text)
            if ms:
                try:
                    stats[key] = cast(ms[-1 if last else 0])
                except ValueError:
                    pass

        # ---- one machine line per bench where it exists --------------
        machine = None
        for ln in text.splitlines():
            if re.match(r'(KMEANS|GRAYSCOTT|GVW) \S*mode=', ln) or \
               ln.startswith(('KMEANS ', 'GRAYSCOTT ', 'GVW ')):
                machine = ln
        if machine:
            for key, out_key in (('ms', 'bench_ms'), ('GB/s', 'gbps'),
                                 ('centroid_checksum', 'centroid_checksum'),
                                 ('v_checksum', 'v_checksum'),
                                 ('checksum', 'checksum'),
                                 ('faults', 'faults'), ('evicts', 'evicts'),
                                 ('puts', 'puts'),
                                 ('get_errors', 'get_errors'),
                                 ('put_errors', 'put_errors')):
                m = re.search(r'(?:^|\s)' + re.escape(key) + r'=([0-9.]+)',
                              machine)
                if m and out_key not in stats:
                    v = float(m.group(1))
                    stats[out_key] = int(v) if v.is_integer() else v

        # ---- per-workload prose lines --------------------------------
        if wl == 'lammps_md':
            grab(r'(\d+) steps in', 'steps_run', int)
            grab(r'steps in ([0-9.]+) ms', 'bench_ms')
            grab(r'\(([0-9.]+) ms/step', 'ms_per_step')
            grab(r'([0-9.]+) Matom-steps/s', 'matom_steps_s')
            grab(r'per-rank VRAM: .*= ([0-9.]+) MB', 'vram_mb_analytic')
            grab(r'halo [0-9.]+ GB in \d+ exchanges \(([0-9.]+) MB/step',
                 'halo_mb_step')
            grab(r'checkpoints: (\d+)', 'ckpt_n', int)
            if var == 'paged':
                grab(r'flush \+ vector\.Copy ([0-9.]+) ms each',
                     'ckpt_ms_each')
                grab(r'x faults=(\d+)', 'faults', int)
                grab(r'x .*evicts=(\d+)', 'evicts', int)
            else:
                grab(r'total ([0-9.]+) ms = [0-9.]+% on top',
                     'ckpt_ms_total')
                grab(r'stage D2H ([0-9.]+) ms each', 'ckpt_stage_ms_each')
                grab(r'durable write ([0-9.]+) ms each',
                     'ckpt_durable_ms_each')
        elif wl == 'gmx':
            grab(r'spread ([0-9.]+) ms', 'spread_ms')
            grab(r'gather\+sum ([0-9.]+) ms', 'gather_ms')
            if 'spread_ms' in stats and 'gather_ms' in stats and \
               'bench_ms' not in stats:
                stats['bench_ms'] = round(stats['spread_ms'] +
                                          stats['gather_ms'], 2)
        elif wl == 'lbann':
            grab(r'paged ([0-9.]+) ms/step', 'ms_per_step')
            grab(r'dense ([0-9.]+) ms/step', 'dense_ms_per_step')
            grab(r'(\d+) steps in', 'steps_run', int)
            grab(r'steps in ([0-9.]+) ms', 'bench_ms')
            if 'ms_per_step' not in stats and 'bench_ms' in stats and \
               c.get('steps'):
                stats['ms_per_step'] = round(
                    stats['bench_ms'] / c['steps'], 3)
        elif wl == 'grayscott':
            grab(r'(\d+) steps in', 'steps_run', int)
            grab(r'steps in ([0-9.]+) ms', 'bench_ms')
            grab(r'v_checksum=([0-9.]+)', 'v_checksum')
        elif wl == 'kmeans':
            grab(r'(\d+) iters in', 'iters_run', int)
            grab(r'iters in ([0-9.]+) ms', 'bench_ms')
            grab(r'centroid_checksum=([0-9.]+)', 'centroid_checksum')
        elif wl == 'weights':
            grab(r'(\d+) passes,', 'passes_run', int)
            grab(r'ranks, ([0-9.]+) ms', 'bench_ms')
