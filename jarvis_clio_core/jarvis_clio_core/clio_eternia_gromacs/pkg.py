"""
GROMACS eternia paged nonbonded package

Drives gmx mdrun with GMX_ETERNIA_NB: the paged Lennard-Jones kernel runs on
nbnxm's own cluster pair list, with the list and coordinates held out of core
and faulted into the GPU from inside the kernel.

The access pattern is a CLUSTER GATHER: an i-supercluster pulls in scattered
j-clusters, so a large page over-fetches -- between the sequential re-read of
a weight matrix and the fully scattered rows of a GNN gather.

The system is a perfect argon lattice, which is what makes the correctness
check exact: the per-atom LJ energy is a finite lattice sum, computed here in
double precision rather than taken from a previous run.

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



class ClioEterniaGromacs(Application):
    """GROMACS with the paged nonbonded kernel."""

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
            {'name': 'clio_conf',
             'msg': 'Clio server config for the in-process runtime. Each fork '
                    'ships one next to its test harness; the hbm tier size in '
                    'it is a GPU allocation taken at startup, so it counts '
                    'against the same VRAM budget as the page cache',
             'type': str, 'default': '/home/llogan/Documents/Projects/iowarp/core/external/gromacs/src/gromacs/eternia/test/clio.yaml'},
            {'name': 'output_dir', 'msg': 'Output directory', 'type': str,
             'default': '/tmp/clio_eternia_gromacs'},
            {'name': 'cells',
             'msg': 'Lattice cells per side; atoms = cells^3. This is the size '
                    'knob: 12 -> 1728 atoms, 20 -> 8000, 40 -> 64000',
             'type': int, 'default': 20},
            {'name': 'steps', 'msg': 'MD steps', 'type': int, 'default': 2},
            {'name': 'ntomp', 'msg': 'OpenMP threads for mdrun', 'type': int,
             'default': 2},
            {'name': 'spacing', 'msg': 'Lattice spacing in nm', 'type': float,
             'default': 0.34},
            {'name': 'sigma', 'msg': 'LJ sigma in nm', 'type': float,
             'default': 0.3345},
            {'name': 'epsilon', 'msg': 'LJ epsilon in kJ/mol', 'type': float,
             'default': 1.045128},
            {'name': 'rcut', 'msg': 'Cutoff in nm', 'type': float,
             'default': 1.0},
        ]

    def _atoms(self):
        return self.config['cells'] ** 3

    def _dataset_mb(self):
        """Coordinates plus the pair list, in MB.

        xq is 4 floats per atom. The list is counted from the measured ratio of
        packed j-cluster entries to atoms (~0.15 entries/atom at these sizes),
        which is approximate -- it is used only to scale `cache_frac`, never to
        report a result.
        """
        xq_mb = self._atoms() * 4 * 4 / (1024.0 * 1024.0)
        list_mb = self._atoms() * 0.15 * 8 * 4 / (1024.0 * 1024.0)
        return xq_mb + list_mb

    def _exact_per_atom(self):
        """Exact per-atom LJ energy of the lattice, in double precision.

        Every atom of a perfect lattice sits at an identical environment, so
        the per-atom energy is a finite sum over the neighbours inside the
        cutoff -- an exact reference, and independent of anything the kernel
        does. Computed here rather than hard-coded so it stays correct when
        spacing, sigma or the cutoff are swept.
        """
        import math
        c = self.config
        sp, sig, eps, rc = (c['spacing'], c['sigma'], c['epsilon'], c['rcut'])
        lim = int(rc / sp) + 2
        terms = []
        for i in range(-lim, lim + 1):
            for j in range(-lim, lim + 1):
                for k in range(-lim, lim + 1):
                    if i == j == k == 0:
                        continue
                    r2 = (sp * sp) * (i * i + j * j + k * k)
                    if r2 < rc * rc:
                        s6 = (sig * sig / r2) ** 3
                        terms.append(4.0 * eps * (s6 * s6 - s6))
        terms.sort(key=abs)
        return 0.5 * math.fsum(terms)

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        slots = self._slots()
        self._check_vram()
        if c['rcut'] >= c['cells'] * c['spacing'] / 2.0:
            raise Exception(
                f"cutoff {c['rcut']} nm is at least half the "
                f"{c['cells'] * c['spacing']:.2f} nm box, so the minimum image "
                f"convention does not hold and the exact lattice reference "
                f"would not apply.")
        self.log('GROMACS eternia paged nonbonded configured')
        self.log(f'  system:    {self._atoms()} atoms, '
                 f'{c["cells"] * c["spacing"]:.2f} nm box, '
                 f'{self._dataset_mb():.1f} MB paged')
        self.log(f'  parallel:  {c["blocks"]} blocks, {slots} pages/block '
                 f'({self._actual_cache_mb():.1f} MB cache)')
        self.log(f'  reference: {self._exact_per_atom():.6f} kJ/mol per atom '
                 f'(exact lattice sum)')

    def _tpr(self):
        return os.path.join(self.config['output_dir'], 'argon.tpr')

    def _output_file(self):
        c = self.config
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{self._slots()}'
               f'_n{self._atoms()}_st{c["steps"]}')
        return os.path.join(c['output_dir'], f'gromacs_{tag}.log')

    def _prepare_inputs(self, gmx):
        """Generate the lattice and run grompp.

        Done per cell because `cells` is a size axis, and grompp writes the
        atom count into the tpr; a shared tpr would run one size for the whole
        sweep while the label said otherwise.
        """
        c = self.config
        d = c['output_dir']
        n = self._atoms()
        side = c['cells']
        with open(os.path.join(d, 'conf.gro'), 'w') as f:
            f.write('argon %d\n%d\n' % (n, n))
            i = 0
            for x in range(side):
                for y in range(side):
                    for z in range(side):
                        i += 1
                        w = i % 100000
                        f.write('%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n'
                                % (w, 'AR', 'Ar', w, x * c['spacing'],
                                   y * c['spacing'], z * c['spacing']))
            box = side * c['spacing']
            f.write('%10.5f%10.5f%10.5f\n' % (box, box, box))
        with open(os.path.join(d, 'topol.top'), 'w') as f:
            f.write('[ defaults ]\n1 2 yes 0.5 0.8333\n\n'
                    '[ atomtypes ]\n'
                    'Ar 18 39.948 0.000 A %.5f %.6f\n\n'
                    '[ moleculetype ]\nAr 1\n\n'
                    '[ atoms ]\n1 Ar 1 AR Ar 1 0.0 39.948\n\n'
                    '[ system ]\nargon\n\n'
                    '[ molecules ]\nAr   %d\n'
                    % (c['sigma'], c['epsilon'], n))
        with open(os.path.join(d, 'md.mdp'), 'w') as f:
            f.write('integrator      = md\ndt              = 0.001\n'
                    'nsteps          = %d\nnstlog          = 1\n'
                    'nstcalcenergy   = 1\ncutoff-scheme   = Verlet\n'
                    'nstlist         = 10\nverlet-buffer-tolerance = -1\n'
                    'rlist           = %g\ncoulombtype     = cut-off\n'
                    'rcoulomb        = %g\nvdwtype         = cut-off\n'
                    'vdw-modifier    = none\nrvdw            = %g\n'
                    'DispCorr        = no\npbc             = xyz\n'
                    'gen-vel         = yes\ngen-temp        = 100\n'
                    'gen-seed        = 12345\n'
                    % (c['steps'], c['rcut'], c['rcut'], c['rcut']))
        Exec(f'{gmx} grompp -f md.mdp -c conf.gro -p topol.top '
             f'-o {self._tpr()} -maxwarn 5 > grompp.log 2>&1',
             LocalExecInfo(env=self.mod_env, cwd=d)).run()

    def _get_stat(self, stats):
        # Stats are NAMESPACED. The three eternia packages share a
        # pipeline and would otherwise write the same keys -- faults,
        # completed, correct -- into one row, where the last package to
        # report silently overwrites the others and the table reads as
        # though two of the three workloads failed.
        P = 'gmx_'
        try:
            stats[P + 'slots'] = self._slots()
            stats[P + 'cache_mb_actual'] = round(self._actual_cache_mb(), 1)
            stats[P + 'atoms'] = self._atoms()
            stats[P + 'dataset_mb'] = round(self._dataset_mb(), 1)
        except Exception:
            return
        text = self._read_log(self._output_file())
        if text is None:
            return
        # INVALID_E= rather than E= means failed page reads; it must not parse
        # as an energy, which is why the hook spells it differently.
        es = re.findall(r'(?:^|[^_])E=(-[0-9.]+)', text)
        if not es:
            stats[P + 'completed'] = 0
            if 'RESULT INVALID' in text:
                stats[P + 'get_errors'] = 1
            return
        stats[P + 'completed'] = 1
        first = float(es[0])
        stats[P + 'lj_sr_step0'] = first
        stats[P + 'per_atom'] = round(first / self._atoms(), 6)
        exact = self._exact_per_atom()
        stats[P + 'per_atom_exact'] = round(exact, 6)
        stats[P + 'rel_err'] = abs(stats[P + 'per_atom'] - exact) / abs(exact)
        for key, out in (('xq faults', 'faults'), ('xq evicts', 'evicts'),
                         ('list faults', 'list_faults'),
                         ('get_err', 'get_errors'), ('pairs', 'pairs')):
            m = re.search(re.escape(key) + r'=([0-9]+)', text)
            if m:
                stats[P + out] = int(m.group(1))
        m = re.search(r'force check: max\|f\|=([0-9.eE+-]+)', text)
        if m:
            # Step 0 is the perfect lattice, where the exact force is zero.
            stats[P + 'force_max_step0'] = float(m.group(1))
        if stats.get(P + 'faults'):
            stats[P + 'paged_mb'] = round(
                stats[P + 'faults'] * self.config['page_kb'] / 1024.0, 1)
        stats[P + 'correct'] = int(stats[P + 'rel_err'] < 1e-5
                               and stats.get(P + 'get_errors', 0) == 0
                               and 'RESULT INVALID' not in text)

    def start(self):
        c = self.config
        b = self._bin_dir()
        gmx = os.path.join(b, 'gmx-build', 'bin', 'gmx')
        _reap_stale_runtime(self.log, 'gmx')
        _wait_for_free_vram(self.log, self._actual_cache_mb() / 1024.0 + 1.0)
        os.makedirs(c['output_dir'], exist_ok=True)
        self._prepare_inputs(gmx)
        out = self._output_file()
        env = dict(self.mod_env)
        if c['clio_conf']:
            env['CLIO_SERVER_CONF'] = c['clio_conf']
        s6 = (c['sigma'] ** 6)
        env['GMX_ETERNIA_NB'] = '1'
        env['GMX_ETERNIA_C6'] = repr(4.0 * c['epsilon'] * s6)
        env['GMX_ETERNIA_C12'] = repr(4.0 * c['epsilon'] * s6 * s6)
        env['GMX_ETERNIA_RC'] = repr(c['rcut'])
        env['GMX_ETERNIA_PAGE_KB'] = str(c['page_kb'])
        env['GMX_ETERNIA_BLOCKS'] = str(c['blocks'])
        env['GMX_ETERNIA_SLOTS'] = str(self._slots())
        cmd = [gmx, 'mdrun', '-s', self._tpr(), '-nb', 'gpu', '-ntmpi', '1',
               '-ntomp', str(c['ntomp']), '-deffnm', 'run',
               '-nsteps', str(c['steps'])]
        if c['timeout_sec'] > 0:
            cmd.insert(0, f'timeout {c["timeout_sec"]}')
        self.log(f'Running: {" ".join(cmd)}')
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
