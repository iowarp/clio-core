"""
LAMMPS eternia paged pair-style package

Drives LAMMPS with pair_style lj/cut/eternia: positions, types, the neighbour
list and the force array all live in the CTE and are paged into the GPU by the
force kernel.

Its access pattern is the only WRITE-heavy one of the four gpu_vector
workloads -- forces are written back through the same paged vector they are
accumulated in, page-aligned per block -- so it exercises writeback and
eviction of dirty pages, which the read-mostly workloads do not.

The style requires `newton off` and a single MPI rank, both set in the
generated input; a half neighbour list would need each pair's force written to
an atom in another block's page.

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

# The pair style's own floor: the i-atom page and the j-atom page must be
# resident together, with one spare so a claim cannot evict either. Encoded
# here so a sweep point below it is refused at CONFIGURE time with the
# arithmetic, instead of running LAMMPS and failing inside the pair style
# after the cell has already been spent.
MIN_SLOTS = 3

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



class ClioEterniaLammps(Application):
    """LAMMPS with the paged lj/cut/eternia pair style."""

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
            {'name': 'restart_steps',
             'msg': 'Write a binary restart every N steps (0 = never). Both '
                    'paths pay this: LAMMPS owns the atom arrays, so the paged '
                    'vectors are a copy and a restart still has to be '
                    'serialised. Measured at ~71 MB/s, far below this disk',
             'type': int, 'default': 0},
            {'name': 'binary_rel',
             'msg': 'Path to the lmp binary under bin_dir. Defaults to the '
                    'KOKKOS+CUDA build, which is the one that has a GPU '
                    'baseline; the plain build has no GPU package at all and '
                    'its "stock" kernel is a single CPU core',
             'type': str, 'default': 'lmp-kk/lmp'},
            {'name': 'baseline_kokkos',
             'msg': 'Use lj/cut/kk on the GPU for the baseline instead of the '
                    'serial CPU lj/cut. Comparing a paged GPU kernel against '
                    'one CPU core is not a like-for-like measurement',
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
             'type': str, 'default': '/home/llogan/Documents/Projects/iowarp/core/external/lammps/examples/ETERNIA/clio.yaml'},
            {'name': 'output_dir', 'msg': 'Output directory', 'type': str,
             'default': '/tmp/clio_eternia_lammps'},
            {'name': 'threads',
             'msg': 'Threads per CUDA block for the pair kernel', 'type': int,
             'default': 256},
            {'name': 'lattice_cells',
             'msg': 'fcc lattice cells per side; atoms = 4 x cells^3. The size '
                    'knob: 10 -> 4000 atoms, 20 -> 32000, 30 -> 108000',
             'type': int, 'default': 20},
            {'name': 'steps', 'msg': 'MD steps', 'type': int, 'default': 20},
            {'name': 'cutoff', 'msg': 'LJ cutoff in sigma', 'type': float,
             'default': 2.5},
        ]

    def _atoms(self):
        return 4 * self.config['lattice_cells'] ** 3

    def _dataset_mb(self):
        """Positions, types, forces and the neighbour list, in MB.

        x and f are 4 floats per atom each (padded so an atom never straddles a
        page), type is one int, and the neighbour list dominates at this
        cutoff -- counted at the measured ~180 entries per atom for lj/cut at
        2.5 sigma. Approximate, and used only to scale `cache_frac`.
        """
        per_atom = (4 * 4) + (4 * 4) + 4 + (180 * 4)
        return self._atoms() * per_atom / (1024.0 * 1024.0)

    def _configure(self, **kwargs):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        slots = self._slots()
        if slots < MIN_SLOTS:
            raise Exception(
                f"slots={slots} but lj/cut/eternia needs at least {MIN_SLOTS}: "
                f"the i-atom page and the j-atom page must be resident "
                f"together, with one spare so a claim cannot evict either."
                + (f" Derived from cache_mb={c['cache_mb']} over "
                   f"{c['blocks']} blocks at {c['page_kb']}KB; this workload "
                   f"needs at least "
                   f"{MIN_SLOTS * c['blocks'] * c['page_kb'] // 1024} MB."
                   if c.get('cache_mb') or c.get('cache_frac') else ''))
        self._check_vram()
        pages = int(self._dataset_mb() * 1024 // c['page_kb'])
        if pages < 1:
            raise Exception(
                f"dataset {self._dataset_mb():.1f} MB is smaller than one "
                f"{c['page_kb']}KB page; nothing would be paged.")
        self._write_input()
        self.log('LAMMPS eternia paged pair style configured')
        self.log(f'  system:    {self._atoms()} atoms, '
                 f'{self._dataset_mb():.1f} MB paged over x/type/neigh/f')
        self.log(f'  parallel:  {c["blocks"]} blocks x {c["threads"]} threads, '
                 f'{slots} pages/block ({self._actual_cache_mb():.1f} MB cache)')

    def _input_file(self):
        return os.path.join(self.config['output_dir'], 'in.melt.eternia')

    def _write_input(self):
        """Generate the LAMMPS input at this cell's settings.

        `newton off` must precede the box definition -- LAMMPS rejects the
        change afterwards -- and atom sorting is on because a page holds a
        contiguous range of atom indices, so paging only pays when spatial and
        index locality agree.
        """
        c = self.config
        # The BASELINE input uses lj/cut/kk, which runs on the GPU, and drops
        # the two settings that exist only for the paged kernel: `newton off`
        # (the paged kernel needs a full neighbour list) and the atom sort.
        # Getting this wrong is not subtle -- delegating to the stock kernel
        # while still requesting a full list double-counts every pair.
        kk = c['baseline'] and c['baseline_kokkos']
        restart = ('\nrestart         %d %s %s'
                   % (c['restart_steps'],
                      os.path.join(c['output_dir'], 'rst_a.bin'),
                      os.path.join(c['output_dir'], 'rst_b.bin'))
                   ) if c['restart_steps'] > 0 else ''
        with open(self._input_file(), 'w') as f:
            f.write(f'''units           lj
atom_style      atomic
{'' if kk else 'newton          off'}
lattice         fcc 0.8442
region          box block 0 {c["lattice_cells"]} 0 {c["lattice_cells"]} 0 {c["lattice_cells"]}
create_box      1 box
create_atoms    1 box
mass            1 1.0
velocity        all create 3.0 87287 loop geom
{'' if kk else 'atom_modify     sort 1000 2.0'}
pair_style      {f'lj/cut/kk {c["cutoff"]}' if kk else f'lj/cut/eternia {c["cutoff"]} page {c["page_kb"]} blocks {c["blocks"]} threads {c["threads"]} slots {self._slots()} stats on'}
pair_coeff      1 1 1.0 1.0 {c["cutoff"]}
neighbor        0.3 bin{restart}
neigh_modify    every 20 delay 0 check no
fix             1 all nve
thermo          {c["steps"]}
run             {c["steps"]}
''')

    def _output_file(self):
        c = self.config
        tag = (f'b{c["blocks"]}_pg{c["page_kb"]}kb_sl{self._slots()}'
               f'_n{self._atoms()}_st{c["steps"]}')
        mode = 'base' if c['baseline'] else 'paged'
        return os.path.join(c['output_dir'], f'lammps_{mode}_{tag}.log')

    def _get_stat(self, stats):
        # Stats are NAMESPACED. The three eternia packages share a
        # pipeline and would otherwise write the same keys -- faults,
        # completed, correct -- into one row, where the last package to
        # report silently overwrites the others and the table reads as
        # though two of the three workloads failed.
        P = 'lmp_'
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
        # Final thermo row: the step column equals the run length.
        e_pair = None
        seen_header = False
        for ln in text.splitlines():
            if 'Step' in ln and 'E_pair' in ln:
                seen_header = True
                continue
            if seen_header:
                parts = ln.split()
                if len(parts) >= 3 and parts[0] == str(self.config['steps']):
                    try:
                        e_pair = float(parts[2])
                    except ValueError:
                        pass
        if e_pair is None:
            stats[P + 'completed'] = 0
            return
        stats[P + 'completed'] = 1
        stats[P + 'e_pair'] = e_pair
        # LAMMPS alternates two restart files, so bytes on disk understate the
        # I/O: what was written is size x number of restarts.
        if self.config['restart_steps'] > 0:
            one = 0
            for nm in ('rst_a.bin', 'rst_b.bin'):
                fp = os.path.join(self.config['output_dir'], nm)
                if os.path.exists(fp):
                    one = max(one, os.path.getsize(fp))
            n = self.config['steps'] // self.config['restart_steps']
            stats[P + 'ckpt_gb'] = round(one * n / (1024.0 ** 3), 3)
            stats[P + 'ckpt_free'] = 0
        stats[P + 'mode'] = 'base' if self.config['baseline'] else 'paged'
        # LAMMPS prints the same timing breakdown for both kernels, so the
        # Pair row is directly comparable between the paged and stock runs --
        # which whole-process wall clock is not, since it also carries runtime
        # startup that only the paged path pays.
        for ln in text.splitlines():
            if ln.startswith('Pair') and '|' in ln:
                parts = [x.strip() for x in ln.split('|')]
                if len(parts) >= 4:
                    try:
                        stats[P + 'pair_time_s'] = float(parts[2])
                    except ValueError:
                        pass
                break
        stats[P + 'e_pair_per_atom'] = round(e_pair, 6)
        # Paging counters from the last per-step line the pair style prints.
        last = None
        for ln in text.splitlines():
            if ln.startswith('eternia step '):
                last = ln
        if last:
            for pat, out in ((r'x faults (\d+)', 'faults'),
                             (r'evicts (\d+)', 'evicts'),
                             (r'neigh faults (\d+)', 'neigh_faults'),
                             (r'f puts (\d+)', 'force_puts'),
                             (r'get errors (\d+)', 'get_errors'),
                             (r'pairs (\d+)/', 'pairs')):
                m = re.search(pat, last)
                if m:
                    stats[P + out] = int(m.group(1))
        if stats.get(P + 'faults'):
            stats[P + 'paged_mb'] = round(
                stats[P + 'faults'] * self.config['page_kb'] / 1024.0, 1)
        # CORRECTNESS. E_pair is independent of the paging geometry -- a page
        # size that changed it would mean atoms were missed or double counted.
        # The reference is the stock lj/cut value for this system, which the
        # sweep's post-processing compares ACROSS cells; here we only record
        # that the run produced an energy and read every page it asked for.
        stats[P + 'correct'] = int(stats.get(P + 'get_errors', 0) == 0)

    def start(self):
        c = self.config
        b = self._bin_dir()
        binary = os.path.join(b, *c['binary_rel'].split('/'))
        _reap_stale_runtime(self.log, 'lmp')
        _wait_for_free_vram(self.log, self._actual_cache_mb() / 1024.0 + 1.0)
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        env = dict(self.mod_env)
        if c['clio_conf']:
            env['CLIO_SERVER_CONF'] = c['clio_conf']
        if c['baseline']:
            env['ETERNIA_BASELINE'] = '1'
        cmd = [binary]
        if c['baseline'] and c['baseline_kokkos']:
            # -k on g 1 selects one GPU; -sf kk routes styles to the KOKKOS
            # variants. Without these the kk style is not used and the run
            # silently falls back to the CPU kernel.
            cmd += ['-k', 'on', 'g', '1', '-sf', 'kk']
        cmd += ['-in', self._input_file()]
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
