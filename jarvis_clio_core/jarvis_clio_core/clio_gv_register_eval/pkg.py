"""
Static register-usage evaluation for the gpu_vector workload binaries.

One cell = one (workload, variant) pair. Runs
`cuobjdump --dump-resource-usage` on clio_<workload>_<variant>_bench,
parses every kernel's REG/STACK/SHARED, and exports:

  max_regs        registers of the heaviest kernel -- the number that
                  caps occupancy (and, via the indirect-call rule, what
                  ptxas charges EVERY kernel in a module with an
                  address-taken function set: see
                  coro-register-indirect-call notes in the repo docs)
  max_reg_kernel  demangled name of that kernel (truncated)
  n_kernels       kernels found
  mean_regs       mean over all kernels
  max_stack       largest per-thread stack frame (coroutine frames of the
                  paged variants live here / spill to local)

The full per-kernel table is written to <output_dir>/<wl>_<variant>.regs
for the post: figure and for by-hand inspection. No GPU is touched --
this is fatbin analysis, safe to run alongside anything.

TWO BINARY SHAPES. Most variants are clio_<workload>_<variant>_bench and
carry a CUDA fatbin cuobjdump can read. SYCL is neither: its executable is
clio_<workload>_paged_bench_sycl, and its device code is a clang offload
bundle, NOT a fatbin -- cuobjdump reports "does not contain device code"
and exits 0, so it must not be used as the reader or the SYCL row silently
comes back empty. For that variant the numbers come from `ptxas -v` on the
device module, which is what the SYCL toolchain itself runs. Point
sycl_ptxas_dir at a directory of ptxv_<workload>.txt logs; producing them
needs the SYCL build tree, so it is a separate step rather than something
this package shells out to.
"""
from jarvis_cd.core.pkg import Application
import os
import re
import subprocess


def _cuobjdump():
    for cand in ('cuobjdump', '/usr/local/cuda/bin/cuobjdump'):
        try:
            subprocess.run([cand, '--version'], capture_output=True,
                           timeout=10)
            return cand
        except (OSError, subprocess.SubprocessError):
            continue
    raise Exception('cuobjdump not found (PATH or /usr/local/cuda/bin)')


def _demangle(names):
    try:
        out = subprocess.run(['c++filt'], input='\n'.join(names),
                             capture_output=True, text=True, timeout=30)
        return out.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        return names


def _sycl_name(sym):
    """Readable name for a SYCL kernel symbol.

    The entry point is the mangled type of a lambda, so the workload's own
    name only survives as the enclosing Launch<Name> function. Itanium
    mangling writes identifiers as <length><chars>, so the name ends after
    exactly <length> characters -- matching [A-Za-z0-9_]* instead runs
    straight through the following components and yields a 200-char blob.
    """
    for pat in (r'(\d+)(Launch[A-Z])', r'(\d+)([A-Z][A-Za-z0-9_]{3,})'):
        m = re.search(pat, sym)
        if m:
            n = int(m.group(1))
            start = m.start(2)
            return sym[start:start + n]
    return sym[:48]


def _parse_ptxas(path):
    """Rows of (kernel, regs, stack, shared, local) from a `ptxas -v` log."""
    rows, sym = [], None
    with open(path, errors='replace') as f:
        for ln in f:
            m = re.search(r"Compiling entry function '([^']+)'", ln)
            if m:
                sym = m.group(1)
                continue
            m = re.search(r'Used (\d+) registers', ln)
            if m and sym:
                stack = re.search(r'(\d+) bytes cumulative stack size', ln)
                smem = re.search(r'(\d+) bytes smem', ln)
                spill = re.search(r'(\d+) bytes spill stores', ln)
                rows.append((_sycl_name(sym), int(m.group(1)),
                             int(stack.group(1)) if stack else 0,
                             int(smem.group(1)) if smem else 0,
                             int(spill.group(1)) if spill else 0))
                sym = None
    return rows


class ClioGvRegisterEval(Application):
    """Per-kernel register usage of one workload/variant binary."""

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'workload',
             'msg': 'lammps_md | gmx | lbann | grayscott | kmeans | weights',
             'type': str, 'default': 'kmeans'},
            {'name': 'variant',
             'msg': 'mpi | nccl | nvshmem | bam | kokkos | paged | sycl',
             'type': str, 'default': 'mpi'},
            {'name': 'bin_dir',
             'msg': 'directory holding the benchmark binaries (empty = '
                    'resolve from PATH)', 'type': str, 'default': ''},
            {'name': 'sycl_ptxas_dir',
             'msg': 'directory of ptxv_<workload>.txt `ptxas -v` logs; only '
                    'read when variant=sycl', 'type': str, 'default': ''},
            {'name': 'output_dir', 'msg': 'where the per-kernel tables go',
             'type': str, 'default': '/tmp/clio_gv_register_eval'},
        ]

    def _binary(self):
        c = self.config
        if c['variant'] == 'sycl':
            # The SYCL edition is the paged driver relinked against the SYCL
            # launch TU, so it keeps the paged name with a suffix.
            name = 'clio_%s_paged_bench_sycl' % c['workload']
        else:
            name = 'clio_%s_%s_bench' % (c['workload'], c['variant'])
        if c['bin_dir']:
            return os.path.join(c['bin_dir'], name)
        return name

    def _table_file(self):
        c = self.config
        return os.path.join(c['output_dir'],
                            '%s_%s.regs' % (c['workload'], c['variant']))

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)

    def start(self):
        c = self.config
        os.makedirs(c['output_dir'], exist_ok=True)
        if c['variant'] == 'sycl':
            return self._start_sycl()
        binary = self._binary()
        if c['bin_dir'] == '':
            from shutil import which
            resolved = which(binary, path=self.mod_env.get('PATH'))
            if resolved is None:
                # A variant that was not built (e.g. nccl exists only for
                # lammps_md) is an EMPTY table, not an error: the post
                # figure then shows the hole instead of the sweep dying.
                self.log('  %s not on PATH -- writing empty table' % binary)
                open(self._table_file(), 'w').close()
                return
            binary = resolved
        out = subprocess.run([_cuobjdump(), '--dump-resource-usage', binary],
                             capture_output=True, text=True, timeout=300)
        rows = []
        cur = None
        for ln in out.stdout.splitlines():
            m = re.match(r'\s*Function (\S+):', ln)
            if m:
                cur = m.group(1)
                continue
            m = re.search(r'REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+'
                          r'LOCAL:(\d+)', ln)
            if m and cur:
                rows.append((cur, int(m.group(1)), int(m.group(2)),
                             int(m.group(3)), int(m.group(4))))
                cur = None
        # A fatbin can carry the same kernel for several arches; keep the
        # WORST (max regs) instance per name so the table has one row per
        # kernel and the number reported is the binding one.
        best = {}
        for name, reg, stack, shared, local in rows:
            if name not in best or reg > best[name][0]:
                best[name] = (reg, stack, shared, local)
        names = list(best.keys())
        pretty = _demangle(names)
        with open(self._table_file(), 'w') as f:
            f.write('# kernel\tregs\tstack\tshared\tlocal\n')
            for name, nice in zip(names, pretty):
                reg, stack, shared, local = best[name]
                short = re.sub(r'\(.*$', '', nice).split('::')[-1] or name
                f.write('%s\t%d\t%d\t%d\t%d\n'
                        % (short, reg, stack, shared, local))
        self.log('  %s: %d kernels -> %s'
                 % (os.path.basename(binary), len(best),
                    self._table_file()))

    def _start_sycl(self):
        c = self.config
        log = os.path.join(c['sycl_ptxas_dir'] or '',
                           'ptxv_%s.txt' % c['workload'])
        if not c['sycl_ptxas_dir'] or not os.path.exists(log):
            # Same contract as an unbuilt variant: an empty table becomes a
            # hole in the figure rather than killing the sweep.
            self.log('  no ptxas log at %s -- writing empty table' % log)
            open(self._table_file(), 'w').close()
            return
        rows = _parse_ptxas(log)
        best = {}
        for name, reg, stack, shared, local in rows:
            if name not in best or reg > best[name][0]:
                best[name] = (reg, stack, shared, local)
        with open(self._table_file(), 'w') as f:
            f.write('# kernel\tregs\tstack\tshared\tlocal\n')
            for name, (reg, stack, shared, local) in best.items():
                f.write('%s\t%d\t%d\t%d\t%d\n'
                        % (name, reg, stack, shared, local))
        self.log('  %s: %d kernels -> %s'
                 % (os.path.basename(log), len(best), self._table_file()))

    def stop(self):
        pass

    def clean(self):
        out = self.config['output_dir']
        for name in (os.listdir(out) if os.path.isdir(out) else []):
            if name.endswith('.regs'):
                os.remove(os.path.join(out, name))

    def _get_stat(self, stats):
        path = self._table_file()
        stats['completed'] = 0
        if not os.path.exists(path):
            return
        rows = []
        with open(path) as f:
            for ln in f:
                if ln.startswith('#') or not ln.strip():
                    continue
                name, reg, stack, shared, local = ln.rstrip('\n').split('\t')
                rows.append((name, int(reg), int(stack), int(shared),
                             int(local)))
        if not rows:
            return  # variant not built: completed stays 0, no reg stats
        stats['completed'] = 1
        stats['n_kernels'] = len(rows)
        worst = max(rows, key=lambda r: r[1])
        stats['max_regs'] = worst[1]
        stats['max_reg_kernel'] = worst[0][:48]
        stats['mean_regs'] = round(sum(r[1] for r in rows) / len(rows), 1)
        stats['max_stack'] = max(r[2] for r in rows)
