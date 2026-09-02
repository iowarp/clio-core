"""
Generic driver for the gpu_vector science-workload benchmarks.

ONE package covers every (workload, variant) pair under
context-transfer-engine/adapter/gpu_vector/benchmark/:

  workload:  lammps_md | gmx | lbann | grayscott | kmeans | weights
  variant:   mpi | nccl | nvshmem | bam | paged

DISTRIBUTED. Two different mechanisms, and they are not interchangeable:

  mpi / nccl / nvshmem   `nprocs` ranks under `mpi_launcher`. These ARE
                         MPI programs; across nodes use `srun -n {n}
                         --ntasks-per-node=1` inside a multi-node
                         allocation.
  paged                  `nodes` PROCESSES, one per node, placed by
                         `node_launcher` (srun) and joined into one clio
                         cluster through a generated CLIO_SERVER_CONF.
                         They are not MPI programs -- each hosts the
                         runtime in-process and they meet through the CTE.

The declared problem size is GLOBAL and split, for every workload except
weights (whose --pages is per block per node, so its model grows with the
node count). 32 GB per node therefore means --data-mb 64000 at 2 nodes for
kmeans/grayscott, and --data-mb 32000 for weights.

vram_peak_mb in a distributed cell is the HEAD NODE's GPU only -- nothing
samples the peers.

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


def _wait_for_gpu_idle(log, max_used_mb=700, timeout_s=240):
    """Block until the GPU has released the PREVIOUS cell's memory.

    A finished process releases its VRAM LAZILY, and back-to-back sweep
    cells at ~6 GB footprints can otherwise start against a GPU that
    still holds most of the last cell's allocations. Measured in the
    combined memory-pressure sweep: two lammps_md blocks=1 cells ran 2x
    slow AND failed their physics gates mid-sweep, then passed cleanly
    standalone -- the squeeze, not the vector. (The per-workload kmeans
    pkg documented the same pathology for a GNN sweep.)"""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=15)
            used = int(out.stdout.strip().splitlines()[0])
        except Exception:
            return
        if used <= max_used_mb:
            return
        log('  waiting for GPU to drain: %d MB still in use '
            '(previous cell releasing lazily)' % used)
        time.sleep(5)
    log('  WARNING: GPU still holds %d MB after %ds -- this cell may '
        'run squeezed' % (used, timeout_s))


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
            # ---- paged distributed (one process per NODE) -------------
            {'name': 'nodes',
             'msg': 'paged: run distributed across N nodes (--nodes N, one '
                    'process per node, --node from the launcher). 1 keeps '
                    'the historic single-process behavior',
             'type': int, 'default': 1},
            {'name': 'node_launcher',
             'msg': 'paged distributed: how to place ONE process per node; '
                    '"{n}" is substituted with `nodes`. srun is the only '
                    'launcher that works here -- the paged benches host the '
                    'runtime in-process and are not MPI programs, so mpirun '
                    'has nothing to bootstrap',
             'type': str,
             'default': 'srun -N {n} --ntasks-per-node=1 --gpus-per-node=1'},
            {'name': 'cluster_port',
             'msg': 'paged distributed: port for the generated cluster '
                    'config. Must not collide with anything else in the '
                    'allocation',
             'type': int, 'default': 9425},
            {'name': 'ram_mb',
             'msg': 'paged distributed: host RAM tier MB in the generated '
                    'cluster config (0 = this node SHARE of data_mb, plus '
                    'slack -- see _ram_tier_mb)',
             'type': int, 'default': 0},
            {'name': 'barrier_sec',
             'msg': 'paged distributed: seconds a finished node waits for '
                    'its peers before leaving. A node that exits early '
                    'takes its runtime with it and a peer still paging '
                    'against it waits forever',
             'type': int, 'default': 300},
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
            {'name': 'mpi_launcher',
             'msg': 'launcher for the mpi/nccl/nvshmem variants; "{n}" is '
                    'substituted with nprocs. The default is OpenMPI '
                    '(--oversubscribe is an OpenMPI-only flag). A Cray '
                    'site has no mpirun at all -- cray-mpich launches '
                    'through slurm -- so there use "srun -n {n}"',
             'type': str, 'default': 'mpirun -n {n} --oversubscribe'},
            {'name': 'bin_dir',
             'msg': 'directory holding the benchmark binary. Empty = '
                    'resolve the bare name through PATH. REQUIRED for '
                    'multi-node mpi/nccl cells, which must come from '
                    '<build>/bin-cray (cray-mpich); the PATH build in '
                    'bin/ is HPC-X OpenMPI and cannot cross Slingshot -- '
                    'it degrades to one 1-rank job per task and PASSES, '
                    'so the wrong build is not visible in the output. '
                    'See _binary and build_baselines_cray.sh',
             'type': str, 'default': ''},
            {'name': 'ld_preload',
             'msg': 'LD_PRELOAD for this cell. NCCL over Slingshot needs '
                    'libfabric preloaded ($CLIO_DELTA_NCCL_PRELOAD from '
                    'env.sh) alongside NCCL_NET_PLUGIN=ofi/FI_PROVIDER=cxi',
             'type': str, 'default': ''},
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

    def _binary_name(self):
        """Bare executable name. This is what `pkill -x` matches and what
        /proc/<pid>/exe basenames to, so the reaper and the pre-cell kill
        MUST use this and not the path below."""
        c = self.config
        return 'clio_%s_%s_bench' % (c['workload'], c['variant'])

    def _binary(self):
        """What actually goes on the command line.

        WHY bin_dir EXISTS. Delta needs TWO builds of the CTE-free
        baselines and only one can be on PATH:

          bin/       HPC-X OpenMPI -- single node only; its UCX PML cannot
                     cross Slingshot ("ucp_ep_create failed: Destination is
                     unreachable"), so a MULTI-NODE cell that picks this up
                     does not fail, it runs every rank as its own 1-rank
                     job and passes every gate while measuring nothing.
          bin-cray/  cray-mpich + the aws-ofi-nccl plugin -- multi-node,
                     under `srun --mpi=cray_shasta`.

        Resolving by name from PATH therefore silently produces the WRONG
        answer at 2 nodes rather than an error. Naming the directory is the
        only way to be sure which build a cell measured; see
        build_baselines_cray.sh."""
        c = self.config
        d = (c.get('bin_dir') or '').strip()
        if not d:
            return self._binary_name()
        d = os.path.expanduser(os.path.expandvars(d))
        return os.path.join(d, self._binary_name())

    # ------------------------------------------------------------------
    # paged distributed
    #
    # The paged benches are NOT MPI programs: each hosts the clio runtime
    # in-process, and they find each other through a cluster config naming
    # a hostfile -- the same mechanism test/distributed_workloads/ drives
    # with docker-compose. Three things have to be true, and none of them
    # is a launcher flag:
    #
    #   1. CLIO_SERVER_CONF must be set BEFORE the bench starts. Every
    #      paged bench writes its own config only when nobody else
    #      supplied one, so the cluster config has to be exported.
    #   2. That config must declare the TIERS. The per-bench config is
    #      what sizes hbm/ram/file from --hbm-mb/--data-mb/--nvme-mb; once
    #      we supply our own, that sizing is ours to reproduce, and the
    #      2 GB the docker harness declares is nowhere near a 32 GB deck.
    #   3. No node may LEAVE while a peer is still paging. The runtime
    #      dies with the process and a peer whose pages live on the
    #      departing node waits forever -- hence the done-file barrier,
    #      copied from the compose harness for the same reason.
    # ------------------------------------------------------------------

    def _dist(self):
        """A distributed paged cell. mpi/nccl scale with `nprocs` through
        their own launcher and never come through here."""
        c = self.config
        return c['variant'] == 'paged' and int(c.get('nodes') or 1) > 1

    def _dist_dir(self):
        """Cluster config, per-node logs and barrier files. MUST be the
        SHARED dir: the node logs are written on the node that produced
        them and read back on the head node, and output_dir is node-local
        /tmp by design."""
        return os.path.join(self.shared_dir, 'gvw_dist')

    def _ram_tier_mb(self):
        """Host RAM tier for the generated config.

        THE DECLARED PROBLEM SIZE IS GLOBAL FOR EVERY WORKLOAD BUT
        weights. kmeans/grayscott split --data-mb, gmx splits K, lbann
        splits H and O, lammps_md splits the lattice -- so a node holds
        data_mb/nodes. weights is the exception: --pages is per block PER
        NODE, so its model is nodes x bigger and data_mb is already the
        per-node figure. Sizing the tier off the global number instead
        would prefault N times the memory a node actually uses, which on
        a 32 GB deck is minutes of wall clock spent on nothing."""
        c = self.config
        if c.get('ram_mb'):
            return int(c['ram_mb'])
        per_node = int(c.get('data_mb') or 0)
        if per_node and c['workload'] != 'weights':
            per_node //= max(1, int(c.get('nodes') or 1))
        return (per_node + 512) if per_node else 4096

    def _nvme_tier_mb(self):
        """File tier MB per node -- the only NON-VOLATILE tier in the stack.

        hbm and ram are declared without a persistence_level, so the core
        registers them kVolatile and FlushData will not accept them as
        targets. The file tier alone is `temporary`, which makes it the
        sole destination for the end-of-run BenchFlushData() -- and that
        flush moves EVERY volatile block, i.e. this node's whole share of
        the deck.

        Undersized, the flush fills the tier and then fails per blob with
        rc 10 + kCteAllocNoHealthyTarget = 12, and (before the FlushData
        fix that accompanies this change) each failure had already freed
        the blob's volatile blocks. So this is sized off the SAME per-node
        share as _ram_tier_mb, for the same reason and by the same rule.

        -1 sizes it from the problem; a too-small explicit value is
        refused rather than run.
        """
        c = self.config
        want = int(c.get('nvme_mb') or 0)
        if want == 0:
            return 0  # RAM-only stack: the flush is a clean no-op
        per_node = int(c.get('data_mb') or 0)
        if per_node and c['workload'] != 'weights':
            per_node //= max(1, int(c.get('nodes') or 1))
        need = per_node + max(512, per_node // 20) if per_node else 0
        if want < 0:
            return need or 4096
        if need and want < need:
            raise Exception(
                'nvme_mb=%d is too small: this node holds %d MB and the file '
                'tier is the only non-volatile tier, so the end-of-run flush '
                'pushes all of it there and dies partway once the tier fills '
                '(rc 12). Use nvme_mb >= %d, nvme_mb=-1 to size it '
                'automatically, or nvme_mb=0 for a RAM-only stack.'
                % (want, per_node, need))
        return want

    def _cluster_conf_path(self):
        return os.path.join(self._dist_dir(), 'gvw_cluster.yaml')

    def _launcher_path(self):
        return os.path.join(self._dist_dir(), 'gvw_node.sh')

    def _write_cluster_conf(self):
        """The config the benches would otherwise write for themselves,
        with a hostfile bolted on. Schema notes that cost a run each:
        pool ids are "major.minor" STRINGS (a bare 512 is rejected at load
        with `Invalid UniqueId format`), and a core pool declares
        `storage:`, not `tiers:`."""
        c = self.config
        os.makedirs(self._dist_dir(), exist_ok=True)
        hostfile = self.hostfile.path if self.hostfile else ''
        if not hostfile or not os.path.exists(hostfile):
            # A PIPELINE TEST DOES NOT INHERIT THE SCHEDULER'S HOSTFILE.
            # `scheduler:` binds self.hostfile on the pipeline it loads, but
            # a sweep builds a FRESH pipeline per cell (<name>_runN) and
            # those bind nothing -- self.hostfile comes back None and every
            # cell fails identically. Name the file on the package instead:
            # the scheduler writes it to
            # ${HOME}/.ppi-jarvis/shared/<pipeline>/hostfile.txt at job
            # start, and `hostfile:` on the pkg is read first by
            # Pkg.get_hostfile().
            raise Exception(
                'paged distributed needs an allocation hostfile; '
                'self.hostfile=%r. In a pipeline TEST the scheduler block '
                'does not reach the per-cell pipelines -- set `hostfile:` '
                'on this package to the scheduler hostfile, e.g. '
                '%s/.ppi-jarvis/shared/<pipeline>/hostfile.txt'
                % (hostfile, os.path.expanduser('~')))
        lines = [
            'networking:',
            '  port: %d' % c['cluster_port'],
            '  hostfile: "%s"' % hostfile,
            '',
            'runtime:',
            '  num_threads: 8',
            '  queue_depth: 8192',
            # A worker that falls back to sleeping adds its sleep to every
            # fault, and a fault here is a round trip that may cross the
            # wire. Keep them spinning.
            '  first_busy_wait: 2000000000',
            '',
            'gpu:',
            '  queue_depth: 8192',
            '',
            'compose:',
            '  - mod_name: clio_bdev',
            '    pool_name: "ram::chi_default_bdev"',
            '    pool_query: local',
            '    pool_id: "301.0"',
            '    bdev_type: ram',
            '    capacity: "1GB"',
            '',
            '  - mod_name: clio_cte_core',
            '    pool_name: cte_core',
            '    pool_query: local',
            '    pool_id: "512.0"',
            '    storage:',
            # MaxBwDpe splits on target_score <= blob_score and sorts the
            # preferred group DESCENDING; the vector puts pages at blob
            # score 1.0, so the HIGHER score is the preferred tier. HBM
            # must sit ABOVE the host tier, not below it.
            '      - path: "hbm::gvw_hbm"',
            '        bdev_type: "hbm"',
            '        capacity_limit: "%dMB"' % (c['hbm_mb'] or 256),
            '        score: 1.0',
            '      - path: "ram::gvw_ram"',
            '        bdev_type: "ram"',
            '        capacity_limit: "%dMB"' % self._ram_tier_mb(),
            '        score: 0.2',
        ]
        nvme_mb = self._nvme_tier_mb()
        if nvme_mb:
            # score BELOW the host tier, for the same reason: higher score
            # = preferred, so storage must be last or it becomes the
            # FIRST-choice tier.
            lines += [
                '      - path: "%s"' % c['nvme_path'],
                '        bdev_type: "file"',
                '        persistence_level: "temporary"',
                '        capacity_limit: "%dMB"' % nvme_mb,
                '        score: 0.0',
            ]
        lines += ['    dpe:', '      dpe_type: "max_bw"', '']
        path = self._cluster_conf_path()
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        return path

    def _write_node_launcher(self):
        """One process per node needs a DIFFERENT --node on each, and
        srun hands out no such thing to a non-MPI program -- only
        SLURM_PROCID. This wrapper turns that into --node, keeps each
        node's output in its own file on the shared FS (the head node
        cannot read a peer's /tmp), and holds the barrier."""
        os.makedirs(self._dist_dir(), exist_ok=True)
        path = self._launcher_path()
        with open(path, 'w') as f:
            f.write(
                '#!/bin/sh\n'
                '# generated by jarvis_clio_core.clio_gv_workload\n'
                'i="${SLURM_PROCID:-0}"\n'
                'log="${GVW_LOGBASE}.node${i}.log"\n'
                'echo "host=$(hostname) node=$i nodes=${GVW_NODES}" > "$log"\n'
                '"$@" --node "$i" >> "$log" 2>&1\n'
                'rc=$?\n'
                'touch "${GVW_DONEDIR}/done_${i}"\n'
                'w=0\n'
                'while [ "$(ls "${GVW_DONEDIR}"/done_* 2>/dev/null | wc -l)"'
                ' -lt "${GVW_NODES}" ] && [ "$w" -lt "${GVW_BARRIER}" ]; do\n'
                '  sleep 1; w=$((w+1))\n'
                'done\n'
                'exit $rc\n')
        os.chmod(path, 0o755)
        return path

    def _node_log(self, i):
        return os.path.join(self._dist_dir(),
                            '%s.node%d.log' % (self._tag(), i))

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

        def opt_nvme():
            """--nvme-mb via _nvme_tier_mb, which resolves the -1 sentinel
            and refuses a tier too small to hold the end-of-run flush.
            opt() cannot be used: it treats -1 as "unset" and would drop
            the flag silently."""
            n = self._nvme_tier_mb()
            if n:
                a.append('--nvme-mb %d' % n)

        opt('--blocks', 'blocks')
        opt('--threads', 'threads')
        # --node is PER PROCESS and is appended by the node launcher; only
        # the node COUNT is the same on every process.
        if self._dist():
            a.append('--nodes %d' % c['nodes'])
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
                # TWO CACHE KNOBS, AND THEY ARE NOT INTERCHANGEABLE.
                #
                # --slots sets the x/v frame count DIRECTLY, per block.
                # --vram-mb hands the bench a TOTAL byte budget and lets its
                # own planner divide it across x, v, f and the neighbour
                # list, honouring the measured per-vector floors (kAtomFloor
                # 24 frames, and 3/4 of rows-per-block above that). Those
                # floors are why a residency sweep must go through the
                # budget: --slots 16 at 16 blocks is BELOW the floor and the
                # cell crashes rather than reporting a small cache
                # (job 21721*, dist2_md_ooc row 0: blocks=16 slots=16 ->
                # completed=0 crashed=1, while blocks=4 slots=40 passed).
                #
                # A previous edition of this comment said --vram-mb was
                # "GONE". It is not: the paged bench still parses it and
                # PlanCaches() is driven by it. Only the per-block-cache
                # MODEL changed, not the flag.
                #
                # vram_mb=0 is the bench's own "size for residency" default
                # and is what the 100% rung of a residency sweep wants.
                opt('--vram-mb', 'vram_mb')
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
                opt_nvme()
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
                opt_nvme()
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
                opt_nvme()
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
                opt_nvme()
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
            parts.append(c['mpi_launcher'].format(n=c['nprocs']))
        if self._dist():
            parts.append(c['node_launcher'].format(n=c['nodes']))
            parts.append(self._launcher_path())
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
        keys = ('workload', 'variant', 'nprocs', 'nodes', 'blocks', 'threads',
                'steps', 'ckpt', 'lattice', 'mesh_k', 'atoms', 'hidden',
                'batch', 'data_mb', 'dims', 'clusters', 'flat_pct',
                'page_kb', 'slots', 'cache_mb', 'vram_mb', 'hbm_mb',
                'nvme_mb', 'repeat', 'cap', 'rebin')
        blob = '|'.join('%s=%s' % (k, c.get(k)) for k in keys)
        h = hashlib.md5(blob.encode()).hexdigest()[:8]
        return '%s_%s_np%s_nd%s_%s' % (c['workload'], c['variant'],
                                       c['nprocs'], c.get('nodes') or 1, h)

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
        _reap_stale(self.log, self._binary_name())
        _wait_for_gpu_idle(self.log)
        os.makedirs(c['output_dir'], exist_ok=True)
        out = self._output_file()
        if self._dist():
            self._dist_setup(out)
        # Warmed memory: CLIO_PREFAULT=0 pre-faults the WHOLE RAM tier at
        # compose (mem_bdev_transport.cc), so timings exclude first-touch
        # page population. Exported for every variant; the CTE-free
        # baselines simply ignore it.
        if c['prefault'] != '':
            self.setenv('CLIO_PREFAULT', str(c['prefault']))
        # NCCL's Slingshot path: the aws-ofi-nccl plugin needs libfabric in
        # the process. Expanded here because the value comes from env.sh as
        # $CLIO_DELTA_NCCL_PRELOAD and the sweep YAML carries it verbatim.
        if c.get('ld_preload'):
            self.setenv('LD_PRELOAD', os.path.expandvars(c['ld_preload']))
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
        if self._dist():
            self._dist_collect(out)
        self.log('cell done -> %s' % out)

    def _dist_setup(self, out):
        """Config, launcher, and a CLEAN barrier directory. A leftover
        done_* file from the previous cell lets a node skip the wait
        entirely, which is exactly the failure the barrier exists to
        stop -- and it would show up as an unrelated cell hanging."""
        c = self.config
        d = self._dist_dir()
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(d):
            if name.startswith('done_'):
                os.remove(os.path.join(d, name))
        for i in range(int(c['nodes'])):
            try:
                os.remove(self._node_log(i))
            except OSError:
                pass
        # _reap_stale only sees the HEAD node's /proc. A cell that timed
        # out leaves an orphan on every node, and an orphan still holds
        # the cluster port -- so the next cell dies at bind on a node
        # whose log the head node never looks at. Sweep them all.
        #
        # THE PATTERN MUST BE TRUNCATED TO 15 CHARACTERS. pkill matches
        # against /proc/<pid>/comm, which the kernel caps at
        # TASK_COMM_LEN-1 = 15, so `pkill -x clio_kmeans_paged_bench` (23
        # chars) matched NOTHING and this reap was a silent no-op. The
        # cost was not one bad cell: the first cell to hit timeout_sec
        # left a runtime holding the cluster port, and every cell after it
        # died instantly at bind on the peer while the head node sat out
        # its full timeout at the barrier -- a 96-cell sweep that produced
        # three rows of completed=0 and would have burned its whole wall.
        #
        # `-f` is NOT the fix: it matches the full command line, and the
        # `sh -c` string below contains the binary name, so pkill -f would
        # match and kill its own launcher.
        #
        # output_dir is the cwd every cell runs in, and it exists only on
        # the head node -- srun then prints "couldn't chdir ... going to
        # /tmp instead" on every peer, which silently moves that node's
        # relative paths somewhere else. Make it everywhere.
        Exec('%s sh -c \'mkdir -p %s; pkill -9 -x %s || true; sleep 2\''
             % (c['node_launcher'].format(n=c['nodes']), c['output_dir'],
                self._binary_name()[:15]),
             LocalExecInfo(env=self.mod_env)).run()
        conf = self._write_cluster_conf()
        self._write_node_launcher()
        self.setenv('CLIO_SERVER_CONF', conf)
        self.setenv('CLIO_NUM_CONTAINERS', str(c['nodes']))
        self.setenv('GVW_NODES', str(c['nodes']))
        self.setenv('GVW_BARRIER', str(c['barrier_sec']))
        self.setenv('GVW_DONEDIR', d)
        self.setenv('GVW_LOGBASE', os.path.join(d, self._tag()))
        self.log('  distributed: %d nodes, conf=%s, hosts=%s'
                 % (c['nodes'], conf,
                    ','.join(self.hostfile.hosts[:int(c['nodes'])])))

    def _dist_collect(self, out):
        """Fold every node's log into the cell log.

        srun's own stdout would interleave the nodes line by line, and
        these benches print multi-line summaries -- a machine line spliced
        through another node's is unparseable and, worse, parses WRONG.
        Each node therefore writes its own file and they are concatenated
        in node order here."""
        c = self.config
        parts = []
        try:
            with open(out, errors='replace') as f:
                parts.append(f.read())
        except OSError:
            pass
        for i in range(int(c['nodes'])):
            path = self._node_log(i)
            parts.append('\n===== NODE %d (%s) =====\n' % (i, path))
            try:
                with open(path, errors='replace') as f:
                    parts.append(f.read())
            except OSError:
                parts.append('(no log -- this node produced no output)\n')
        with open(out, 'w') as f:
            f.write(''.join(parts))

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
        stats['binary'] = self._binary_name()
        stats['nodes'] = int(c.get('nodes') or 1)
        # The resolved axis values, so a combined sweep's CSV carries the
        # concrete settings next to the abstract levels.
        stats['blocks_resolved'] = c.get('blocks') or 0
        # THE CACHE RUNG MUST LAND IN THE CSV UNDER ITS OWN SPELLING.
        # A residency sweep whose rows all say the same thing (or nothing)
        # is not a sweep -- lammps_md's budget rung in particular is
        # invisible unless vram_mb is named here, and its 100% rung is
        # vram_mb=0, the bench's "size for residency" sentinel, which is
        # FALSY and would otherwise print as a blank cell rather than as
        # the resident reference point it is.
        if c.get('cap'):
            stats['cache_setting'] = 'cap_pages=%d' % c['cap']
        elif c.get('vram_mb'):
            stats['cache_setting'] = 'vram_mb=%d' % c['vram_mb']
        elif c.get('slots') and not c.get('cache_mb'):
            stats['cache_setting'] = 'slots=%d' % c['slots']
        elif c.get('cache_mb'):
            stats['cache_setting'] = 'cache_mb=%d' % c['cache_mb']
        elif c['workload'] == 'lammps_md' and c['variant'] == 'paged':
            stats['cache_setting'] = 'vram_mb=0 (bench-sized, resident)'
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
            # THE TERMINAL GATE HAS TO BE THERE, not just some gate.
            # STATICS and RESORT both print BEFORE the integration loop, so
            # a run that dies in the force pass still leaves `STATICS GATE:
            # PASS` in the log and every "all gates are PASS" test above
            # passes vacuously. Measured on job 21718877: both nodes aborted
            # with `DEVICE FATAL 5 (AllocatePage: set full)` and the cell
            # landed completed=1 gates_pass=1 nodes_finished=2. NVE is the
            # gate that only exists once the steps actually ran.
            stats['gates_pass'] = int(bool(gates) and
                                      all(g == 'PASS' for g in gates) and
                                      'NVE GATE:' in text)
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

        # EVERY NODE MUST HAVE FINISHED, and the check has to be PER NODE
        # and LAST. Per node, because the cell log is the concatenation of
        # N node logs: a single surviving node's marker satisfies every
        # `in text` test above, so a run whose peer died reads as a clean
        # pass. (Counting markers in the concatenation does not fix it --
        # lammps_md prints four gate lines per node, so one node alone
        # clears a threshold of two.) Last, because the clean-counters
        # fallback directly above sets completed=1 unconditionally and
        # would otherwise undo the veto.
        #
        # THE MARKER IS THE MACHINE LINE, not prose. paged kmeans,
        # grayscott and weights print no "N iters in ..." line at all --
        # only `KMEANS mode=... ms=... get_errors=...` -- so a prose-only
        # marker scored every one of them as zero nodes finished.
        if self._dist():
            n = int(c['nodes'])
            fin = re.compile(r'(?:KMEANS|GRAYSCOTT|GVW) \S*mode=|'
                             r'ALL GATES PASS|GATE: (?:PASS|FAIL)|'
                             r'iters in|steps in|passes,')
            ok, per_node_ms = 0, []
            for i in range(n):
                try:
                    with open(self._node_log(i), errors='replace') as f:
                        node_text = ansi.sub('', f.read())
                except OSError:
                    continue
                if not fin.search(node_text):
                    continue
                ok += 1
                mm = re.findall(r'(?:^|\s)ms=([0-9.]+)', node_text)
                if not mm:
                    mm = re.findall(r'in ([0-9.]+) ms', node_text)
                if mm:
                    per_node_ms.append(float(mm[-1]))
            stats['nodes_finished'] = ok
            # THE CELL IS AS SLOW AS ITS SLOWEST NODE. The concatenated
            # parse takes the LAST machine line, which is whichever node
            # happens to be appended last -- measured 583.7 ms on node 0
            # against 251.5 on node 1 for the same cell, a 2.3x difference
            # decided by nothing. The run is not over until both are.
            if len(per_node_ms) == n:
                stats['bench_ms'] = max(per_node_ms)
                stats['bench_ms_fastest_node'] = min(per_node_ms)
            if ok < n:
                stats['completed'] = 0
                stats['gates_pass'] = 0

        # A CRASH IS NOT A PASS. Every gate test above asks whether a
        # marker is PRESENT, so a run that printed an early marker and then
        # died reads as clean -- and the per-node veto below does not catch
        # it either, because that same early marker satisfies `fin`. These
        # four signatures are terminal by construction: the device-side
        # abort path, libstdc++'s terminate, a CUDA fatal, and a segfault.
        # Applied LAST so neither the clean-counters fallback nor anything
        # else can undo it.
        if any(sig in text for sig in ('DEVICE FATAL', 'terminate called',
                                       'CUDA Error', 'Segmentation fault')):
            stats['completed'] = 0
            stats['gates_pass'] = 0
            stats['crashed'] = 1

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
