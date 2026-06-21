"""
clio_xnode_bdev_bench — jarvis-cd wrapper for the cross-node clio_run
bdev benchmark binary (`clio_xnode_bdev_bench`).

The benchmark:
  - Calls MPI_Init, then clio::run::CLIO_INIT(kClient, false), so it needs
    a running clio_run daemon cluster (clio_runtime in the same pipeline).
  - Rank 0 creates a per-bench bdev pool (id 777.0) on every node via
    AsyncCreate(Broadcast, ..., kRam, 4 GiB).
  - Each rank issues AsyncWrite to DirectHash((my_node + 1) % num_nodes),
    i.e. the next node's container, so every op crosses the network.
  - CLI: --io-size <bytes> --num-ops <N> --depth <D>

This package mirrors the manual sbatch script
`/mnt/common/llogan/xnode_bdev_bench.sbatch`. The non-obvious bits:

  1) SLURM's --ntasks-per-node=1 means OpenMPI sees 1 slot/node from PMI.
     We MUST emit an explicit MPI hostfile with `slots=<ppn>` lines and
     pass it via `--hostfile`. Otherwise mpiexec rejects with
     "There are not enough slots available".

  2) The mpi flags (--mca btl_tcp_if_include enp47s0np0, --mca
     oob_tcp_if_include eno1, --prtemca prte_if_include eno1,
     --prtemca prte_tmpdir_base ..., --mca plm_rsh_agent "env -u
     LD_LIBRARY_PATH ssh") are not hard-coded here — they come from the
     pipeline-level `mpi_cmd` and `ssh_cmd` YAML keys, the same way the
     IOR + clio_runtime pipeline uses them.

  3) We propagate CLIO_SERVER_CONF / CTP_LOG_LEVEL / CLIO_IPC_MODE
     (clio_runtime publishes these via setenv on configure), TMPDIR,
     and CLIO_MEMFD_DIR (top-level env: in the pipeline yaml). The
     first four go through OpenMPI's `-x` (literal values). The fifth,
     CLIO_MEMFD_DIR, intentionally contains ${HOSTNAME} and must be
     expanded by the rank-side shell on each node, so we wrap the
     bench in `bash -c 'export CLIO_MEMFD_DIR=".../$HOSTNAME"; exec
     clio_xnode_bdev_bench ...'`. See _build_mpi_cmd for details.

Runs synchronously; the binary exits on completion. No teardown work in
stop() — clio_runtime owns the daemon lifecycle.
"""

import os

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo


class ClioXnodeBdevBench(Application):
    """
    Cross-node clio_run bdev write-throughput benchmark.
    """

    # ------------------------------------------------------------------
    # Init / menu
    # ------------------------------------------------------------------

    def _init(self):
        self.benchmark_executable = 'clio_xnode_bdev_bench'

    def _configure_menu(self):
        return [
            {
                'name': 'nprocs',
                'msg': 'Total number of MPI ranks (across all nodes)',
                'type': int,
                'default': 96,
            },
            {
                'name': 'ppn',
                'msg': 'Ranks per node (used to write slots=N into the '
                       'MPI hostfile)',
                'type': int,
                'default': 24,
            },
            {
                'name': 'io_size',
                'msg': 'Bytes per AsyncWrite (passed as --io-size)',
                'type': int,
                'default': 1 << 20,   # 1 MiB
            },
            {
                'name': 'num_ops',
                'msg': 'AsyncWrites per rank (passed as --num-ops)',
                'type': int,
                'default': 512,
            },
            {
                'name': 'depth',
                'msg': 'Outstanding writes per rank (passed as --depth)',
                'type': int,
                'default': 16,
            },
            {
                'name': 'bind_to',
                'msg': 'OpenMPI --bind-to value (none is what the manual '
                       'sbatch script uses; clio_run spawns its own threads)',
                'type': str,
                'default': 'none',
            },
            {
                'name': 'extra_args',
                'msg': 'Free-form extra args appended to the bench command',
                'type': str,
                'default': '',
            },
        ]

    def _configure(self, **kwargs):
        # Sanity-check the basic numerics.
        for key in ('nprocs', 'ppn', 'io_size', 'num_ops', 'depth'):
            if self.config[key] is None or int(self.config[key]) <= 0:
                raise ValueError(
                    f"clio_xnode_bdev_bench: {key} must be a positive int "
                    f"(got {self.config[key]!r})")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_mpi_hostfile(self):
        """
        Write an MPI hostfile with explicit `slots=<ppn>` lines, using
        the host list that the pipeline (or scheduler block) made
        available via self.hostfile.

        Returns the absolute path.
        """
        hf = self.hostfile
        hosts = list(hf.hosts) if hf else []
        if not hosts:
            raise RuntimeError(
                "clio_xnode_bdev_bench: no hosts in self.hostfile — does the "
                "scheduler block populate hostfile.txt?")

        ppn = int(self.config['ppn'])
        path = os.path.join(self.shared_dir, 'mpi_hostfile')
        os.makedirs(self.shared_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as fp:
            for h in hosts:
                fp.write(f"{h} slots={ppn}\n")
        self.log(f"Wrote MPI hostfile (slots={ppn}) at {path}")
        with open(path, 'r', encoding='utf-8') as fp:
            self.log(fp.read().rstrip())
        return path

    def _build_mpi_cmd(self, mpi_hostfile_path):
        """
        Build the full mpiexec command.

        The base launcher (``mpiexec`` plus all --mca / --prtemca flags)
        comes from the pipeline-level `mpi_cmd` YAML key. The rsh agent
        comes from `ssh_cmd`. Both fall back to plain `mpiexec` if
        unset, but the ares pipelines always set them.

        Per-rank env: ``CLIO_SERVER_CONF`` / ``CTP_LOG_LEVEL`` /
        ``CLIO_IPC_MODE`` / ``TMPDIR`` go through OpenMPI's `-x` —
        verbatim values, no shell expansion. ``CLIO_MEMFD_DIR``, by
        contrast, intentionally embeds the literal token ``$HOSTNAME``
        (the pipeline YAML's `env:` block writes
        ``/home/llogan/iowarp_clio_tmp_${HOSTNAME}``) so that it can
        resolve to the *local* hostname on every node — needed because
        clio_run's per-PID memfd symlinks point into /proc/<pid>/fd and
        a single NFS-shared dir clobbers them across nodes
        (project_chi_memfd_dir_must_be_node_local). ``-x`` would
        forward the literal token, so each rank would look for the
        unsubstituted ``${HOSTNAME}`` path and `shm_open` ENOENT-fails.
        Instead we wrap the bench in ``bash -lc 'export
        CLIO_MEMFD_DIR=...$HOSTNAME...; exec <bench> ...'`` — the rank-
        side shell expands ``$HOSTNAME`` to its own node. Same trick
        the per-node clio_run daemon spawn uses (PsshExec's
        ``KEY=value <cmd>`` prefix, eval'd by the remote shell).
        """
        pipeline = self.pipeline
        mpi_launcher = (getattr(pipeline, 'mpi_cmd', None)
                        or 'mpiexec')
        ssh_cmd = getattr(pipeline, 'ssh_cmd', None)

        nprocs = int(self.config['nprocs'])
        bind_to = self.config['bind_to']

        # Env vars that don't depend on hostname — safe to forward with
        # -x as literal values from this process's env.
        forwarded_env = [
            'CLIO_SERVER_CONF',
            'CTP_LOG_LEVEL',
            'CLIO_IPC_MODE',
            'TMPDIR',
        ]
        x_flags = ' '.join(f'-x {k}' for k in forwarded_env)

        bench_args = (
            f'--io-size {int(self.config["io_size"])} '
            f'--num-ops {int(self.config["num_ops"])} '
            f'--depth {int(self.config["depth"])}'
        )
        if self.config['extra_args']:
            bench_args += f' {self.config["extra_args"]}'

        # The mod_env CLIO_MEMFD_DIR is the un-expanded template from
        # the pipeline yaml (e.g.
        # ``/home/llogan/iowarp_clio_tmp_${HOSTNAME}``). Each rank
        # must resolve ${HOSTNAME} to its own node, which means letting
        # the *rank-side* bash do the expansion. Use double quotes in
        # the inner script so $HOSTNAME expands there. The outer
        # invocation single-quotes the whole inner script so the
        # head-node shell (this Python process's shell=True subprocess)
        # passes $HOSTNAME through verbatim.
        memfd_template = self.mod_env.get(
            'CLIO_MEMFD_DIR',
            '/tmp/clio_$USER')
        # Normalize ${HOSTNAME} → $HOSTNAME (bash treats them the same;
        # we use the bare form so simple double-quoted expansion works
        # without brace gymnastics inside a single-quoted outer wrap).
        memfd_template = memfd_template.replace('${HOSTNAME}', '$HOSTNAME')
        # Inner script: double-quote the path so the rank-side bash
        # expands $HOSTNAME. We then exec the bench so signals / exit
        # codes pass through cleanly.
        inner = (
            f'export CLIO_MEMFD_DIR="{memfd_template}"; '
            f'exec {self.benchmark_executable} {bench_args}'
        )
        # Wrap in single-quotes for the outer mpiexec line so the
        # head-node shell does NOT expand anything inside (especially
        # not $HOSTNAME — that would resolve to the head node, breaking
        # rank-side substitution). Escape any embedded single quotes.
        inner_escaped = inner.replace("'", "'\\''")
        bench_invocation = f"bash -c '{inner_escaped}'"

        parts = [mpi_launcher]
        if ssh_cmd:
            parts.append(f'--mca plm_rsh_agent "{ssh_cmd}"')
        if bind_to:
            parts.append(f'--bind-to {bind_to}')
        parts.extend([
            f'-np {nprocs}',
            f'--hostfile {mpi_hostfile_path}',
            x_flags,
            bench_invocation,
        ])
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """
        Build the MPI hostfile, then launch the bench under mpiexec and
        wait for it to finish. The aggregate result line
        (``CLUSTER ranks=... aggregate=... MiB/s``) lands in the SLURM
        stdout because rank 0's HLOG(kInfo, ...) goes to stdout.
        """
        mpi_hostfile = self._write_mpi_hostfile()
        cmd = self._build_mpi_cmd(mpi_hostfile)
        self.log(f"Bench cmd: {cmd}")

        # Run with the full pipeline env (CLIO_SERVER_CONF + TMPDIR +
        # CLIO_MEMFD_DIR + module-loaded mpiexec on PATH).
        Exec(cmd, LocalExecInfo(env=self.mod_env)).run()

    def stop(self):
        # Benchmark binary exits on its own; clio_runtime owns daemon
        # teardown. Nothing to do here.
        return True

    def clean(self):
        # Best-effort: remove the per-pkg shared dir contents (the
        # generated mpi_hostfile). Don't touch anything outside.
        try:
            mpi_hostfile = os.path.join(self.shared_dir, 'mpi_hostfile')
            if os.path.exists(mpi_hostfile):
                os.remove(mpi_hostfile)
        except OSError as exc:
            self.log(f"clio_xnode_bdev_bench clean: {exc}")
        return True
