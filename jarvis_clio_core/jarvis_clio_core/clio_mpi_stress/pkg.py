"""
Co-located MPI load for the DTSchedule evaluations.

Starts ``dtschedule_mpi_stress`` (context-transfer-engine/dtschedule/tools/
mpi_stress) on every host of the pipeline before the workflow and stops it
afterwards, so the CTE runtime and the workflow tasks compete with a
bulk-synchronous MPI job for cores, memory bandwidth and the network.

Defaults saturate the node: one rank per hardware thread (``ppn``), bound
to its thread. Each rank holds ``mem_mb`` of memory, computes for
``compute_ms`` and then exchanges a ``halo_kb`` halo with its ring
neighbours plus an allreduce. Put this package BEFORE the workload in the
pipeline so it is running when the workload starts and is stopped after it.

Stats: ``<pkg>.ranks``, ``<pkg>.iters_min``, ``<pkg>.iters_max``,
``<pkg>.iter_ms_p50``, ``<pkg>.wall_s`` (from rank 0's summary line).
"""
import os
import re
import time

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo

SUMMARY_RE = re.compile(
    r'mpi_stress ranks=(\d+) iters=(\d+)\.\.(\d+) '
    r'iter_ms_p50=([0-9.]+) wall_s=([0-9.]+)')


class ClioMpiStress(Application):
    """Run a CPU/memory/network-saturating MPI job next to the workload."""

    def _init(self):
        """No state beyond the config."""

    def _configure_menu(self):
        """Menu: launcher, placement, and the per-rank load shape."""
        return [
            {'name': 'mpirun', 'msg': 'mpirun executable (absolute path; its '
             'parent dir is passed as --prefix for remote daemons)',
             'type': str, 'default': 'mpirun'},
            {'name': 'launch_agent', 'msg': 'Remote daemon for Open MPI 5 '
             '(PRRTE prted, a separate spack package); empty = mpirun default',
             'type': str, 'default': ''},
            {'name': 'binary', 'msg': 'dtschedule_mpi_stress executable',
             'type': str, 'default': 'dtschedule_mpi_stress'},
            {'name': 'ppn', 'msg': 'Ranks per node (default: every hardware '
             'thread of a 40-thread ares node)', 'type': int, 'default': 40},
            {'name': 'mem_mb', 'msg': 'Working set held per rank (MB)',
             'type': int, 'default': 256},
            {'name': 'compute_ms', 'msg': 'Compute phase per iteration (ms)',
             'type': float, 'default': 50.0},
            {'name': 'halo_kb', 'msg': 'Halo per neighbour per iteration (KB)',
             'type': int, 'default': 256},
            {'name': 'duration_s', 'msg': 'Stop after this long (0 = run '
             'until the package is stopped)', 'type': float, 'default': 0.0},
            {'name': 'net_if', 'msg': 'CIDR the MPI traffic must use (e.g. '
             '172.20.0.0/16 to share the 1 GbE link with CTE); empty = any',
             'type': str, 'default': ''},
            {'name': 'warmup_s', 'msg': 'Seconds to wait after launch so the '
             'load is running before the workload starts', 'type': float,
             'default': 5.0},
        ]

    def _configure(self, **kwargs):
        """Store the config; nothing to generate up front."""
        super()._configure(**kwargs)

    def _paths(self):
        """Return (hostfile path, log path) under the shared dir."""
        return (os.path.join(self.shared_dir, 'mpi_stress_hosts.txt'),
                os.path.join(self.shared_dir, 'mpi_stress.log'))

    def _write_hostfile(self, path):
        """Write an Open MPI hostfile with ``slots=ppn`` per pipeline host."""
        ppn = int(self.config['ppn'])
        with open(path, 'w') as fh:
            for host in self.hostfile.hosts:
                fh.write(f'{host} slots={ppn}\n')

    def _mpirun_cmd(self, hostfile, log):
        """Build the single-line mpirun command (output to ``log``)."""
        c = self.config
        mpirun = c['mpirun']
        prefix = ''
        if c.get('launch_agent'):
            prefix = f'--launch-agent {c["launch_agent"]} '
        elif os.path.isabs(mpirun):
            prefix = f'--prefix {os.path.dirname(os.path.dirname(mpirun))} '
        net = ''
        if c.get('net_if'):
            net = (f'--mca btl_tcp_if_include {c["net_if"]} '
                   f'--mca oob_tcp_if_include {c["net_if"]} ')
        nranks = int(c['ppn']) * len(self.hostfile.hosts)
        return (f'{mpirun} {prefix}-np {nranks} --hostfile {hostfile} '
                f'--map-by ppr:{int(c["ppn"])}:node:HWTCPUS '
                f'--bind-to hwthread {net}'
                f'{c["binary"]} --mem-mb {int(c["mem_mb"])} '
                f'--compute-ms {float(c["compute_ms"])} '
                f'--halo-kb {int(c["halo_kb"])} '
                f'--duration-s {float(c["duration_s"])} > {log} 2>&1')

    def start(self):
        """Launch the MPI job asynchronously and let it warm up."""
        hostfile, log = self._paths()
        self._write_hostfile(hostfile)
        cmd = self._mpirun_cmd(hostfile, log)
        self.log(f'mpi_stress: {cmd}')
        Exec(cmd, LocalExecInfo(env=self.mod_env, exec_async=True)).run()
        time.sleep(float(self.config['warmup_s']))

    def stop(self):
        """SIGTERM the ranks (they agree to stop and print a summary)."""
        Exec('pkill -TERM -f "[d]tschedule_mpi_stress" || true',
             PsshExecInfo(env=self.mod_env, hostfile=self.hostfile)).run()
        _, log = self._paths()
        for _ in range(60):
            if self._summary(log) is not None:
                break
            time.sleep(1)
        Exec('pkill -9 -f "[d]tschedule_mpi_stress" || true',
             PsshExecInfo(env=self.mod_env, hostfile=self.hostfile)).run()

    def clean(self):
        """Remove the generated hostfile and log."""
        for path in self._paths():
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _summary(log):
        """Parse rank 0's summary line, or None if it is not written yet."""
        if not os.path.exists(log):
            return None
        with open(log) as fh:
            for line in fh:
                m = SUMMARY_RE.search(line)
                if m:
                    return m
        return None

    def _get_stat(self, stat_dict):
        """Report the MPI job's progress (iterations and iteration time)."""
        pid = self.pkg_id
        m = self._summary(self._paths()[1])
        if m is None:
            stat_dict[f'{pid}.status'] = 'no_summary'
            return
        stat_dict[f'{pid}.ranks'] = int(m.group(1))
        stat_dict[f'{pid}.iters_min'] = int(m.group(2))
        stat_dict[f'{pid}.iters_max'] = int(m.group(3))
        stat_dict[f'{pid}.iter_ms_p50'] = float(m.group(4))
        stat_dict[f'{pid}.wall_s'] = float(m.group(5))
