"""
This module drives an ``fio`` workload against a mounted filesystem (here a
JuiceFS FUSE mount) and parses fio's JSON report into ``results.csv``
columns.

It mirrors the start()/_get_stat() contract used by ``clio_cte_bench``:
``start()`` captures the benchmark report to a file under ``shared_dir`` and
``_get_stat()`` re-reads that file. This indirection is required because the
jarvis_cd sweep runner reloads a *fresh* package instance before calling
``_get_stat`` -- an in-memory buffer would be lost, so the on-disk fio JSON
is the contract between the two methods.

Parallelism is expressed via fio ``--numjobs`` + ``--thread`` (real threads
sharing one address space), which is the closest match to the project's
"threads" knob.
"""
import os
import json
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_clio_core.container_utils import (
    container_kwargs, eff_hostfile, single_instance_menu_opt)


class JuicefsBench(Application):
    """
    fio-based throughput/latency benchmark for a POSIX mountpoint, with a
    JSON-parsing ``_get_stat`` that populates results.csv.
    """

    def _init(self):
        """
        Initialize package state.
        """
        self.json_path = None

    def _configure_menu(self):
        """
        Configure the application menu.

        :return: List(dict)
        """
        return [
            {
                'name': 'target_dir',
                'msg': 'Directory to run fio in (the JuiceFS mountpoint)',
                'type': str,
                'default': '${HOME}/juicefs_mnt',
            },
            {
                'name': 'io_size',
                'msg': 'fio block size (--bs), e.g. 4k/16k/64k/128k',
                'type': str,
                'default': '4k',
            },
            {
                'name': 'num_threads',
                'msg': 'Number of fio jobs/threads (--numjobs --thread)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'runtime',
                'msg': 'Seconds to run (fio --runtime --time_based)',
                'type': int,
                'default': 60,
            },
            {
                'name': 'mode',
                'msg': 'fio rw mode',
                'type': str,
                'choices': ['write', 'read', 'randwrite', 'randread'],
                'default': 'write',
            },
            {
                'name': 'size_per_job',
                'msg': 'File size per job (--size)',
                'type': str,
                'default': '1G',
            },
            {
                'name': 'ioengine',
                'msg': 'fio I/O engine (--ioengine)',
                'type': str,
                'default': 'psync',
            },
            {
                'name': 'direct',
                'msg': 'Use O_DIRECT (often unsupported on FUSE)',
                'type': bool,
                'default': False,
            },
            {
                'name': 'fallocate',
                'msg': 'fio --fallocate mode. Use "none" for FUSE mounts '
                       'that reject fallocate (e.g. the CTE libfuse mount); '
                       '"native" preserves fio\'s default.',
                'type': str,
                'choices': ['none', 'native', 'posix', 'keep'],
                'default': 'native',
            },
            {
                'name': 'fio_bin',
                'msg': 'Path to the fio binary',
                'type': str,
                'default': 'fio',
            },
            {
                'name': 'output_file',
                'msg': 'fio JSON report filename (under shared_dir)',
                'type': str,
                'default': 'fio_output.json',
            },
            # Head-node-only pinning: these fio drivers are single-client
            # baselines (one head-node client writing to NFS / the JuiceFS
            # mount). On a multi-node pipeline, fanning fio out to every node
            # makes all N nodes clobber the one shared-dir JSON report and, for
            # jfs_fio, run against a mountpoint that only exists on the head.
            single_instance_menu_opt(),
        ]

    def _configure(self, **kwargs):
        """
        Validate config and resolve the JSON report path.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        if self.config['num_threads'] <= 0:
            raise ValueError('juicefs_bench: num_threads must be > 0')
        if (int(self.config['runtime']) <= 0):
            raise ValueError('juicefs_bench: runtime must be > 0')
        if self.config['mode'] not in ('write', 'read', 'randwrite',
                                       'randread'):
            raise ValueError(
                f"juicefs_bench: invalid mode {self.config['mode']}")
        self.json_path = os.path.join(self.shared_dir,
                                      self.config['output_file'])

    def _op_label(self):
        """
        Map the fio rw mode onto the fio JSON section ('read' or 'write').

        :return: str
        """
        return 'read' if 'read' in self.config['mode'] else 'write'

    def start(self):
        """
        Run fio against the target directory, writing a JSON report to
        ``<shared_dir>/<output_file>`` (the start->_get_stat contract).

        :return: None
        """
        target_dir = os.path.expanduser(
            os.path.expandvars(str(self.config['target_dir'])))
        # Create the target dir in the deployment context. For the container
        # deploy target_dir may be an in-container path (a FUSE mountpoint
        # like /tmp/cte_mnt or /tmp/jfs_mnt, which under tmp_bind_root is a
        # different directory from the host's /tmp), so it must be created
        # via a wrapped Exec, not host-side os.makedirs. Bare-metal it is a
        # harmless local mkdir -p. (json_path below stays on the auto-mounted
        # shared_dir -- written by fio, host-visible.)
        Exec(f'mkdir -p {target_dir}',
             PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()
        json_path = os.path.join(self.shared_dir, self.config['output_file'])

        cmd = [
            self.config['fio_bin'],
            '--name=jfs',
            f'--directory={target_dir}',
            f"--rw={self.config['mode']}",
            f"--bs={self.config['io_size']}",
            f"--numjobs={self.config['num_threads']}",
            '--thread',
            f"--runtime={self.config['runtime']}",
            '--time_based',
            f"--size={self.config['size_per_job']}",
            f"--ioengine={self.config['ioengine']}",
            f"--direct={1 if self.config['direct'] else 0}",
            f"--fallocate={self.config['fallocate']}",
            '--group_reporting',
            '--output-format=json',
            f'--output={json_path}',
        ]
        cmd_str = ' '.join(cmd)
        self.log(f'Executing: {cmd_str}')
        # Wrapped: fio must run inside the instance so it writes through the
        # FUSE mounts that live in the instance's mount namespace (and comes
        # from the SIF, not the host env).
        Exec(cmd_str, PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                                   **container_kwargs(self))).run()
        self.log(f'fio completed. JSON report at: {json_path}')

    def stop(self):
        """
        fio runs to completion; nothing to stop.

        :return: bool
        """
        return True

    def clean(self):
        """
        Remove the fio JSON report.

        :return: None
        """
        json_path = os.path.join(self.shared_dir, self.config['output_file'])
        if os.path.exists(json_path):
            try:
                os.remove(json_path)
                self.log(f'Removed {json_path}')
            except OSError as e:
                self.log(f'Error removing {json_path}: {e}')

    # ------------------------------------------------------------------
    # Statistics collection
    # ------------------------------------------------------------------

    def _get_stat(self, stat_dict):
        """
        Parse the fio JSON report and populate ``stat_dict``.

        Called by the sweep runner on a freshly-loaded package instance, so
        metrics are read back from ``<shared_dir>/<output_file>`` rather than
        from memory. Each extracted metric becomes a results.csv column keyed
        ``<pkg_id>.<op>.<metric>``.

        :param stat_dict: Dict the framework serialises into results.csv.
        :return: None
        """
        json_path = os.path.join(self.shared_dir, self.config['output_file'])
        if not os.path.exists(json_path):
            self.log(f'No fio report found at {json_path}')
            return
        with open(json_path, 'r') as f:
            raw = f.read()
        if not raw.strip():
            self.log(f'fio report is empty: {json_path}')
            return
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            self.log(f'Could not parse fio JSON {json_path}: {e}')
            return

        self._parse_output(data, stat_dict)

    def _parse_output(self, data, stat_dict):
        """
        Extract bandwidth/IOPS/latency from a parsed fio JSON document.

        :param data: Parsed fio JSON (dict).
        :param stat_dict: Dict to populate with ``<pkg_id>.<op>.<metric>``.
        :return: None
        """
        jobs = data.get('jobs') or []
        if not jobs:
            self.log('fio JSON has no jobs array; no metrics extracted')
            return
        op = self._op_label()
        section = jobs[0].get(op, {})
        prefix = f'{self.pkg_id}.{op}'
        before = len(stat_dict)

        # bw is reported in KiB/s.
        if 'bw' in section:
            stat_dict[f'{prefix}.agg_bw_mbps'] = section['bw'] / 1024.0
        if 'iops' in section:
            stat_dict[f'{prefix}.iops'] = float(section['iops'])
        lat_ns = section.get('lat_ns', {})
        if 'mean' in lat_ns:
            stat_dict[f'{prefix}.lat_mean_us'] = lat_ns['mean'] / 1000.0
        clat_pct = section.get('clat_ns', {}).get('percentile', {})
        if '99.000000' in clat_pct:
            stat_dict[f'{prefix}.lat_p99_us'] = \
                clat_pct['99.000000'] / 1000.0
        if 'io_bytes' in section:
            stat_dict[f'{prefix}.total_io_mb'] = \
                section['io_bytes'] / (1024.0 * 1024.0)

        if len(stat_dict) == before:
            self.log(f'Warning: no fio metrics extracted for op={op}; '
                     f'check the fio JSON schema in {self.json_path}')
