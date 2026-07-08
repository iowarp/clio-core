from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo, LocalExecInfo
from jarvis_clio_core.container_utils import container_kwargs
import glob
import os
import re


class ClioRedisBench(Application):
    """
    Redis throughput benchmark — apples-to-apples mirror of clio_cte_bench.

    Drives the `clio_redis_bench` executable (built when
    CLIO_CORE_ENABLE_REDIS=ON) which shares its CLI/metrics with
    clio_cte_bench via bench_common.h. Each client thread holds its own
    hiredis connection and issues SET (Put) / GET (Get) with --depth
    pipelining, mirroring CTE PutBlob/GetBlob. Supports --time-limit
    (run N seconds) and --max-total-blobs (global keyspace split
    evenly across threads, cycling).

    Connects to a Redis server via REDIS_HOST / REDIS_PORT.

    Because clio_redis_bench emits the *identical* bench_common.h
    ``PrintResults`` report that clio_cte_bench does, the metric-capture
    half (``start`` writes ``<shared_dir>/bench_output.txt``; ``_get_stat``
    / ``_parse_output`` read it back) is the SAME contract clio_cte_bench
    uses, reused verbatim here so a sweep records the redis row into
    results.csv exactly the way the CTE row is recorded. bench_common.h
    remains the single source of truth for the metric math; this regex
    table is only the parsing half of that contract.
    """

    # How each metric folds when more than one rank/host reports (multi-node,
    # nprocs>1 with ppn=1 -> one bench rank per host). Identical to
    # clio_cte_bench so the CTE and redis multi-node CSVs aggregate the same
    # way: SUM cumulative system totals, MAX wall-clock, mean the rest.
    _FOLD_SUM = {
        'agg_bw_mbps',
        'agg_ops_per_sec',
        'total_data_mb',
        'total_ops',
    }
    _FOLD_MAX = {
        'time_max_us',
    }

    def _init(self):
        self.benchmark_executable = 'clio_redis_bench'
        self.output_file = None

    def _configure_menu(self):
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark test case to run',
                'type': str,
                'choices': ['Put', 'Get', 'PutGet'],
                'default': 'Put',
                'help': 'Put: SET, Get: GET, PutGet: SET+GET per key',
            },
            {
                'name': 'num_threads',
                'msg': 'Number of client threads (each its own connection)',
                'type': int,
                'default': 1,
                'help': 'Maps to clio_redis_bench --threads',
            },
            {
                'name': 'depth',
                'msg': 'Pipelined requests in flight per thread',
                'type': int,
                'default': 1,
                'help': 'Maps to --depth (Redis command pipelining)',
            },
            {
                'name': 'io_size',
                'msg': 'Size of each value',
                'type': str,
                'default': '1m',
                'help': 'k/K (KB), m/M (MB), g/G (GB). e.g. 4k, 1m',
            },
            {
                'name': 'io_count',
                'msg': 'Ops per thread (ignored when time_limit > 0)',
                'type': int,
                'default': 1000,
                'help': 'Maps to --io-count',
            },
            {
                'name': 'time_limit',
                'msg': 'Run each phase for N seconds instead of io_count',
                'type': int,
                'default': 0,
                'help': '0 = use io_count; >0 = run that many seconds '
                        '(--time-limit)',
            },
            {
                'name': 'max_total_blobs',
                'msg': 'Cap TOTAL distinct keys across all threads',
                'type': int,
                'default': 0,
                'help': '0 = unbounded; else global keyspace split evenly '
                        'across threads (--max-total-blobs). Total unique '
                        'bytes = max_total_blobs * io_size, independent of '
                        'num_threads.',
            },
            {
                'name': 'redis_host',
                'msg': 'Redis server host',
                'type': str,
                'default': '127.0.0.1',
                'help': 'Exported as REDIS_HOST',
            },
            {
                'name': 'redis_port',
                'msg': 'Redis server port',
                'type': int,
                'default': 6379,
                'help': 'Exported as REDIS_PORT',
            },
            {
                'name': 'nprocs',
                'msg': 'Number of processes (Pssh)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 1,
            },
            {
                'name': 'output_file',
                'msg': 'Optional user-facing copy of the benchmark results',
                'type': str,
                'default': '',
                'help': 'If set, the report is ALSO copied here (under '
                        'shared_dir). Metrics are always captured to '
                        'bench_output.txt regardless, for _get_stat.',
            },
            {
                'name': 'bin_dir',
                'msg': 'Directory holding clio_redis_bench (absolute). Empty = '
                       'resolve via PATH.',
                'type': str,
                'default': '',
                'help': 'For the apptainer deploy set this to the SIF spack-view '
                        'bin (/opt/iowarp-view/bin): jarvis runs the bench in a '
                        'shell whose PATH does NOT include the view bin, so the '
                        'bare name is "command not found". An ABSOLUTE path needs '
                        'no PATH lookup. Empty (bare-metal) uses PATH as before.',
            },
        ]

    def _configure(self, **kwargs):
        self.log("Configuring Redis benchmark application...")

        if self.config['test_case'] not in ['Put', 'Get', 'PutGet']:
            raise ValueError(
                f"Invalid test_case: {self.config['test_case']}")
        if self.config['num_threads'] <= 0:
            raise ValueError("num_threads must be > 0")
        if self.config['depth'] <= 0:
            raise ValueError("depth must be > 0")
        if (self.config['time_limit'] <= 0 and
                self.config['io_count'] <= 0):
            raise ValueError("need io_count > 0 or time_limit > 0")

        if self.config['output_file']:
            self.output_file = os.path.join(self.shared_dir,
                                            self.config['output_file'])
            self.log(f"Results will also be copied to: {self.output_file}")
        else:
            self.output_file = None

        # Redis connection for the benchmark process.
        self.setenv('REDIS_HOST', str(self.config['redis_host']))
        self.setenv('REDIS_PORT', str(self.config['redis_port']))

        self.log("Redis benchmark configuration completed")

    def start(self):
        """
        Run the redis benchmark, capturing its bench_common.h report to
        ``<shared_dir>/bench_output.txt`` so ``_get_stat`` can parse it
        afterwards. The sweep runner reloads a *fresh* package instance
        before calling ``_get_stat``, so an in-memory buffer would be lost —
        the on-disk file is the contract between start() and _get_stat().
        This mirrors clio_cte_bench.start() exactly.
        """
        self.log(f"Starting Redis benchmark: {self.config['test_case']}")

        # Resolve the executable. In the apptainer deploy bin_dir is the SIF
        # spack-view bin (/opt/iowarp-view/bin); an ABSOLUTE path sidesteps the
        # fact that jarvis's exec shell PATH omits the view bin (bare name ->
        # "command not found"). Empty bin_dir (bare-metal) keeps the bare name.
        bin_dir = str(self.config.get('bin_dir', '') or '')
        exe = os.path.join(bin_dir, self.benchmark_executable) if bin_dir \
            else self.benchmark_executable

        # Semantic-flag CLI (bench_common.h). Use --time-limit when set,
        # otherwise --io-count.
        cmd = [
            exe,
            '--op', str(self.config['test_case']),
            '--threads', str(self.config['num_threads']),
            '--depth', str(self.config['depth']),
            '--io-size', str(self.config['io_size']),
        ]
        if int(self.config['time_limit']) > 0:
            cmd += ['--time-limit', str(self.config['time_limit'])]
        else:
            cmd += ['--io-count', str(self.config['io_count'])]
        if int(self.config['max_total_blobs']) > 0:
            cmd += ['--max-total-blobs', str(self.config['max_total_blobs'])]

        cmd_str = ' '.join(cmd)
        output_path = os.path.join(self.shared_dir, 'bench_output.txt')

        if self.config['nprocs'] > 1:
            # Multi-rank / multi-node: with ppn=1 each host runs one bench
            # rank. Redirect each rank to a PER-HOST file so concurrent ranks
            # never clobber one shared file on the NFS shared_dir; _get_stat
            # globs + folds them. (Same per-host capture clio_cte_bench uses.)
            self.log(
                f"Running benchmark via Pssh: {self.config['nprocs']} procs, "
                f"{self.config['ppn']} per node (per-host output capture)"
            )
            # container_kwargs routes the command into the run's apptainer
            # instance (jarvis only wraps when exec_info.container is set;
            # nothing wraps automatically). The command stays RAW — the whole
            # string, redirect included, lands inside the wrap's
            # `bash -c '...'`, so `$(hostname)` expands per-host in-container
            # and the SIF's clio_redis_bench resolves. output_path is on the
            # auto-mounted shared_dir, so the report is host-visible.
            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=self.config['nprocs'],
                ppn=self.config['ppn'],
                **container_kwargs(self),
            )
            cmd_str = f'{cmd_str} > {output_path}.$(hostname) 2>&1'
        else:
            # Single-rank: container_kwargs routes the command into the run's
            # apptainer instance, where the SIF's clio_redis_bench exists
            # (it is absent host-side). Keep the command RAW — redirect and
            # all — so it rides inside the wrap's `bash -c '...'`. exe is the
            # absolute view-bin path (bin_dir); output_path is on the
            # auto-mounted shared_dir.
            self.log("Running benchmark locally (single rank)")
            exec_info = PsshExecInfo(env=self.mod_env, hostfile=self.hostfile,
                                     **container_kwargs(self))
            cmd_str = f'{cmd_str} > {output_path} 2>&1'

        self.log(f"Executing: {cmd_str}")
        Exec(cmd_str, exec_info).run()

        # Optional user-facing copy of the canonical single-rank report.
        if self.output_file and os.path.exists(output_path):
            try:
                import shutil
                shutil.copyfile(output_path, self.output_file)
            except OSError as e:
                self.log(f"Could not copy results to {self.output_file}: {e}")

        self.log(f"Redis benchmark completed. Output captured to: "
                 f"{output_path}")

    def stop(self):
        self.log("ClioRedisBench is an application - runs to completion")
        return True

    def clean(self):
        # The canonical results file parsed by _get_stat, plus any per-host
        # siblings (bench_output.txt.<hostname>) written in multi-node mode,
        # plus the optional user-facing copy.
        canonical = os.path.join(self.shared_dir, 'bench_output.txt')
        paths = [canonical] + glob.glob(canonical + '.*')
        if self.output_file:
            paths.append(self.output_file)
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    self.log(f"Removed: {path}")
                except Exception as e:
                    self.log(f"Error removing output file: {e}")

    # ------------------------------------------------------------------
    # Statistics collection — reused verbatim from clio_cte_bench, since
    # clio_redis_bench prints the identical bench_common.h report.
    # ------------------------------------------------------------------

    def _fold_host_stats(self, per_host_stats, stat_dict):
        """
        Aggregate per-host stat dicts into ``stat_dict``: SUM cumulative
        system totals, MAX wall-clock, mean everything else. A single host
        folds to itself, so the single-node path is unchanged.

        :param per_host_stats: List of ``<pkg_id>.<op>.<metric>`` -> value
                               dicts, one per host.
        :param stat_dict: Dict the framework serialises into results.csv.
        """
        from collections import defaultdict
        buckets = defaultdict(list)
        for host_stat in per_host_stats:
            for key, value in host_stat.items():
                buckets[key].append(value)
        for key, values in buckets.items():
            metric = key.rsplit('.', 1)[-1]
            if metric in self._FOLD_SUM:
                stat_dict[key] = sum(values)
            elif metric in self._FOLD_MAX:
                stat_dict[key] = max(values)
            else:
                stat_dict[key] = sum(values) / len(values)

    def _get_stat(self, stat_dict):
        """
        Parse the captured benchmark output and populate ``stat_dict``.

        Called by the jarvis_cd sweep runner after each pipeline run on a
        freshly-loaded package instance, so metrics are read from disk.
        Single-node parses ``<shared_dir>/bench_output.txt``; multi-node
        (nprocs>1, ppn=1) globs the per-host ``bench_output.txt.<hostname>``
        files and folds them via :meth:`_fold_host_stats`.

        :param stat_dict: Dict the framework serialises into results.csv;
                          keys are ``<pkg_id>.<operation>.<metric>``.
        """
        canonical = os.path.join(self.shared_dir, 'bench_output.txt')
        per_host = sorted(glob.glob(canonical + '.*'))
        if per_host:
            files = per_host
        elif os.path.exists(canonical):
            files = [canonical]
        else:
            self.log(f'No output file found at {canonical} (or per-host '
                     f'{canonical}.<hostname>)')
            return

        per_host_stats = []
        for fpath in files:
            with open(fpath, 'r') as f:
                output = f.read()
            if not output.strip():
                self.log(f'Output file is empty: {fpath}')
                continue
            host_stat = {}
            self._parse_output(output, host_stat)
            if host_stat:
                per_host_stats.append(host_stat)
            else:
                self.log(f'Warning: No metrics extracted from {fpath} '
                         f'({len(output)} bytes).')

        if not per_host_stats:
            return
        if len(per_host_stats) == 1:
            stat_dict.update(per_host_stats[0])
        else:
            self.log(f'Folding stats across {len(per_host_stats)} hosts '
                     f'(SUM aggregates, MAX wall-clock, mean the rest)')
            self._fold_host_stats(per_host_stats, stat_dict)

    def _parse_output(self, output, stat_dict):
        """
        Extract metrics from clio_redis_bench stdout into ``stat_dict``.

        The C++ binary emits its results via HLOG (bench_common.h
        ``PrintResults``), one labelled metric per line — the SAME wording
        clio_cte_bench parses, so this regex table is identical to that one.
        The patterns must match the wording/units printed by PrintResults.

        :param output: Raw captured benchmark output (stdout+stderr).
        :param stat_dict: Dict to populate with ``<pkg_id>.<op>.<metric>``.
        """
        # Strip ANSI escape codes from HLOG output.
        output = re.sub(r'\033\[[0-9;]*m', '', output)

        # Detect which test case header appeared (Put, Get, or PutGet).
        header_match = re.search(r'=== (\w+) Benchmark Results ===', output)
        operation = header_match.group(1).lower() if header_match else 'unknown'

        patterns = {
            'time_min_us': r'Time \(min\):\s+([\d.e+\-]+)\s+us',
            'time_max_us': r'Time \(max\):\s+([\d.e+\-]+)\s+us',
            'time_avg_us': r'Time \(avg\):\s+([\d.e+\-]+)\s+us',
            'bw_per_thread_min_mbps':
                r'Bandwidth per thread \(min\):\s+([\d.e+\-]+)\s+MB/s',
            'bw_per_thread_max_mbps':
                r'Bandwidth per thread \(max\):\s+([\d.e+\-]+)\s+MB/s',
            'bw_per_thread_avg_mbps':
                r'Bandwidth per thread \(avg\):\s+([\d.e+\-]+)\s+MB/s',
            'agg_bw_mbps': r'Aggregate bandwidth:\s+([\d.e+\-]+)\s+MB/s',
            'agg_ops_per_sec': r'Aggregate IOPS:\s+([\d.e+\-]+)',
            'ops_per_thread_avg_per_sec':
                r'IOPS per thread \(avg\):\s+([\d.e+\-]+)',
            'avg_latency_per_op_us':
                r'Avg latency per op:\s+([\d.e+\-]+)\s+us',
            'latency_stddev_us': r'Latency stddev:\s+([\d.e+\-]+)\s+us',
            'total_data_mb': r'Total data:\s+([\d.e+\-]+)\s+MB',
            'total_ops': r'Total ops:\s+(\d+)',
        }

        for metric, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                stat_dict[f'{self.pkg_id}.{operation}.{metric}'] = float(
                    match.group(1))
