from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo, LocalExecInfo
import os
import re

class ClioCteBench(Application):
    """
    CTE Core Benchmark Application

    This application runs benchmarks for Put, Get, and GetTagSize operations
    in the Content Transfer Engine (CTE) with MPI support for parallel I/O.

    The benchmark measures:
    - Put operations: Write data to CTE
    - Get operations: Read data from CTE
    - PutGet operations: Combined write and read operations

    It supports async operation depth configuration for testing concurrent I/O.
    """

    def _init(self):
        """
        Initialize the ClioCteBench application.

        This method is called during application initialization.
        """
        self.benchmark_executable = 'clio_cte_bench'
        self.output_file = None

    def _configure_menu(self):
        """
        Configure the application menu.

        Returns:
            List[Dict]: Configuration menu options for the benchmark.
        """
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark test case to run',
                'type': str,
                'choices': ['Put', 'Get', 'PutGet'],
                'default': 'Put',
                'help': 'Put: Write benchmark, Get: Read benchmark, PutGet: Combined write+read benchmark'
            },
            {
                'name': 'num_threads',
                'msg': 'Number of worker threads',
                'type': int,
                'default': 4,
                'help': 'Number of worker threads for parallel I/O (e.g., 4)'
            },
            {
                'name': 'depth',
                'msg': 'Number of async requests per thread',
                'type': int,
                'default': 4,
                'help': 'Number of concurrent async I/O operations per thread (e.g., 4 means 4 operations in flight per thread)'
            },
            {
                'name': 'io_size',
                'msg': 'Size of I/O operations',
                'type': str,
                'default': '1m',
                'help': 'I/O size with suffix: k/K (KB), m/M (MB), g/G (GB). Examples: 4k, 1m, 2g'
            },
            {
                'name': 'io_count',
                'msg': 'Number of I/O operations to generate per node',
                'type': int,
                'default': 100,
                'help': 'Total number of I/O operations each MPI rank will perform'
            },
            {
                'name': 'time_limit',
                'msg': 'Run each phase for N seconds instead of io_count',
                'type': int,
                'default': 0,
                'help': '0 = use io_count; >0 = run that many seconds '
                        '(maps to clio_cte_bench --time-limit)'
            },
            {
                'name': 'max_total_blobs',
                'msg': 'Cap TOTAL distinct blobs across all threads',
                'type': int,
                'default': 0,
                'help': '0 = unbounded; else global keyspace split evenly '
                        'across threads (maps to --max-total-blobs). Total '
                        'unique bytes = max_total_blobs * io_size, '
                        'independent of num_threads.'
            },
            {
                'name': 'nprocs',
                'msg': 'Number of MPI processes',
                'type': int,
                'default': 1,
                'help': 'Number of MPI processes to use for parallel I/O'
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 1,
                'help': 'Number of MPI processes per node'
            },
            {
                'name': 'output_file',
                'msg': 'Output file for benchmark results',
                'type': str,
                'default': '',
                'help': 'Path to save benchmark results. If empty, results are printed to stdout'
            },
            {
                'name': 'init_runtime',
                'msg': 'Initialize Clio runtime (otherwise assumes runtime already running)',
                'type': bool,
                'default': False,
                'help': 'Set to True to initialize runtime, False to only initialize client'
            },
            {
                'name': 'query_type',
                'msg': 'PoolQuery used by the bench (maps to --query-type)',
                'type': str,
                'choices': ['local', 'dynamic', 'direct0'],
                'default': 'local',
                'help': 'local: PoolQuery::Local (co-located target); '
                        'dynamic: PoolQuery::Dynamic; direct0: '
                        'PoolQuery::DirectHash(0) -- explicit non-Local '
                        'routing to node 0, useful for measuring '
                        'cross-node CTE paths.'
            }
        ]

    def _configure(self, **kwargs):
        """
        Configure the CTE benchmark application with provided keyword arguments.

        This method sets up the benchmark executable path and output file.

        Args:
            **kwargs: Configuration arguments from _configure_menu.
        """
        self.log("Configuring CTE benchmark application...")

        # Validate test_case
        if self.config['test_case'] not in ['Put', 'Get', 'PutGet']:
            raise ValueError(f"Invalid test_case: {self.config['test_case']}. Must be Put, Get, or PutGet")

        # Validate num_threads
        if self.config['num_threads'] <= 0:
            raise ValueError(f"Invalid num_threads: {self.config['num_threads']}. Must be > 0")

        # Validate depth
        if self.config['depth'] <= 0:
            raise ValueError(f"Invalid depth: {self.config['depth']}. Must be > 0")

        # Validate io_count (only required when not time-limited)
        if (int(self.config['time_limit']) <= 0 and
                self.config['io_count'] <= 0):
            raise ValueError(
                f"Invalid io_count: {self.config['io_count']}. Must be > 0 "
                f"(or set time_limit > 0)")

        # Validate io_size format (should have k/K, m/M, g/G suffix or be a number)
        io_size_str = str(self.config['io_size']).lower()
        if not (io_size_str[-1] in ['k', 'm', 'g'] or io_size_str.isdigit()):
            self.log(f"Warning: io_size '{self.config['io_size']}' should end with k/K, m/M, or g/G suffix")

        # Set output file if specified
        if self.config['output_file']:
            self.output_file = os.path.join(self.shared_dir, self.config['output_file'])
            self.log(f"Benchmark results will be saved to: {self.output_file}")
        else:
            self.output_file = None
            self.log("Benchmark results will be printed to stdout")

        # Set environment variables for the benchmark
        self.setenv('CTE_BENCH_TEST_CASE', self.config['test_case'])
        self.setenv('CTE_BENCH_DEPTH', str(self.config['depth']))
        self.setenv('CTE_BENCH_IO_SIZE', str(self.config['io_size']))
        self.setenv('CTE_BENCH_IO_COUNT', str(self.config['io_count']))

        # Set CLIO_WITH_RUNTIME environment variable based on configuration
        if self.config['init_runtime']:
            self.setenv('CLIO_WITH_RUNTIME', '1')
            self.log("Runtime initialization enabled (CLIO_WITH_RUNTIME=1)")
        else:
            self.setenv('CLIO_WITH_RUNTIME', '0')
            self.log("Runtime initialization disabled (CLIO_WITH_RUNTIME=0)")

        self.log("CTE benchmark configuration completed successfully")

    def start(self):
        """
        Run the CTE benchmark application.

        This method executes the benchmark with MPI support and configured
        parameters. The benchmark's stdout+stderr (the HLOG results report)
        is always captured to ``<shared_dir>/bench_output.txt`` so that
        ``_get_stat`` can parse the metrics afterwards. The sweep runner
        (jarvis_cd pipeline_test) re-loads a *fresh* package instance before
        calling ``_get_stat``, so an in-memory buffer would be lost -- the
        on-disk file is the contract between start() and _get_stat().
        """
        self.log(f"Starting CTE benchmark: {self.config['test_case']}")

        # Semantic-flag CLI (bench_common.h). Use --time-limit when set,
        # otherwise --io-count. (The binary still accepts the legacy
        # positional form, but flags make time_limit/max_total_blobs
        # possible.)
        cmd = [
            self.benchmark_executable,
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
        cmd += ['--query-type', str(self.config['query_type'])]

        cmd_str = ' '.join(cmd)
        output_path = os.path.join(self.shared_dir, 'bench_output.txt')

        if self.config['nprocs'] > 1:
            # Multi-rank: redirect in the shell so every rank's output lands
            # in the shared file (shared_dir is on a shared FS on Ares).
            self.log(
                f"Running benchmark via Pssh: {self.config['nprocs']} procs, "
                f"{self.config['ppn']} per node"
            )
            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=self.config['nprocs'],
                ppn=self.config['ppn'],
            )
            cmd_str += f' > {output_path} 2>&1'
        else:
            # Single-rank: pipe both streams to the file via LocalExecInfo
            # (matches the proven archived capture path; HLOG -> stderr).
            self.log("Running benchmark locally (single rank)")
            exec_info = LocalExecInfo(
                env=self.mod_env,
                pipe_stdout=output_path,
                pipe_stderr=output_path,
            )

        self.log(f"Executing: {cmd_str}")
        Exec(cmd_str, exec_info).run()
        self.log(f"Benchmark completed. Output captured to: {output_path}")

    def stop(self):
        """
        Stop the benchmark application.

        Since this is an application that runs to completion, this method
        is typically not needed but is provided for consistency.
        """
        self.log("ClioCteBench is an application - it runs to completion")
        return True

    def clean(self):
        """
        Clean up benchmark output files.
        """
        self.log("Cleaning up CTE benchmark output files...")

        # The canonical results file parsed by _get_stat.
        paths = [os.path.join(self.shared_dir, 'bench_output.txt')]
        # Plus the optional user-facing copy, if one was configured.
        if self.output_file:
            paths.append(self.output_file)

        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    self.log(f"Removed benchmark output file: {path}")
                except Exception as e:
                    self.log(f"Error removing output file {path}: {e}")

        self.log("CTE benchmark cleanup completed")

    # ------------------------------------------------------------------
    # Statistics collection
    # ------------------------------------------------------------------

    def _get_stat(self, stat_dict):
        """
        Parse the captured benchmark output and populate ``stat_dict``.

        Called by the jarvis_cd sweep runner after each pipeline run (on a
        freshly-loaded package instance), which is why the metrics are read
        back from ``<shared_dir>/bench_output.txt`` rather than from an
        in-memory buffer. Each extracted metric becomes a results.csv column.

        :param stat_dict: Dict the framework serialises into results.csv;
                          keys are ``<pkg_id>.<operation>.<metric>``.
        """
        output_path = os.path.join(self.shared_dir, 'bench_output.txt')
        if not os.path.exists(output_path):
            self.log(f'No output file found at {output_path}')
            return

        with open(output_path, 'r') as f:
            output = f.read()

        if not output.strip():
            self.log(f'Output file is empty: {output_path}')
            return

        before_count = len(stat_dict)
        self._parse_output(output, stat_dict)
        after_count = len(stat_dict)
        if after_count == before_count:
            self.log(f'Warning: No metrics extracted from {output_path} '
                     f'({len(output)} bytes). '
                     f'Check if benchmark results are present in output.')

    def _parse_output(self, output, stat_dict):
        """
        Extract metrics from clio_cte_bench stdout into ``stat_dict``.

        The C++ binary emits its results via HLOG (bench_common.h
        ``PrintResults``), one labelled metric per line. This regex table is
        the parsing half of that contract -- the patterns must match the
        wording/units printed by PrintResults exactly.

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
