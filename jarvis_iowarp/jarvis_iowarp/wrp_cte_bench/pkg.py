from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo
import os

class WrpCteBench(Application):
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
        Initialize the WrpCteBench application.

        This method is called during application initialization.
        """
        self.benchmark_executable = 'wrp_cte_bench'
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
                'msg': 'Initialize Chimaera runtime (otherwise assumes runtime already running)',
                'type': bool,
                'default': False,
                'help': 'Set to True to initialize runtime, False to only initialize client'
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

        # Validate io_count
        if self.config['io_count'] <= 0:
            raise ValueError(f"Invalid io_count: {self.config['io_count']}. Must be > 0")

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

        # Set CHI_WITH_RUNTIME environment variable based on configuration
        if self.config['init_runtime']:
            self.setenv('CHI_WITH_RUNTIME', '1')
            self.log("Runtime initialization enabled (CHI_WITH_RUNTIME=1)")
        else:
            self.setenv('CHI_WITH_RUNTIME', '0')
            self.log("Runtime initialization disabled (CHI_WITH_RUNTIME=0)")

        self.log("CTE benchmark configuration completed successfully")

    def start(self):
        """
        Run the CTE benchmark application.

        This method executes the benchmark with MPI support and configured parameters.
        """
        self.log(f"Starting CTE benchmark: {self.config['test_case']}")

        # Build command line arguments
        # Usage: wrp_cte_bench <test_case> <num_threads> <depth> <io_size> <io_count>
        cmd = [
            self.benchmark_executable,
            self.config['test_case'],
            str(self.config['num_threads']),
            str(self.config['depth']),
            str(self.config['io_size']),
            str(self.config['io_count'])
        ]

        # Execute with MPI if nprocs > 1
        if self.config['nprocs'] > 1:
            self.log(f"Running benchmark with MPI: {self.config['nprocs']} processes, {self.config['ppn']} per node")

            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=self.config['nprocs'],
                ppn=self.config['ppn']
            )
        else:
            self.log("Running benchmark without MPI (single process)")
            from jarvis_cd.shell import LocalExecInfo
            exec_info = LocalExecInfo(env=self.mod_env)

        # Execute the benchmark
        cmd_str = ' '.join(cmd)

        if self.output_file:
            # Redirect output to file
            cmd_str += f' > {self.output_file} 2>&1'
            self.log(f"Executing: {cmd_str}")
            Exec(cmd_str, exec_info).run()
            self.log(f"Benchmark completed. Results saved to: {self.output_file}")
        else:
            # Print to stdout
            self.log(f"Executing: {cmd_str}")
            Exec(cmd_str, exec_info).run()
            self.log("Benchmark completed")

    def stop(self):
        """
        Stop the benchmark application.

        Since this is an application that runs to completion, this method
        is typically not needed but is provided for consistency.
        """
        self.log("WrpCteBench is an application - it runs to completion")
        return True

    def clean(self):
        """
        Clean up benchmark output files.
        """
        self.log("Cleaning up CTE benchmark output files...")

        # Clean output file if it exists
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
                self.log(f"Removed benchmark output file: {self.output_file}")
            except Exception as e:
                self.log(f"Error removing output file: {e}")

        self.log("CTE benchmark cleanup completed")
