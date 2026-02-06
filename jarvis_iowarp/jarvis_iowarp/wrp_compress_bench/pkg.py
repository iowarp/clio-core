from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo
import os


class WrpCompressBench(Application):
    """
    CTE Compression Benchmark Application

    This application runs synthetic workloads with configurable data patterns
    to benchmark compression performance in the Content Transfer Engine (CTE).

    The benchmark generates data with various patterns that simulate scientific
    simulation output (e.g., Gray-Scott reaction-diffusion patterns) and measures:
    - Compute time: Time spent in simulated computation
    - I/O time: Time spent in data transfer operations
    - Compression time: Time spent compressing data
    - Compression ratio: Achieved compression ratio

    Data patterns include:
    - grayscott: Bimodal distribution mimicking Gray-Scott simulation (~70% background, ~20% spots)
    - gaussian: Normal distribution
    - uniform: Uniform random distribution
    - constant: All same values (maximally compressible)
    - gradient: Linear gradient
    - sinusoidal: Sinusoidal wave pattern
    - repeating: Repeating pattern
    - bimodal: Generic bimodal distribution
    - exponential: Exponential distribution

    Patterns can be mixed with percentages, e.g., "grayscott:70,gaussian:20,uniform:10"
    """

    def _init(self):
        """
        Initialize the WrpCompressBench application.

        This method is called during application initialization.
        """
        self.benchmark_executable = 'synthetic_workload_exec'
        self.output_file = None

    def _configure_menu(self):
        """
        Configure the application menu.

        Returns:
            List[Dict]: Configuration menu options for the benchmark.
        """
        return [
            {
                'name': 'io_size',
                'msg': 'I/O size per rank',
                'type': str,
                'default': '1MB',
                'help': 'Total I/O data size per MPI rank (e.g., "1MB", "128KB", "4GB")'
            },
            {
                'name': 'transfer_size',
                'msg': 'Transfer chunk size',
                'type': str,
                'default': '64KB',
                'help': 'Size of each transfer chunk (e.g., "64KB", "1MB")'
            },
            {
                'name': 'compute_time',
                'msg': 'Compute time per iteration (ms)',
                'type': int,
                'default': 100,
                'help': 'Simulated compute time per iteration in milliseconds'
            },
            {
                'name': 'iterations',
                'msg': 'Number of iterations',
                'type': int,
                'default': 10,
                'help': 'Number of compute+I/O iterations to perform'
            },
            {
                'name': 'pattern',
                'msg': 'Data pattern specification',
                'type': str,
                'default': 'grayscott:100',
                'help': 'Data pattern(s) with percentages. Format: <pattern>:<pct>,... '
                        'Available: uniform, gaussian, constant, gradient, sinusoidal, '
                        'repeating, grayscott, bimodal, exponential. '
                        'Example: "grayscott:70,gaussian:20,uniform:10"'
            },
            {
                'name': 'compress',
                'msg': 'Compression option',
                'type': str,
                'choices': ['none', 'dynamic', 'zstd', 'lz4', 'brotli', 'bzip2',
                           'blosc2', 'fpzip', 'lzma', 'snappy', 'sz3', 'zfp', 'zlib'],
                'default': 'dynamic',
                'help': 'Compression method: "none" (no compression), "dynamic" (AI-selected), '
                        'or a specific library name'
            },
            {
                'name': 'nprocs',
                'msg': 'Number of MPI processes',
                'type': int,
                'default': 1,
                'help': 'Number of MPI processes to use'
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
                'name': 'trace',
                'msg': 'Enable compression tracing',
                'type': bool,
                'default': False,
                'help': 'Enable detailed compression tracing for performance analysis'
            }
        ]

    def _configure(self, **kwargs):
        """
        Configure the compression benchmark application with provided keyword arguments.

        This method validates configuration and sets up environment variables.

        Args:
            **kwargs: Configuration arguments from _configure_menu.
        """
        self.log("Configuring CTE compression benchmark application...")

        # Validate compute_time
        if self.config['compute_time'] < 0:
            raise ValueError(f"Invalid compute_time: {self.config['compute_time']}. Must be >= 0")

        # Validate iterations
        if self.config['iterations'] <= 0:
            raise ValueError(f"Invalid iterations: {self.config['iterations']}. Must be > 0")

        # Validate nprocs
        if self.config['nprocs'] <= 0:
            raise ValueError(f"Invalid nprocs: {self.config['nprocs']}. Must be > 0")

        # Validate ppn
        if self.config['ppn'] <= 0:
            raise ValueError(f"Invalid ppn: {self.config['ppn']}. Must be > 0")

        # Validate pattern format
        pattern = self.config['pattern']
        valid_patterns = ['uniform', 'gaussian', 'constant', 'gradient', 'sinusoidal',
                         'repeating', 'grayscott', 'bimodal', 'exponential']
        for part in pattern.split(','):
            if ':' in part:
                pname = part.split(':')[0].strip()
                if pname not in valid_patterns:
                    self.log(f"Warning: Unknown pattern '{pname}'. Valid patterns: {valid_patterns}")

        # Set output file if specified
        if self.config['output_file']:
            self.output_file = os.path.join(self.shared_dir, self.config['output_file'])
            self.log(f"Benchmark results will be saved to: {self.output_file}")
        else:
            self.output_file = None
            self.log("Benchmark results will be printed to stdout")

        # Set trace environment variable
        if self.config['trace']:
            self.setenv('WRP_CTE_COMPRESS_TRACE', 'on')
            self.log("Compression tracing enabled")

        self.log(f"Configuration: io_size={self.config['io_size']}, "
                f"transfer_size={self.config['transfer_size']}, "
                f"compute_time={self.config['compute_time']}ms, "
                f"iterations={self.config['iterations']}, "
                f"pattern={self.config['pattern']}, "
                f"compress={self.config['compress']}")
        self.log("CTE compression benchmark configuration completed successfully")

    def start(self):
        """
        Run the CTE compression benchmark application.

        This method executes the benchmark with MPI support and configured parameters.
        """
        self.log(f"Starting CTE compression benchmark with pattern: {self.config['pattern']}")

        # Build command line arguments
        cmd = [
            self.benchmark_executable,
            f"--io-size {self.config['io_size']}",
            f"--transfer-size {self.config['transfer_size']}",
            f"--compute-time {self.config['compute_time']}",
            f"--iterations {self.config['iterations']}",
            f"--pattern {self.config['pattern']}",
            f"--compress {self.config['compress']}"
        ]

        # Add trace flag if enabled
        if self.config['trace']:
            cmd.append("--trace")

        # Execute with MPI
        if self.config['nprocs'] > 1:
            self.log(f"Running benchmark with MPI: {self.config['nprocs']} processes, "
                    f"{self.config['ppn']} per node")

            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=self.config['nprocs'],
                ppn=self.config['ppn']
            )
        else:
            self.log("Running benchmark with single MPI process")
            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=1,
                ppn=1
            )

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
        self.log("WrpCompressBench is an application - it runs to completion")
        return True

    def clean(self):
        """
        Clean up benchmark output files.
        """
        self.log("Cleaning up CTE compression benchmark output files...")

        # Clean output file if it exists
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
                self.log(f"Removed benchmark output file: {self.output_file}")
            except Exception as e:
                self.log(f"Error removing output file: {e}")

        self.log("CTE compression benchmark cleanup completed")
