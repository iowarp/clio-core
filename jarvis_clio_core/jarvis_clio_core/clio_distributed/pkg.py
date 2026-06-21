"""
IOWarp Distributed Unit Test Package

Runs distributed bdev unit tests with different PoolQuery strategies:
- DirectHash: Uses loop iterator as hash for distributed allocation
- Range: Uses range query across multiple nodes
- Broadcast: Uses broadcast query to all nodes

Requires clio_runtime to be running.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os


class ClioDistributed(Application):
    """
    IOWarp Distributed Unit Test

    Executes distributed bdev tests that verify proper operation of
    PoolQuery strategies (DirectHash, Range, Broadcast) across multiple nodes.

    Parameters:
    - num_nodes: Number of nodes in the distributed system
    - test_case: Which test case to run (direct, range, broadcast, all)
    - output_dir: Directory for test results
    - verbose: Enable verbose test output

    Assumes clio_run_distributed_bdev_tests is installed and available in PATH.
    Requires clio_runtime to be running.
    """

    def _init(self):
        """Initialize test package variables"""
        self.output_file = None

    def _configure_menu(self):
        """Define configuration options for distributed test"""
        return [
            {
                'name': 'num_nodes',
                'msg': 'Number of distributed nodes',
                'type': int,
                'default': 4
            },
            {
                'name': 'test_case',
                'msg': 'Test case to run (direct, range, broadcast, all)',
                'type': str,
                'choices': ['direct', 'range', 'broadcast', 'all'],
                'default': 'all'
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for test results',
                'type': str,
                'default': '/tmp/clio_distributed'
            },
            {
                'name': 'verbose',
                'msg': 'Enable verbose test output',
                'type': bool,
                'default': True
            }
        ]

    def _configure(self, **kwargs):
        """Configure the distributed test"""
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.output_file = os.path.join(self.config['output_dir'], 'distributed_test_results.txt')

        # Set test environment variables
        self.setenv('CLIO_DISTRIBUTED_OUTPUT_DIR', self.config['output_dir'])

        self.log("IOWarp distributed test configured")
        self.log(f"  Num nodes: {self.config['num_nodes']}")
        self.log(f"  Test case: {self.config['test_case']}")
        self.log(f"  Output directory: {self.config['output_dir']}")

    def start(self):
        """Run the distributed test"""
        # Verify test executable is available
        Which('clio_run_distributed_bdev_tests', LocalExecInfo(env=self.mod_env)).run()

        self.log(f"Starting distributed bdev test: {self.config['test_case']}")

        # Determine which test cases to run
        test_cases = []
        if self.config['test_case'] == 'all':
            test_cases = ['direct', 'range', 'broadcast']
        else:
            test_cases = [self.config['test_case']]

        # Run each test case
        for test_case in test_cases:
            self.log(f"Running {test_case} test case...")

            # Build test command
            cmd_parts = [
                'clio_run_distributed_bdev_tests',
                f'--num-nodes {self.config["num_nodes"]}',
                f'--test-case {test_case}'
            ]

            if self.config['verbose']:
                cmd_parts.append('--verbose')

            cmd = ' '.join(cmd_parts)

            # Create test-specific output file
            test_output_file = os.path.join(
                self.config['output_dir'],
                f'distributed_test_{test_case}.txt'
            )

            # Redirect output to file and console
            cmd_with_redirect = f'{cmd} 2>&1 | tee {test_output_file}'

            # Execute test
            self.log(f"Executing: {cmd}")
            Exec(cmd_with_redirect, LocalExecInfo(env=self.mod_env)).run()

            self.log(f"Test case '{test_case}' completed - results saved to {test_output_file}")

        self.log(f"All distributed tests completed")

    def stop(self):
        """Stop method - tests complete automatically"""
        pass

    def clean(self):
        """Clean test output"""
        self.log("Cleaning distributed test data")

        # Remove output files
        if self.output_file and os.path.exists(self.output_file):
            os.remove(self.output_file)

        # Remove test-specific output files
        for test_case in ['direct', 'range', 'broadcast']:
            test_output_file = os.path.join(
                self.config['output_dir'],
                f'distributed_test_{test_case}.txt'
            )
            if os.path.exists(test_output_file):
                os.remove(test_output_file)

        # Remove output directory if empty
        try:
            os.rmdir(self.config['output_dir'])
        except OSError:
            pass  # Directory not empty or doesn't exist

        self.log("Cleanup completed")
