"""
GPU Runtime Benchmark Package

Benchmarks GPU task submission latency using the bench_gpu_runtime binary.
bench_gpu_runtime is self-contained: it starts its own Clio runtime
internally, so no clio_runtime package is needed in the pipeline.

Measures round-trip task latency from GPU client kernels to the GPU work
orchestrator.  Configurable block/thread counts and batch sizes allow
sweeping over parallelism levels.

Parameters:
- rt_blocks:      GPU runtime orchestrator block count
- rt_threads:     GPU runtime orchestrator threads per block
- client_blocks:  GPU client kernel blocks (one thread per block)
- batch_size:     Tasks submitted per batch per GPU thread
- total_tasks:    Total tasks per GPU thread
- output_dir:     Directory for benchmark result files

Assumes bench_gpu_runtime is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os


class ClioGpuBench(Application):
    """
    GPU Runtime Latency Benchmark

    Runs the self-contained bench_gpu_runtime binary to measure GPU task
    submission round-trip latency against the Clio GPU work orchestrator.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        """Define configuration options for the GPU benchmark"""
        return [
            {
                'name': 'rt_blocks',
                'msg': 'GPU runtime orchestrator block count',
                'type': int,
                'default': 1
            },
            {
                'name': 'rt_threads',
                'msg': 'GPU runtime orchestrator threads per block',
                'type': int,
                'default': 32
            },
            {
                'name': 'client_blocks',
                'msg': 'GPU client kernel block count (one thread per block)',
                'type': int,
                'default': 1
            },
            {
                'name': 'batch_size',
                'msg': 'Tasks submitted per batch per GPU thread',
                'type': int,
                'default': 1
            },
            {
                'name': 'total_tasks',
                'msg': 'Total tasks per GPU thread',
                'type': int,
                'default': 100
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/clio_gpu_bench'
            }
        ]

    def _configure(self, **kwargs):
        """Configure the GPU benchmark"""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.setenv('CLIO_WITH_RUNTIME', '0')

        self.log("GPU runtime benchmark configured")
        self.log(f"  RT blocks:     {self.config['rt_blocks']}")
        self.log(f"  RT threads:    {self.config['rt_threads']}")
        self.log(f"  Client blocks: {self.config['client_blocks']}")
        self.log(f"  Batch size:    {self.config['batch_size']}")
        self.log(f"  Total tasks:   {self.config['total_tasks']}")
        self.log(f"  Output dir:    {self.config['output_dir']}")

    def start(self):
        """Run the GPU benchmark"""
        Which('bench_gpu_runtime', LocalExecInfo(env=self.mod_env)).run()

        output_file = os.path.join(self.config['output_dir'],
                                   'gpu_bench_results.txt')
        os.makedirs(self.config['output_dir'], exist_ok=True)

        self.log("Starting GPU runtime latency benchmark")

        cmd = ' '.join([
            'bench_gpu_runtime',
            '--test-case latency',
            f'--rt-blocks {self.config["rt_blocks"]}',
            f'--rt-threads {self.config["rt_threads"]}',
            f'--client-blocks {self.config["client_blocks"]}',
            f'--batch-size {self.config["batch_size"]}',
            f'--total-tasks {self.config["total_tasks"]}',
        ])

        Exec(f'{cmd} 2>&1 | tee {output_file}',
             LocalExecInfo(env=self.mod_env)).run()

        self.log(f"Benchmark completed — results saved to {output_file}")

    def stop(self):
        """Stop method — benchmark completes automatically"""
        pass

    def clean(self):
        """Clean benchmark output"""
        self.log("Cleaning GPU benchmark data")

        output_file = os.path.join(self.config['output_dir'],
                                   'gpu_bench_results.txt')
        if os.path.exists(output_file):
            os.remove(output_file)

        try:
            os.rmdir(self.config['output_dir'])
        except OSError:
            pass

        self.log("Cleanup completed")
