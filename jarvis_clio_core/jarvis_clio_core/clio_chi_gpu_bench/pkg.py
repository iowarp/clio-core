"""
Clio GPU Runtime Benchmark Package

Benchmarks GPU task submission latency and throughput using the
bench_gpu_runtime binary.  bench_gpu_runtime is self-contained: it starts
its own Clio runtime internally, so no clio_runtime package is needed.

Supported test cases:
  latency      -- GPU task round-trip latency (default)
  coroutine    -- GPU coroutine subtask latency
  alloc        -- GPU scratch allocator throughput
  serde        -- GPU serialization/deserialization throughput
  string_alloc -- GPU string allocation throughput
  putblob      -- GPU PutBlob via GPU->CPU path (ToLocalCpu)
  putblob_gpu  -- GPU PutBlob via GPU-local path (Local)

Parameters:
- test_case:      Benchmark mode
- rt_blocks:      GPU runtime orchestrator block count
- rt_threads:     GPU runtime orchestrator threads per block
- client_blocks:  GPU client kernel blocks
- client_threads: GPU client kernel threads per block
- batch_size:     Tasks per batch per GPU thread (latency test)
- total_tasks:    Total tasks per GPU thread
- subtasks:       Subtasks per coroutine task (coroutine test)
- io_size:        Total I/O size in bytes (putblob tests)
- output_dir:     Directory for benchmark result files

Assumes bench_gpu_runtime is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re
import signal
import subprocess
import time


class ClioChiGpuBench(Application):
    """
    Clio GPU Runtime Benchmark

    Runs bench_gpu_runtime to measure GPU task submission latency and
    throughput against the Clio GPU work orchestrator.
    The benchmark is self-contained and starts its own Clio runtime.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark test case',
                'type': str,
                'choices': [
                    'latency', 'coroutine', 'alloc', 'alloc_serde',
                    'serde', 'string_alloc', 'putblob', 'putblob_gpu'
                ],
                'default': 'latency',
            },
            {
                'name': 'rt_blocks',
                'msg': 'GPU runtime orchestrator block count',
                'type': int,
                'default': 1,
            },
            {
                'name': 'rt_threads',
                'msg': 'GPU runtime orchestrator threads per block',
                'type': int,
                'default': 32,
            },
            {
                'name': 'client_blocks',
                'msg': 'GPU client kernel blocks',
                'type': int,
                'default': 1,
            },
            {
                'name': 'client_threads',
                'msg': 'GPU client kernel threads per block',
                'type': int,
                'default': 32,
            },
            {
                'name': 'batch_size',
                'msg': 'Tasks per batch per GPU thread (latency test)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'total_tasks',
                'msg': 'Total tasks per GPU thread',
                'type': int,
                'default': 100,
            },
            {
                'name': 'subtasks',
                'msg': 'Subtasks per coroutine task (coroutine test)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'io_size',
                'msg': 'Total I/O size in bytes (putblob tests)',
                'type': int,
                'default': 67108864,
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/clio_chi_gpu_bench',
            },
        ]

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)

        warps = (self.config['client_blocks'] *
                 self.config['client_threads']) // 32
        self.log("Clio GPU runtime benchmark configured")
        self.log(f"  Test case:      {self.config['test_case']}")
        self.log(f"  RT config:      {self.config['rt_blocks']}b x "
                 f"{self.config['rt_threads']}t")
        self.log(f"  Client config:  {self.config['client_blocks']}b x "
                 f"{self.config['client_threads']}t ({warps} warps)")
        self.log(f"  Batch size:     {self.config['batch_size']}")
        self.log(f"  Total tasks:    {self.config['total_tasks']}")

    def _kill_stale_processes(self):
        """Kill any leftover bench_gpu_runtime or clio_run processes and
        free port 9413 so the next run can start cleanly."""
        for proc_name in ['bench_gpu_runtime', 'clio_run']:
            try:
                subprocess.run(
                    ['pkill', '-9', '-f', proc_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except Exception:
                pass
        try:
            result = subprocess.run(
                ['lsof', '-ti', ':9413'],
                capture_output=True, text=True)
            for pid in result.stdout.strip().split('\n'):
                if pid.strip():
                    try:
                        os.kill(int(pid.strip()), signal.SIGKILL)
                    except (ProcessLookupError, ValueError):
                        pass
        except Exception:
            pass
        time.sleep(2)

    def start(self):
        self._kill_stale_processes()

        Which('bench_gpu_runtime', LocalExecInfo(env=self.mod_env)).run()

        output_file = os.path.join(
            self.config['output_dir'],
            f"chi_gpu_{self.config['test_case']}.txt")

        cmd_parts = [
            'bench_gpu_runtime',
            f'--test-case {self.config["test_case"]}',
            f'--rt-blocks {self.config["rt_blocks"]}',
            f'--rt-threads {self.config["rt_threads"]}',
            f'--client-blocks {self.config["client_blocks"]}',
            f'--client-threads {self.config["client_threads"]}',
            f'--batch-size {self.config["batch_size"]}',
            f'--total-tasks {self.config["total_tasks"]}',
        ]
        if self.config['test_case'] == 'coroutine':
            cmd_parts.append(f'--subtasks {self.config["subtasks"]}')
        if self.config['test_case'] in ('putblob', 'putblob_gpu'):
            cmd_parts.append(f'--io-size {self.config["io_size"]}')

        cmd = ' '.join(cmd_parts)
        self.log(f"Running: {cmd}")
        self.exec = Exec(
            f'{cmd} 2>&1 | tee {output_file}',
            LocalExecInfo(env=self.mod_env,
                          collect_output=True)).run()
        self.log(f"Results saved to {output_file}")

    def _get_stat(self, stat_dict):
        output = self.exec.stdout['localhost']
        elapsed = re.search(r'Elapsed time:\s+([0-9.]+)\s+ms', output)
        if elapsed:
            stat_dict[f'{self.pkg_id}.elapsed_ms'] = float(elapsed.group(1))
        throughput = re.search(r'Throughput:\s+([0-9.]+)\s+tasks/sec', output)
        if throughput:
            stat_dict[f'{self.pkg_id}.throughput'] = float(throughput.group(1))
        latency = re.search(r'Avg latency:\s+([0-9.]+)\s+us', output)
        if latency:
            stat_dict[f'{self.pkg_id}.latency_us'] = float(latency.group(1))
        bandwidth = re.search(r'Bandwidth:\s+([0-9.]+)\s+GB/s', output)
        if bandwidth:
            stat_dict[f'{self.pkg_id}.bandwidth_gbps'] = float(
                bandwidth.group(1))
        stat_dict[f'{self.pkg_id}.test_case'] = self.config['test_case']
        stat_dict[f'{self.pkg_id}.warps'] = (
            self.config['client_blocks'] * self.config['client_threads']) // 32

    def _plot(self, results_csv, output_dir):
        try:
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("Skipping plots: pandas or matplotlib not installed")
            return

        df = pd.read_csv(results_csv)

        throughput_col = None
        latency_col = None
        warps_col = None
        for col in df.columns:
            if col.endswith('.throughput'):
                throughput_col = col
            elif col.endswith('.latency_us'):
                latency_col = col
            elif col.endswith('.warps'):
                warps_col = col

        if not throughput_col or not warps_col:
            return

        # Throughput vs warps
        if len(df[warps_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = df.groupby(warps_col)[throughput_col].mean()
            grouped.plot(kind='bar', ax=ax)
            ax.set_xlabel('Warps')
            ax.set_ylabel('Throughput (tasks/sec)')
            ax.set_title('GPU Runtime Throughput vs Warp Count')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     'throughput_vs_warps.png'), dpi=150)
            plt.close(fig)

        # Latency vs warps
        if latency_col and len(df[warps_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(8, 5))
            grouped = df.groupby(warps_col)[latency_col].mean()
            grouped.plot(kind='bar', ax=ax)
            ax.set_xlabel('Warps')
            ax.set_ylabel('Avg Latency (us/task/warp)')
            ax.set_title('GPU Runtime Latency vs Warp Count')
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     'latency_vs_warps.png'), dpi=150)
            plt.close(fig)

        self.log(f"Plots saved to {output_dir}")

    def stop(self):
        self._kill_stale_processes()

    def clean(self):
        output_dir = self.config['output_dir']
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                path = os.path.join(output_dir, f)
                if os.path.isfile(path) and f.startswith('chi_gpu_'):
                    os.remove(path)
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
