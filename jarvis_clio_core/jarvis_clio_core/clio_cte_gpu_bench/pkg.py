"""
CTE GPU Benchmark Package

Benchmarks GPU-initiated I/O operations through the Content Transfer
Engine (CTE) and compares against BaM page cache and baselines.
clio_cte_gpu_bench is self-contained: it starts its own Clio runtime
internally, so no clio_runtime package is needed.

Supported test cases:
  client_overhead      -- Measure AsyncPutBlob GPU-side submit cost (Local)
  client_overhead_cpu  -- Same but via ToLocalCpu routing
  synthetic            -- Synthetic I/O (modes: hbm, direct, bam, cte)
  pagerank             -- PageRank workload
  gnn                  -- GNN feature gather workload
  gray_scott           -- Gray-Scott stencil simulation
  llm_kvcache          -- LLM KV cache offloading

Parameters:
- test_case:      Benchmark workload
- workload_mode:  I/O mode (hbm, direct, bam, cte)
- routing:        CTE task routing (local, to_cpu)
- rt_blocks:      GPU runtime orchestrator block count
- rt_threads:     GPU runtime orchestrator threads per block
- client_blocks:  GPU client kernel blocks
- client_threads: GPU client kernel threads per block
- io_size:        Per-warp I/O size (supports k/m/g suffixes)
- iterations:     Number of iterations per warp
- targets:        CTE storage targets as [type, size] pairs
- timeout:        PollDone timeout in seconds

Assumes clio_cte_gpu_bench is installed and available in PATH.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Which
import os
import re
import signal
import subprocess
import time


class ClioCteGpuBench(Application):
    """
    CTE GPU Bandwidth Benchmark

    Runs clio_cte_gpu_bench to measure GPU-initiated CTE throughput
    and compare against BaM page cache and baselines.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {
                'name': 'test_case',
                'msg': 'Benchmark workload',
                'type': str,
                'choices': [
                    'client_overhead', 'client_overhead_cpu',
                    'synthetic', 'pagerank', 'gnn',
                    'gray_scott', 'llm_kvcache',
                ],
                'default': 'synthetic',
            },
            {
                'name': 'workload_mode',
                'msg': 'I/O mode (hbm, direct, bam, cte)',
                'type': str,
                'choices': ['hbm', 'direct', 'bam', 'cte'],
                'default': 'hbm',
            },
            {
                'name': 'routing',
                'msg': 'CTE task routing (local or to_cpu)',
                'type': str,
                'choices': ['local', 'to_cpu'],
                'default': 'local',
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
                'default': 256,
            },
            {
                'name': 'io_size',
                'msg': 'Per-warp I/O size (supports k/m/g suffixes)',
                'type': str,
                'default': '128k',
            },
            {
                'name': 'iterations',
                'msg': 'Iterations per warp',
                'type': int,
                'default': 16,
            },
            {
                'name': 'timeout',
                'msg': 'PollDone timeout in seconds',
                'type': int,
                'default': 60,
            },
            {
                'name': 'output_dir',
                'msg': 'Output directory for benchmark results',
                'type': str,
                'default': '/tmp/clio_cte_gpu_bench',
            },
            {
                'name': 'targets',
                'msg': 'CTE storage targets as [type, size] pairs',
                'type': list,
                'default': [],
                'help': 'Example: [["hbm", "256m"], ["pinned", "256m"]]. '
                        'Types: hbm, pinned, ram. '
                        'If empty, defaults to single hbm:256m target.',
            },
        ]

    def _configure(self, **kwargs):
        os.makedirs(self.config['output_dir'], exist_ok=True)

        warps = (self.config['client_blocks'] *
                 self.config['client_threads']) // 32
        self.log("CTE GPU benchmark configured")
        self.log(f"  Test case:      {self.config['test_case']}")
        self.log(f"  Workload mode:  {self.config['workload_mode']}")
        self.log(f"  Routing:        {self.config['routing']}")
        self.log(f"  RT config:      {self.config['rt_blocks']}b x "
                 f"{self.config['rt_threads']}t")
        self.log(f"  Client config:  {self.config['client_blocks']}b x "
                 f"{self.config['client_threads']}t ({warps} warps)")
        self.log(f"  IO/warp:        {self.config['io_size']}")
        self.log(f"  Iterations:     {self.config['iterations']}")
        self.log(f"  Timeout:        {self.config['timeout']}s")
        targets = self.config.get('targets', [])
        if targets:
            self.log(f"  Targets:        {targets}")
        else:
            self.log(f"  Targets:        (default: hbm:256m)")

    def _kill_stale_processes(self):
        """Kill any leftover clio_cte_gpu_bench or clio_run processes and
        free port 9413 so the next run can start cleanly."""
        for proc_name in ['clio_cte_gpu_bench', 'clio_run']:
            try:
                subprocess.run(
                    ['pkill', '-9', '-f', proc_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except Exception:
                pass
        # Also kill anything holding port 9413
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
        # Kill stale processes from previous runs before starting
        self._kill_stale_processes()

        Which('clio_cte_gpu_bench', LocalExecInfo(env=self.mod_env)).run()

        tc = self.config['test_case']
        output_file = os.path.join(
            self.config['output_dir'],
            f"cte_gpu_{tc}.txt")

        # Build command with all parameters
        cmd_parts = [
            'clio_cte_gpu_bench',
            f'--test-case {tc}',
            f'--rt-blocks {self.config["rt_blocks"]}',
            f'--rt-threads {self.config["rt_threads"]}',
            f'--client-blocks {self.config["client_blocks"]}',
            f'--client-threads {self.config["client_threads"]}',
            f'--io-size {self.config["io_size"]}',
            f'--iterations {self.config["iterations"]}',
            f'--timeout {self.config["timeout"]}',
        ]

        # Add workload-mode and routing for workload test cases
        if tc in ('synthetic', 'pagerank', 'gnn', 'gray_scott', 'llm_kvcache'):
            cmd_parts.append(
                f'--workload-mode {self.config["workload_mode"]}')
            cmd_parts.append(f'--routing {self.config["routing"]}')

        # Add storage targets
        for target in self.config.get('targets', []):
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                cmd_parts.append(f'--target {target[0]}:{target[1]}')

        cmd = ' '.join(cmd_parts)

        self.log(f"Running: {cmd}")
        self.exec = Exec(f'{cmd} 2>&1 | tee {output_file}',
             LocalExecInfo(env=self.mod_env,
                           collect_output=True)).run()
        self.log(f"Results saved to {output_file}")

    def _get_stat(self, stat_dict):
        output = self.exec.stdout['localhost']
        pid = self.pkg_id

        # --- Capture all package config variables ---
        stat_dict[f'{pid}.test_case'] = self.config['test_case']
        stat_dict[f'{pid}.workload_mode'] = self.config['workload_mode']
        stat_dict[f'{pid}.routing'] = self.config['routing']
        stat_dict[f'{pid}.rt_blocks'] = self.config['rt_blocks']
        stat_dict[f'{pid}.rt_threads'] = self.config['rt_threads']
        stat_dict[f'{pid}.client_blocks'] = self.config['client_blocks']
        stat_dict[f'{pid}.client_threads'] = self.config['client_threads']
        stat_dict[f'{pid}.io_size'] = self.config['io_size']
        stat_dict[f'{pid}.iterations'] = self.config['iterations']
        stat_dict[f'{pid}.targets'] = str(self.config.get('targets', []))
        stat_dict[f'{pid}.warps'] = (
            self.config['client_blocks'] * self.config['client_threads']) // 32

        # --- Parse benchmark output ---

        # Elapsed time
        elapsed = re.search(
            r'(?:Wall )?[Ee]lapsed:\s+([0-9.]+)\s*ms', output)
        if elapsed:
            stat_dict[f'{pid}.elapsed_ms'] = float(elapsed.group(1))

        # Bandwidth
        bandwidth = re.search(
            r'Bandwidth:\s+([0-9.]+)\s+GB/s', output)
        if bandwidth:
            stat_dict[f'{pid}.bandwidth_gbps'] = float(bandwidth.group(1))

        # Client overhead (submit cost)
        submit_cost = re.search(
            r'Avg submit cost:\s+([0-9.]+)\s+us/call', output)
        if submit_cost:
            stat_dict[f'{pid}.submit_us'] = float(submit_cost.group(1))

        # Workload-specific metric (e.g., putgets/sec, edges/sec, tokens/sec)
        metric_match = re.search(
            r'^(\S+/sec\S*)\s+([0-9.eE+-]+)', output, re.MULTILINE)
        if metric_match:
            stat_dict[f'{pid}.metric_name'] = metric_match.group(1)
            stat_dict[f'{pid}.metric_value'] = float(metric_match.group(2))

    @staticmethod
    def _io_sort_key(s):
        """Sort I/O size strings numerically (4k < 64k < 1m < 16m)."""
        s = str(s).strip().lower()
        m = re.match(r'([0-9.]+)\s*([kmgt]?)', s)
        if not m:
            return 0
        val = float(m.group(1))
        suffix = m.group(2)
        mult = {'': 1, 'k': 1024, 'm': 1024**2, 'g': 1024**3, 't': 1024**4}
        return val * mult.get(suffix, 1)

    def _plot(self, results_yaml, output_dir):
        try:
            import yaml as _yaml
            import pandas as pd
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.log("Skipping plots: pandas, pyyaml, or matplotlib not installed")
            return

        # Load results from YAML
        with open(results_yaml, 'r') as f:
            data = _yaml.safe_load(f)

        if not data or 'results' not in data:
            return

        # Flatten each result into a single row: variables + stats
        rows = []
        for result in data['results']:
            row = {}
            row.update(result.get('variables', {}))
            row.update(result.get('stats', {}))
            row['status'] = result.get('status', 'unknown')
            row['runtime'] = result.get('runtime', None)
            rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)

        # Find columns for this package
        pid = self.pkg_id
        bw_col = f'{pid}.bandwidth_gbps'
        tc_col = f'{pid}.test_case'
        mode_col = f'{pid}.workload_mode'
        routing_col = f'{pid}.routing'
        io_col = f'{pid}.io_size'

        if bw_col not in df.columns:
            return

        # Build a combined mode label: "cte_local", "cte_to_cpu", "bam", etc.
        if mode_col in df.columns and routing_col in df.columns:
            df['_mode_label'] = df.apply(
                lambda r: f"{r[mode_col]}_{r[routing_col]}"
                if r[mode_col] == 'cte' else str(r[mode_col]),
                axis=1)
        elif mode_col in df.columns:
            df['_mode_label'] = df[mode_col]
        else:
            df['_mode_label'] = 'unknown'

        # --- Plot 1: Bandwidth vs I/O size, grouped by mode ---
        if io_col in df.columns and len(df[io_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = df.groupby([io_col, '_mode_label'])[bw_col].mean().unstack()
            pivot = pivot.reindex(
                sorted(pivot.index, key=self._io_sort_key))
            pivot.plot(kind='bar', ax=ax)
            ax.set_xlabel('I/O Size per Warp')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title('Bandwidth by I/O Size and Mode')
            ax.legend(title='Mode', bbox_to_anchor=(1.02, 1),
                      loc='upper left', fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     'bandwidth_vs_iosize_by_mode.png'),
                        dpi=150)
            plt.close(fig)

        # --- Plot 2: Bandwidth vs workload, grouped by mode ---
        if tc_col in df.columns and len(df[tc_col].dropna().unique()) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            pivot = df.groupby([tc_col, '_mode_label'])[bw_col].mean().unstack()
            pivot.plot(kind='bar', ax=ax)
            ax.set_xlabel('Workload')
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title('Bandwidth by Workload and Mode')
            ax.legend(title='Mode', bbox_to_anchor=(1.02, 1),
                      loc='upper left', fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir,
                                     'bandwidth_vs_workload_by_mode.png'),
                        dpi=150)
            plt.close(fig)

        self.log(f"Plots saved to {output_dir}")

    def stop(self):
        self._kill_stale_processes()

    def clean(self):
        output_dir = self.config['output_dir']
        if os.path.isdir(output_dir):
            for f in os.listdir(output_dir):
                path = os.path.join(output_dir, f)
                if os.path.isfile(path) and f.startswith('cte_gpu_'):
                    os.remove(path)
            try:
                os.rmdir(output_dir)
            except OSError:
                pass
