from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo
import os
import re


class ClioCteSemanticBench(Application):
    """
    CTE SemanticSearch benchmark.

    Writes a configurable number of configurable-size blobs (all containing
    the same searched keyword) under a single tag, then issues ONE broadcast
    SemanticSearch returning a configurable number of top-k results. Exercises
    the broadcast -> SemanticSearchTask::Aggregate (merge-by-score) path.

    Runs the `clio_cte_semantic_bench` binary as a CTE client; assumes a
    Chimaera runtime + CTE pool are already up (init_runtime=False).
    """

    def _init(self):
        self.benchmark_executable = 'clio_cte_semantic_bench'
        self.output_file = None

    def _configure_menu(self):
        return [
            {
                'name': 'blobs',
                'msg': 'Number of blobs to write under the tag',
                'type': int,
                'default': 1000,
            },
            {
                'name': 'blob_size',
                'msg': 'Size of each blob in bytes',
                'type': int,
                'default': 4096,
            },
            {
                'name': 'results',
                'msg': 'Number of top-k results to return (0 = all)',
                'type': int,
                'default': 10,
            },
            {
                'name': 'keyword',
                'msg': 'Keyword stored in / searched for in every blob',
                'type': str,
                'default': 'needle',
            },
            {
                'name': 'query_iters',
                'msg': 'Repeat the search N times for a stable latency stat',
                'type': int,
                'default': 5,
            },
            {
                'name': 'nprocs',
                'msg': 'Number of client processes',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'Client processes per node',
                'type': int,
                'default': 1,
            },
            {
                'name': 'output_file',
                'msg': 'File to save benchmark output (empty = stdout)',
                'type': str,
                'default': '',
            },
            {
                'name': 'init_runtime',
                'msg': 'Initialize Chimaera runtime (else assume already running)',
                'type': bool,
                'default': False,
            },
        ]

    def _configure(self, **kwargs):
        self.log("Configuring CTE SemanticSearch benchmark...")
        if self.config['blobs'] <= 0:
            raise ValueError(f"Invalid blobs: {self.config['blobs']}. Must be > 0")
        if self.config['blob_size'] <= 0:
            raise ValueError(
                f"Invalid blob_size: {self.config['blob_size']}. Must be > 0")

        if self.config['output_file']:
            self.output_file = os.path.join(self.shared_dir,
                                            self.config['output_file'])
        else:
            self.output_file = None

        if self.config['init_runtime']:
            self.setenv('CLIO_WITH_RUNTIME', '1')
        else:
            self.setenv('CLIO_WITH_RUNTIME', '0')
        self.log("CTE SemanticSearch benchmark configuration completed")

    def start(self):
        cmd = [
            self.benchmark_executable,
            '--blobs', str(self.config['blobs']),
            '--size', str(self.config['blob_size']),
            '--results', str(self.config['results']),
            '--keyword', str(self.config['keyword']),
            '--query-iters', str(self.config['query_iters']),
        ]
        exec_info = PsshExecInfo(
            env=self.mod_env,
            hostfile=self.hostfile,
            nprocs=self.config['nprocs'],
            ppn=self.config['ppn'],
            collect_output=True,
        )
        cmd_str = ' '.join(cmd)
        self.log(f"Executing: {cmd_str}")
        # Run and keep the Exec so _get_stat can parse the [SEM_BENCH] line.
        self.exec = Exec(cmd_str, exec_info).run()
        out = '\n'.join(self.exec.stdout.values()) if self.exec.stdout else ''
        # Persist output to the SHARED dir. In pipeline-test Mode B, start()
        # runs inside a scheduler job (compute node) but _get_stat is invoked
        # on a FRESH instance on the head node — which has no self.exec — so
        # the stats must come from a shared file, not in-process stdout.
        try:
            with open(self._stat_file(), 'w') as f:
                f.write(out)
        except Exception as e:
            self.log(f"Could not write stat file: {e}")
        if self.output_file:
            with open(self.output_file, 'w') as f:
                f.write(out)
        self.log("SemanticSearch benchmark completed")

    def _stat_file(self):
        return os.path.join(self.shared_dir, 'sem_bench.out')

    def _get_stat(self, stat_dict):
        """Parse the benchmark's retrieval-performance summary.

        Pulls the [SEM_BENCH] key=value record (query latency to retrieve the
        top-k blobs, plus ingest stats) out of the benchmark output. Reads the
        shared file written by start() (works in Mode B); falls back to
        self.exec.stdout for in-process runs.
        """
        output = ''
        if os.path.exists(self._stat_file()):
            with open(self._stat_file()) as f:
                output = f.read()
        elif getattr(self, 'exec', None) is not None and self.exec.stdout:
            output = '\n'.join(self.exec.stdout.values())
        # Strip ANSI color codes the logger emits.
        output = re.sub(r'\x1b\[[0-9;]*m', '', output)
        line = ''
        for ln in output.splitlines():
            if '[SEM_BENCH]' in ln:
                line = ln
                break
        for key in ('query_avg_us', 'query_min_us', 'query_max_us',
                    'results', 'k', 'blobs', 'blob_size', 'query_iters',
                    'ingest_s', 'ingest_blobs_per_s'):
            m = re.search(rf'{key}=([0-9.]+)', line)
            if m:
                stat_dict[f'{self.pkg_id}.{key}'] = float(m.group(1))

    def stop(self):
        return True

    def clean(self):
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except Exception as e:
                self.log(f"Error removing output file: {e}")
        return True
