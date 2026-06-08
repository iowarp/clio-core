from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, PsshExecInfo
import os


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
        ]
        exec_info = PsshExecInfo(
            env=self.mod_env,
            hostfile=self.hostfile,
            nprocs=self.config['nprocs'],
            ppn=self.config['ppn'],
        )
        cmd_str = ' '.join(cmd)
        if self.output_file:
            cmd_str += f' > {self.output_file} 2>&1'
        self.log(f"Executing: {cmd_str}")
        Exec(cmd_str, exec_info).run()
        self.log("SemanticSearch benchmark completed")

    def stop(self):
        return True

    def clean(self):
        if self.output_file and os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except Exception as e:
                self.log(f"Error removing output file: {e}")
        return True
