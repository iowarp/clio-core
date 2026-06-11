"""
Mofka Server Service — starts and stops the Bedrock/Mofka daemon,
creates a topic and partition so that downstream benchmark packages
can produce and consume events.

Ported from the archived ``jarvis_iowarp.mofka_server`` (branch
``claude/mofka-multinode``) into the ``jarvis_clio_core`` repo. The only
behavioural addition is publishing ``MOFKA_SERVER_HOST`` (the node that
actually runs bedrock) so ``mofka_bench`` can exclude it from the PSSH
client hostfile under the Slurm scheduler (which seeds the hostfile with
*all* allocated nodes, head included). See
``jarvis_clio_core/pipelines/mofka/architecture_decisions.md``.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, LocalExecInfo
from jarvis_cd.shell.process import Kill
import os
import json
import socket
import time
import shutil


class MofkaServer(Service):
    """Manages the Mofka/Bedrock daemon lifecycle."""

    def _init(self):
        self.bedrock_pid_file = None

    def _configure_menu(self):
        return [
            {
                'name': 'protocol',
                'msg': 'Mercury transport protocol',
                'type': str,
                'default': 'tcp',
                'choices': ['tcp', 'na+sm', 'ofi+tcp', 'ofi+verbs', 'ofi+cxi'],
            },
            {
                'name': 'workdir',
                'msg': 'Working directory for Mofka runtime files',
                'type': str,
                'default': '/tmp/mofka-bench',
            },
            {
                'name': 'topic',
                'msg': 'Mofka topic name to create',
                'type': str,
                'default': 'benchmark_topic',
            },
            {
                'name': 'partition_type',
                'msg': 'Partition storage type',
                'type': str,
                'default': 'memory',
                'choices': ['memory'],
            },
        ]

    def _configure(self, **kwargs):
        # Expand $USER / ~ in workdir so per-user paths like
        # /mnt/nvme/$USER/mofka-bench resolve before any FS op uses them.
        # Without this, the literal '$USER' would become a directory name
        # and shutil.rmtree() in start()/clean() would either no-op or
        # operate on the wrong path.
        workdir = os.path.expandvars(
            os.path.expanduser(self.config['workdir']))
        self.config['workdir'] = workdir
        group_file = os.path.join(workdir, 'mofka.json')
        driver_config = os.path.join(workdir, 'driver_config.json')
        self.bedrock_pid_file = os.path.join(workdir, 'mofka.pid')

        self.setenv('MOFKA_PROTOCOL', self.config['protocol'])
        self.setenv('MOFKA_WORKDIR', workdir)
        self.setenv('MOFKA_GROUP_FILE', group_file)
        self.setenv('MOFKA_TOPIC', self.config['topic'])
        self.setenv('MOFKA_DRIVER_CONFIG', driver_config)

        # Publish the host that will run bedrock. Under the Slurm scheduler
        # block, configure + start both execute on the first allocated node
        # (the batch node), which is also where bedrock binds via
        # LocalExecInfo below. mofka_bench reads this to drop the server
        # node from its PSSH client hostfile. Stored as the short hostname;
        # mofka_bench compares on short names too. Harmless single-node /
        # Pattern-A (the value simply won't appear in a client-only
        # hostfile, so the filter is a no-op).
        self.setenv('MOFKA_SERVER_HOST', socket.gethostname().split('.')[0])

        # Locate an active Spack environment view. $SPACK_ENV is set by
        # `spack env activate <name>` and points at a user-owned spack env
        # (e.g. ~/mofka-spack-env on Ares). Fall back to the system-spack
        # path so existing /opt/spack-env installs keep working.
        spack_env_view = None
        spack_env_root = os.environ.get('SPACK_ENV')
        if spack_env_root:
            candidate = os.path.join(spack_env_root, '.spack-env', 'view')
            if os.path.isdir(candidate):
                spack_env_view = candidate
        if spack_env_view is None and os.path.isdir(
                '/opt/spack-env/.spack-env/view'):
            spack_env_view = '/opt/spack-env/.spack-env/view'

        if spack_env_view:
            self.prepend_env('PATH', os.path.join(spack_env_view, 'bin'))
            self.prepend_env(
                'LD_LIBRARY_PATH',
                os.path.join(spack_env_view, 'lib'),
            )
            self.prepend_env(
                'LD_LIBRARY_PATH',
                os.path.join(spack_env_view, 'lib64'),
            )
            # Auto-detect Python version for site-packages
            lib_dir = os.path.join(spack_env_view, 'lib')
            if os.path.isdir(lib_dir):
                for entry in os.listdir(lib_dir):
                    if entry.startswith('python3.'):
                        python_site = os.path.join(
                            lib_dir, entry, 'site-packages'
                        )
                        if os.path.isdir(python_site):
                            self.prepend_env('PYTHONPATH', python_site)
                            break

        self.log('Mofka server configured')

    def start(self):
        workdir = self.config['workdir']
        group_file = os.path.join(workdir, 'mofka.json')
        config_path = os.path.join(self.pkg_dir, 'config', 'config.json')

        # Step 1: Prepare working directory
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
        os.makedirs(workdir, exist_ok=True)

        # Step 2: Start bedrock daemon
        self.log(f'Starting bedrock: protocol={self.config["protocol"]}')
        cmd = f'bedrock {self.config["protocol"]} -c {config_path}'
        Exec(cmd, LocalExecInfo(
            env=self.mod_env,
            cwd=workdir,
            exec_async=True,
        )).run()

        # Wait for group file
        self.log('Waiting for mofka.json...')
        for i in range(30):
            if os.path.exists(group_file):
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f'mofka.json did not appear after 30s in {workdir}'
            )
        self.log('Bedrock is ready')

        # Read and log the actual bound address from group_file. Bedrock
        # writes mofka.json as {"members":[{"address":"<scheme>://...", ...}]}.
        # Asserting against the scheme catches the silent TCP fallback failure
        # mode for RDMA: bedrock accepts `ofi+verbs` on the CLI but libfabric
        # returns a TCP-backed domain when verbs hardware init quietly fails,
        # producing a structurally-clean run with bogus comparison numbers.
        try:
            with open(group_file, 'r') as f:
                group_data = json.load(f)
            members = group_data.get('members', [])
            bound_addr = members[0].get('address', '') if members else ''
        except Exception as e:
            bound_addr = f'<failed to read mofka.json: {e}>'
        self.log(f'Bedrock bound address: {bound_addr}')
        if 'verbs' in self.config['protocol'] and 'verbs' not in bound_addr:
            raise RuntimeError(
                f'Protocol mismatch: requested protocol='
                f'{self.config["protocol"]} but bedrock bound address='
                f'{bound_addr} (silent fallback?)'
            )

        # Step 3+4: Create topic + add partition via the MofkaDriver Python
        # API (scripts/setup_topic.py). We do NOT use `mofkactl` here: on
        # current mochi builds its typer/Click CLI crashes — Click 8.2 changed
        # Parameter.make_metavar() to require a `ctx` arg and the bundled typer
        # calls it the old way, so any mofkactl invocation that renders options
        # raises "make_metavar() missing 1 required positional argument: 'ctx'".
        # The bindings themselves are fine, so we drive MofkaDriver directly.
        # See pipelines/mofka/architecture_decisions.md.
        topic = self.config['topic']
        setup_script = os.path.join(self.pkg_dir, 'scripts', 'setup_topic.py')
        self.log(f'Creating topic + partition via MofkaDriver API: {topic}')
        Exec(
            f'python3 {setup_script}'
            f' --group-file {group_file}'
            f' --topic {topic}'
            f' --partition-type {self.config["partition_type"]}',
            LocalExecInfo(env=self.mod_env),
        ).run()

        # Step 5: Write driver config
        driver_config_path = os.path.join(workdir, 'driver_config.json')
        driver_cfg = {
            'group_file': group_file,
            'margo': {'use_progress_thread': True},
        }
        with open(driver_config_path, 'w') as f:
            json.dump(driver_cfg, f, indent=4)
        self.log(f'Driver config written to {driver_config_path}')

        self.log('Mofka server is ready for benchmarks')

    def stop(self):
        self.log('Stopping bedrock daemon')
        Kill('bedrock', LocalExecInfo(env=self.mod_env), partial=True).run()
        time.sleep(1)

    def clean(self):
        # Expand here too: clean() can be called without a prior _configure()
        # in the same Python process, so we can't rely on the in-memory
        # config having already been rewritten.
        workdir = os.path.expandvars(os.path.expanduser(
            self.config.get('workdir', '/tmp/mofka-bench')))
        if os.path.exists(workdir):
            shutil.rmtree(workdir)
            self.log(f'Removed working directory: {workdir}')
