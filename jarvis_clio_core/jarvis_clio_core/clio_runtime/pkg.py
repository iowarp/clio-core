"""
IOWarp Runtime Service Package

Manages the Clio runtime deployment. Supports both bare-metal (default)
and container deployment modes via deploy_mode configuration.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo
from jarvis_cd.shell.process import Kill, GdbServer
from jarvis_cd.util import SizeType
from jarvis_cd.util.logger import Color
import os
import subprocess
import time
import yaml


# The container build for this package is driven by two template files
# next to this pkg.py:
#   - build.sh          : shell commands executed inside the jarvis pipeline
#                         build container (started from container_base by
#                         pipeline._build_pipeline_container). Installs all
#                         IOWarp build deps, clones clio-core at the chosen
#                         branch, and builds via the chosen CMakePresets
#                         entry.
#   - Dockerfile.deploy : multi-stage Dockerfile that copies the built
#                         artifacts out of the committed build image into
#                         a minimal Ubuntu deploy image.
# See jarvis-cd dev branch commit 4717991 for the single-build-container
# architecture this package targets.


class ClioRuntime(Service):
    """
    IOWarp Runtime Service supporting default and container deployment.
    """

    def _init(self):
        self.config_file = f'{self.shared_dir}/clio_config.yaml'

    def _configure_menu(self):
        return [
            {
                'name': 'num_threads',
                'msg': 'Number of worker threads for task execution',
                'type': int,
                'default': 4
            },
            {
                'name': 'process_reaper_workers',
                'msg': 'Number of process reaper worker threads',
                'type': int,
                'default': 1
            },
            {
                'name': 'main_segment_size',
                'msg': 'Main memory segment size (e.g., 1G, 512M, or "auto")',
                'type': str,
                'default': 'auto'
            },
            {
                'name': 'client_data_segment_size',
                'msg': 'Client data segment size (e.g., 512M, 256M)',
                'type': str,
                'default': '512M'
            },
            {
                'name': 'port',
                'msg': 'ZeroMQ port for networking',
                'type': int,
                'default': 9413
            },
            {
                'name': 'ipc_mode',
                'msg': 'IPC transport mode for client-server communication',
                'type': str,
                'choices': ['tcp', 'ipc', 'shm'],
                'default': 'shm'
            },
            {
                'name': 'log_level',
                'msg': 'Logging level',
                'type': str,
                'choices': ['debug', 'info', 'warning', 'error'],
                'default': 'info'
            },
            {
                'name': 'queue_depth',
                'msg': 'Task queue depth per worker',
                'type': int,
                'default': 1024
            },
            {
                'name': 'local_sched',
                'msg': 'Local task scheduler',
                'type': str,
                'default': 'default'
            },
            {
                'name': 'heartbeat_interval',
                'msg': 'Runtime heartbeat interval (milliseconds)',
                'type': int,
                'default': 1000
            },
            {
                'name': 'first_busy_wait',
                'msg': 'Busy wait duration before sleeping (microseconds)',
                'type': int,
                'default': 50
            },
            {
                'name': 'max_sleep',
                'msg': 'Maximum sleep duration cap (microseconds)',
                'type': int,
                'default': 50000
            },
            {
                'name': 'git_branch',
                'msg': 'Branch of iowarp/clio-core to clone inside the container build',
                'type': str,
                'default': 'main'
            },
            {
                'name': 'cmake_preset',
                'msg': 'CMakePresets.json preset used to configure the IOWarp build',
                'type': str,
                'default': 'release-adapter'
            },
            {
                'name': 'swim_enabled',
                'msg': ('Whether SWIM membership detection runs. When false, '
                        'the HeartbeatProbe periodic is a no-op: no direct/'
                        'indirect probes, no suspicion timeouts, no SetDead, '
                        'no recovery. Use only on stable multi-node setups '
                        'where you trust nodes not to disappear mid-run.'),
                'type': bool,
                'default': True
            },
            {
                'name': 'swim_direct_probe_timeout_sec',
                'msg': ('SWIM direct-probe timeout (seconds). A peer that '
                        'doesn\'t reply to a Heartbeat within this window is '
                        'escalated to indirect probing. Default 30s matches '
                        'the prior hard-coded value.'),
                'type': float,
                'default': 30.0
            },
            {
                'name': 'swim_indirect_probe_timeout_sec',
                'msg': ('SWIM indirect-probe timeout (seconds). If a helper '
                        'node\'s indirect probe doesn\'t return within this '
                        'window the target is marked suspected.'),
                'type': float,
                'default': 15.0
            },
            {
                'name': 'swim_suspicion_timeout_sec',
                'msg': ('SWIM suspicion timeout (seconds). A node that stays '
                        'in the suspected state this long is promoted to '
                        'dead, triggering SetDead + recovery.'),
                'type': float,
                'default': 60.0
            },
            {
                'name': 'do_start',
                'msg': ('Whether this package should actually launch the '
                        '`clio_run runtime start` daemon. Set to false when '
                        'another package in the pipeline owns the runtime '
                        'process (e.g. clio_cte_libfuse with '
                        'embedded_runtime=true), in which case clio_runtime '
                        'still generates clio_config.yaml + manages '
                        'configure/stop but skips the daemon spawn.'),
                'type': bool,
                'default': True
            },
        ]

    # ------------------------------------------------------------------
    # Container build — single-container architecture
    # ------------------------------------------------------------------
    # jarvis-cd's pipeline spins up one long-running build container from
    # container_base and exec's each package's build.sh inside it (see
    # jarvis_cd/core/pipeline.py::_build_pipeline_container). For each
    # configured package the script is template-substituted via
    # _read_build_script(...) and copy'd into the container. The container
    # is then committed to ##BUILD_IMAGE##, which Dockerfile.deploy
    # multi-stage-copies into a minimal runtime image.

    def _build_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        branch = self.config.get('git_branch', 'main')
        preset = self.config.get('cmake_preset', 'release-adapter')
        content = self._read_build_script('build.sh', {
            'GIT_BRANCH': branch,
            'CMAKE_PRESET': preset,
        })
        return content, preset

    def _build_deploy_phase(self):
        if self.config.get('deploy_mode') != 'container':
            return None
        suffix = getattr(self, '_build_suffix', '')
        content = self._read_template('Dockerfile.deploy', {
            'BUILD_IMAGE': self.build_image_name(),
            'DEPLOY_BASE': 'ubuntu:24.04',
        })
        return content, suffix

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.config_file = f'{self.shared_dir}/clio_config.yaml'

        self.setenv('CLIO_SERVER_CONF', self.config_file)
        self.setenv('CTP_LOG_LEVEL', self.config['log_level'])
        self.setenv('CLIO_IPC_MODE', self.config['ipc_mode'].upper())

        self._generate_config()

        self.log(f"IOWarp runtime configured")
        self.log(f"  Config file: {self.config_file}")

    def _generate_config(self):
        if self.config['main_segment_size'] == 'auto':
            main_size = 'auto'
        else:
            main_size = SizeType(self.config['main_segment_size']).bytes
        client_size = SizeType(self.config['client_data_segment_size']).bytes

        # Prefer the hostfile copy that jarvis.pipeline.save() stamps into
        # the pipeline's shared_dir (<pipeline_shared_dir>/hostfile). That
        # location is bind-mounted into every deploy container at the same
        # path and is reachable from both host and container. Fall back to
        # the effective hostfile's original path only if the copy doesn't
        # exist (e.g., running before pipeline.save() has been called).
        pipeline_shared = self.jarvis.get_pipeline_shared_dir(self.pipeline.name)
        hostfile_shared = os.path.join(str(pipeline_shared), 'hostfile')
        if os.path.exists(hostfile_shared):
            hostfile_path = hostfile_shared
        else:
            hostfile_path = self.hostfile.path if self.hostfile.path else ''

        config_dict = {
            'memory': {
                'main_segment_size': main_size,
                'client_data_segment_size': client_size
            },
            'networking': {
                'port': self.config['port'],
                'hostfile': hostfile_path
            },
            'logging': {
                'level': self.config['log_level'],
                'file': f"{self.shared_dir}/clio_run.log"
            },
            'runtime': {
                'num_threads': self.config['num_threads'],
                'process_reaper_threads': self.config['process_reaper_workers'],
                'queue_depth': self.config['queue_depth'],
                'local_sched': self.config['local_sched'],
                'heartbeat_interval': self.config['heartbeat_interval'],
                'first_busy_wait': self.config['first_busy_wait'],
                'max_sleep': self.config['max_sleep']
            },
            # Parsed by clio::run::ConfigManager::ParseYAML — keys must match
            # the names used there (swim.enabled, swim.direct_probe_timeout_sec,
            # swim.indirect_probe_timeout_sec, swim.suspicion_timeout_sec).
            'swim': {
                'enabled': self.config['swim_enabled'],
                'direct_probe_timeout_sec':
                    self.config['swim_direct_probe_timeout_sec'],
                'indirect_probe_timeout_sec':
                    self.config['swim_indirect_probe_timeout_sec'],
                'suspicion_timeout_sec':
                    self.config['swim_suspicion_timeout_sec'],
            }
        }

        with open(self.config_file, 'w') as f:
            f.write('# Clio Runtime Configuration\n\n')
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        # When `do_start: false`, another package in the pipeline owns
        # the clio_run runtime process (e.g. clio_cte_libfuse with
        # embedded_runtime=true). clio_runtime stays in the pipeline to
        # generate clio_config.yaml and handle configure/stop, but
        # the daemon spawn is skipped to avoid a port-9413 conflict.
        # Ordering note: any package that consumes the runtime as a
        # client (clio_cte, workloads) must appear *after* the package
        # that actually owns the runtime in the pipeline pkg list.
        if not self.config.get('do_start', True):
            self.log("do_start=false: skipping clio_run daemon spawn "
                     "(another package owns the runtime)",
                     color=Color.YELLOW)
            return

        self.log("Starting IOWarp runtime")

        cmd = 'clio_run runtime start'

        exec_info = PsshExecInfo(
            env=self.env,
            hostfile=self.hostfile,
            exec_async=True,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            private_dir=self.private_dir,
            bind_mounts=self.container_mounts,
        )
        if self.config.get('do_dbg', False):
            GdbServer(cmd, self.config['dbg_port'], exec_info).run()
        else:
            Exec(cmd, exec_info).run()

        self.sleep()

        port = self.config['port']
        host = self.hostfile.hosts[0] if self.hostfile.hosts else '127.0.0.1'
        self.log(f'Waiting for runtime on {host}:{port}', color=Color.YELLOW)
        for i in range(30):
            try:
                ret = subprocess.run(
                    ['bash', '-c', f'echo > /dev/tcp/{host}/{port}'],
                    capture_output=True, timeout=2)
                if ret.returncode == 0:
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(1)
        else:
            self.log(f'WARNING: Runtime did not respond on {host}:{port} after 30s',
                     color=Color.RED)

        self.log("IOWarp runtime started")

    def stop(self):
        # Symmetric with start(): when this package didn't spawn the
        # runtime, it isn't responsible for stopping it either. The
        # package that owns the runtime (e.g. clio_cte_libfuse) tears
        # down its own daemon during its own stop().
        if not self.config.get('do_start', True):
            self.log("do_start=false: skipping clio_run daemon stop "
                     "(another package owns the runtime)",
                     color=Color.YELLOW)
            return

        self.log("Stopping IOWarp runtime")

        Exec('clio_run runtime stop',
             PsshExecInfo(env=self.env, hostfile=self.hostfile)).run()
        Kill('clio_run',
             PsshExecInfo(env=self.env, hostfile=self.hostfile)).run()

        port = self.config['port']
        host = self.hostfile.hosts[0] if self.hostfile.hosts else '127.0.0.1'
        for i in range(10):
            try:
                ret = subprocess.run(
                    ['bash', '-c', f'echo > /dev/tcp/{host}/{port}'],
                    capture_output=True, timeout=2)
                if ret.returncode != 0:
                    break
            except subprocess.TimeoutExpired:
                break
            time.sleep(1)
        time.sleep(1)

        self.log("IOWarp runtime stopped")

    def kill(self):
        self.log("Forcibly killing IOWarp runtime")
        Kill('clio_run', PsshExecInfo(
            hostfile=self.hostfile
        )).run()

    def clean(self):
        self.log("Cleaning IOWarp runtime data")

        if self.config_file and os.path.exists(self.config_file):
            os.remove(self.config_file)

        log_file = f'{self.shared_dir}/clio_run.log'
        if os.path.exists(log_file):
            os.remove(log_file)

        Exec('rm -f /dev/shm/chi_*', PsshExecInfo(
            hostfile=self.hostfile
        )).run()
