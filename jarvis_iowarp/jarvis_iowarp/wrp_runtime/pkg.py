"""
IOWarp Runtime Service Package

This package deploys and manages the IOWarp (Chimaera) runtime service across distributed nodes.
Assumes chimaera has been installed and binaries are available in PATH.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_cd.shell.process import Kill, GdbServer
from jarvis_cd.util import SizeType
import os
import yaml


class WrpRuntime(Service):
    """
    IOWarp Runtime Service

    Manages the Chimaera runtime deployment across distributed nodes,
    including configuration generation and runtime lifecycle management.

    Assumes chimaera_start_runtime and chimaera_stop_runtime are installed
    and available in PATH.
    """

    def _init(self):
        """Initialize package-specific variables"""
        self.config_file = None

    def _configure_menu(self):
        """Define configuration options for IOWarp runtime"""
        return [
            {
                'name': 'sched_workers',
                'msg': 'Number of unified scheduler worker threads',
                'type': int,
                'default': 8
            },
            {
                'name': 'process_reaper_workers',
                'msg': 'Number of process reaper worker threads',
                'type': int,
                'default': 1
            },
            {
                'name': 'main_segment_size',
                'msg': 'Main memory segment size (e.g., 1G, 512M)',
                'type': str,
                'default': '1G'
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
                'default': 5555
            },
            {
                'name': 'log_level',
                'msg': 'Logging level',
                'type': str,
                'choices': ['debug', 'info', 'warning', 'error'],
                'default': 'info'
            },
            {
                'name': 'stack_size',
                'msg': 'Stack size per task (bytes)',
                'type': int,
                'default': 65536
            },
            {
                'name': 'queue_depth',
                'msg': 'Task queue depth',
                'type': int,
                'default': 10000
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
            }
        ]

    def _configure(self, **kwargs):
        """Configure the IOWarp runtime service"""
        # Set configuration file path in shared directory
        self.config_file = f'{self.shared_dir}/chimaera_config.yaml'

        # Set the CHI_SERVER_CONF environment variable
        # This is what both RuntimeInit and ClientInit check
        self.setenv('CHI_SERVER_CONF', self.config_file)

        # Set HSHM_LOG_LEVEL for debug logging
        self.setenv('HSHM_LOG_LEVEL', self.config['log_level'])

        # Generate chimaera configuration
        self._generate_config()

        self.log(f"IOWarp runtime configured")
        self.log(f"  Config file: {self.config_file}")
        self.log(f"  CHI_SERVER_CONF: {self.config_file}")
        self.log(f"  HSHM_LOG_LEVEL: {self.config['log_level']}")

    def _generate_config(self):
        """Generate Chimaera runtime configuration file"""
        # Parse size strings to bytes
        main_size = SizeType(self.config['main_segment_size']).bytes
        client_size = SizeType(self.config['client_data_segment_size']).bytes

        # Build configuration dictionary matching chimaera_default.yaml format
        # Worker threads are now consolidated into runtime section
        config_dict = {
            'memory': {
                'main_segment_size': main_size,
                'client_data_segment_size': client_size
            },
            'networking': {
                'port': self.config['port'],
                'hostfile': self.jarvis.hostfile.path if self.jarvis.hostfile.path else ''
            },
            'logging': {
                'level': self.config['log_level'],
                'file': f"{self.shared_dir}/chimaera.log"
            },
            'runtime': {
                # Worker thread configuration
                'sched_threads': self.config['sched_workers'],
                'process_reaper_threads': self.config['process_reaper_workers'],
                # Task execution configuration
                'stack_size': self.config['stack_size'],
                'queue_depth': self.config['queue_depth'],
                'local_sched': self.config['local_sched'],
                'heartbeat_interval': self.config['heartbeat_interval'],
                # Worker sleep configuration
                'first_busy_wait': self.config['first_busy_wait'],
                'max_sleep': self.config['max_sleep']
            }
        }

        # Write configuration to YAML file
        with open(self.config_file, 'w') as f:
            f.write('# Chimaera Runtime Configuration\n')
            f.write('# Generated by Jarvis IOWarp Runtime Package\n\n')
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        self.log(f"Generated Chimaera configuration: {self.config_file}")

    def start(self):
        """Start the IOWarp runtime service on all nodes""" 
        # Launch runtime on all nodes using PsshExecInfo
        # IMPORTANT: Use env (shared environment), not mod_env
        self.log(f"Starting IOWarp runtime on all nodes")
        self.log(f"  Config (CHI_SERVER_CONF): {self.config_file}")
        self.log(f"  Nodes: {len(self.jarvis.hostfile)}")

        # The chimaera_start_runtime binary will read CHI_SERVER_CONF from environment
        cmd = 'chimaera_start_runtime'

        # Execute with or without debugging
        if self.config.get('do_dbg', False):
            self.log(f"Starting with GDB server on port {self.config['dbg_port']}")
            GdbServer(cmd, self.config['dbg_port'], PsshExecInfo(
                env=self.env,
                hostfile=self.jarvis.hostfile,
                exec_async=True
            )).run()
        else:
            Exec(cmd, PsshExecInfo(
                env=self.env,  # Use env, not mod_env
                hostfile=self.jarvis.hostfile,
                exec_async=True
            )).run()

        self.sleep()

        self.log("IOWarp runtime started successfully on all nodes")

    def stop(self):
        """Stop the IOWarp runtime service on all nodes"""
        self.log("Stopping IOWarp runtime on all nodes")

        # Use chimaera_stop_runtime to gracefully shutdown
        # The stop binary will also read CHI_SERVER_CONF from environment
        cmd = 'chimaera_stop_runtime'

        Exec(cmd, LocalExecInfo(
            env=self.env,
            hostfile=self.jarvis.hostfile
        )).run()

        self.log("IOWarp runtime stopped on all nodes")

    def kill(self):
        """Forcibly terminate the IOWarp runtime on all nodes"""
        self.log("Forcibly killing IOWarp runtime on all nodes")

        Kill('chimaera_start_runtime', PsshExecInfo(
            hostfile=self.jarvis.hostfile
        )).run()

        self.log("IOWarp runtime killed on all nodes")

    def status(self) -> str:
        """Check IOWarp runtime status"""
        # Could enhance this by checking if processes are actually running
        # For now, return basic status
        return "unknown"

    def clean(self):
        """Clean IOWarp runtime data and temporary files"""
        self.log("Cleaning IOWarp runtime data")

        # Remove configuration file
        if self.config_file and os.path.exists(self.config_file):
            os.remove(self.config_file)

        # Remove log file
        log_file = f'{self.shared_dir}/chimaera.log'
        if os.path.exists(log_file):
            os.remove(log_file)

        # Clean shared memory segments on all nodes
        self.log("Cleaning shared memory segments on all nodes")
        cmd = 'rm -f /dev/shm/chi_*'
        Exec(cmd, PsshExecInfo(
            hostfile=self.jarvis.hostfile
        )).run()

        self.log("Cleanup completed")
