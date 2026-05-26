"""
Content Transfer Engine (CTE) configuration service for IOWarp.

Configures and deploys CTE by generating a clio_run compose YAML config
and running `clio_run compose`. Supports both bare-metal and container modes.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.util import SizeType
from jarvis_cd.shell.process import Rm, Mkdir
from jarvis_cd.shell import PsshExecInfo, Exec
import yaml
import os
import re


class ClioCte(Service):
    """
    CTE Service supporting default and container deployment.

    deploy_mode='default': runs clio_run compose on the host.
    deploy_mode='container': runs clio_run compose inside the deploy container.
    """

    def _init(self):
        self.compose_config_path = os.path.join(self.shared_dir, 'cte_compose.yaml')

    def _configure_menu(self):
        return [
            {
                'name': 'pool_name',
                'msg': ('Name of the CTE pool. MUST match the C++ client\'s '
                        'kCtePoolName constant ("clio_cte_core") — that\'s '
                        'the name the FUSE / GPU adapters use when they '
                        'do AsyncCreate during ClientInit. If compose '
                        'registers a different name (e.g. "clio_cte"), the '
                        'client\'s AsyncCreate then *also* tries to create '
                        'pool "clio_cte_core" at the same pool_id; on '
                        'multi-node the cross-node admin propagation of '
                        'that second create races with the first, leaving '
                        'the remote container half-initialized and FUSE '
                        'callbacks wedged.'),
                'type': str,
                'default': 'clio_cte_core'
            },
            {
                'name': 'pool_id',
                'msg': 'Pool ID for the CTE pool',
                'type': float,
                'default': 512.0
            },
            {
                'name': 'pool_query',
                'msg': 'Pool query type (local or dynamic)',
                'type': str,
                'choices': ['local', 'dynamic'],
                'default': 'local'
            },
            {
                'name': 'devices',
                'msg': 'List of storage devices as tuples (path, capacity, score)',
                'type': list,
                'default': [],
            },
            {
                'name': 'dpe_type',
                'msg': 'Data Placement Engine type',
                'type': str,
                'choices': ['random', 'round_robin', 'max_bw'],
                'default': 'max_bw'
            },
            {
                'name': 'neighborhood',
                'msg': 'Number of targets (nodes CTE can buffer to)',
                'type': int,
                'default': 4
            },
            {
                'name': 'default_target_timeout_ms',
                'msg': 'Default target timeout in milliseconds',
                'type': int,
                'default': 30000
            },
            {
                'name': 'poll_period_ms',
                'msg': 'Period at which targets should be rescanned (ms)',
                'type': int,
                'default': 5000
            },
            {
                'name': 'monitor_interval_ms',
                'msg': 'Compression monitor interval (ms)',
                'type': int,
                'default': 5
            },
            {
                'name': 'dnn_model_weights_path',
                'msg': 'Path to DNN model weights JSON file (empty = disabled)',
                'type': str,
                'default': ''
            },
            {
                'name': 'dnn_samples_before_reinforce',
                'msg': 'Samples to collect before reinforcing DNN model',
                'type': int,
                'default': 1000
            },
            {
                'name': 'trace_folder_path',
                'msg': 'Path to folder for CTE trace logs (empty = disabled)',
                'type': str,
                'default': ''
            },
            {
                'name': 'iowarp_compress',
                'msg': 'Compression mode for IOWarp Engine',
                'type': str,
                'default': 'none',
            },
            {
                'name': 'iowarp_compress_trace',
                'msg': 'Enable compression tracing',
                'type': str,
                'choices': ['on', 'off'],
                'default': 'off',
            }
        ]

    # ------------------------------------------------------------------
    # Container — no separate build needed, shares clio_runtime's image
    # ------------------------------------------------------------------

    def _build_deploy_phase(self) -> str:
        """
        No build needed — clio_run compose is already in the container_base
        image (e.g., iowarp/deploy-cpu:latest).
        """
        return None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.compose_config_path = os.path.join(self.shared_dir, 'cte_compose.yaml')
        self.log("Configuring Content Transfer Engine (CTE)...")

        devices = self.config.get('devices', [])
        if not devices:
            devices = self._get_devices_from_resource_graph()
            if not devices:
                devices = self._get_default_devices()
        else:
            devices = self._validate_and_convert_devices(devices)

        compose_config = self._build_compose_config(devices)

        with open(self.compose_config_path, 'w') as f:
            f.write('# Content Transfer Engine (CTE) Compose Configuration\n\n')
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)

        for path, _, _ in devices:
            if path.startswith('ram::'):
                continue
            parent_dir = os.path.dirname(path)
            if not parent_dir:
                continue
            Mkdir(parent_dir,
                  PsshExecInfo(env=self.mod_env, hostfile=self.hostfile)).run()

        self.log("CTE configuration completed")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.log("Starting CTE using clio_run compose...")

        if not os.path.exists(self.compose_config_path):
            self.log(f"Error: Compose config not found: {self.compose_config_path}")
            return False

        # WORKAROUND — proper fix lives in clio-core (chimaera client).
        #
        # At >=64 chimaera daemons on Aurora apptainer the very first
        # `clio_run compose` after clio_runtime startup hits a ZMTP greeting
        # timeout against the daemon's local 9416 ROUTER (the daemon's I/O
        # threads are still saturated by initial SWIM probes). The compose
        # process's ZMQ shared context (chimaera GetSharedContext singleton)
        # then ends up half-open, and no in-process retry can recover from
        # it: ROUTER_HANDOVER=1, in-process DEALER recreate, and
        # WaitForLocalServer per-attempt timeouts were all tried and all
        # fail because the broken state is in the ZMQ ctx, not the socket.
        # A brand-new chimaera process gets a fresh context and connects in
        # <1s. The bash loop forks a new process per retry to sidestep the
        # in-process recovery problem.
        #
        # Real fix (TODO, in clio-core/context-runtime):
        #   1. Tear-down + recreate the ZMQ ctx on WaitForLocalServer
        #      failure (GetSharedContext needs coordinated shutdown), OR
        #   2. Make IsServerAlive ZMTP-aware (currently it does only a TCP
        #      connect() probe, so server_alive_ stays true on a half-open
        #      ctx and the existing reconnect path is never triggered).
        # When either lands, drop this loop and revert to:
        #   cmd = f'clio_run compose {self.compose_config_path}'
        # Single-shot compose. The jarvis-cd SSH layer prepends env vars
        # as ``KEY=VAL`` before the command, and bash only forwards them
        # to a *simple* command — wrapping in ``for ... do ... done``
        # would strip the env (notably CLIO_SERVER_CONF), causing the
        # clio_run compose client to fall back to ~/.clio/clio.yaml and
        # load unrelated compose entries that occupy our target pool IDs.
        # The original retry loop existed for an Aurora apptainer
        # ZMTP-greeting race at >=64 daemons; for the bare-metal /
        # single-node path here the simple form is enough.
        cmd = f'clio_run compose {self.compose_config_path}'

        # Pssh fans the compose out to every node. With pool_query:
        # broadcast in the compose YAML, each admin container instance
        # handles a Create replica directly, so the per-node invocations
        # are idempotent: the first to register the pool wins, the
        # others see "Pool with name '...' already exists" and return
        # the existing PoolId. Net effect is one successful pool create
        # per node, which is exactly what 2n needs.
        Exec(cmd, PsshExecInfo(
            env=self.mod_env,
            hostfile=self.hostfile,
            container=getattr(self, '_container_engine', 'none'),
            container_image=self.deploy_image_name(),
            private_dir=self.private_dir,
            bind_mounts=self.container_mounts,
        )).run()

        self.log("CTE started successfully")
        return True

    def stop(self):
        pass

    def kill(self):
        pass

    def clean(self):
        self.log("Cleaning CTE data...")

        if self.compose_config_path and os.path.exists(self.compose_config_path):
            os.remove(self.compose_config_path)

        devices = self.config.get('devices', [])
        if not devices:
            devices.extend(self._get_devices_from_resource_graph())

        for path, _, _ in devices:
            if path.startswith('ram::'):
                continue
            try:
                Rm(path, PsshExecInfo(hostfile=self.hostfile)).run()
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    Rm(parent_dir, PsshExecInfo(hostfile=self.hostfile)).run()
            except Exception as e:
                self.log(f"Error cleaning {path}: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_devices_from_resource_graph(self):
        try:
            from jarvis_cd.core.resource_graph import ResourceGraphManager
            rg_manager = ResourceGraphManager()
            if not rg_manager.resource_graph.get_all_nodes():
                return []
            common_storage = rg_manager.resource_graph.get_common_storage()
            if not common_storage:
                return []
            devices = []
            for _, device_list in common_storage.items():
                for device in device_list:
                    base_path = device.get('mount', '/tmp/cte_storage')
                    path = os.path.join(base_path, 'hermes_data.bin')
                    available_space = device.get('avail', '100GB')
                    capacity = self._adjust_capacity(available_space, 0.5)
                    device_type = device.get('dev_type', 'unknown').lower()
                    score = self._calculate_device_score(device_type, device)
                    devices.append((path, capacity, score))
            return devices
        except Exception:
            return []

    def _adjust_capacity(self, capacity_str, factor):
        match = re.match(r'([\d.]+)\s*([KMGT]?B?)', capacity_str.upper().strip())
        if not match:
            return capacity_str
        value = float(match.group(1))
        suffix = match.group(2)
        adjusted = value * factor
        if adjusted == int(adjusted):
            return f"{int(adjusted)}{suffix}"
        return f"{adjusted:.2f}{suffix}"

    def _calculate_device_score(self, storage_type, storage_info):
        type_scores = {
            'ram': 1.0, 'ramdisk': 1.0, 'tmpfs': 1.0,
            'nvme': 0.9, 'ssd': 0.7, 'hdd': 0.4,
            'network': 0.3, 'unknown': 0.5
        }
        return round(type_scores.get(storage_type, 0.5), 2)

    def _get_default_devices(self):
        # Default backing store lives inside the pipeline's shared_dir
        # rather than /tmp: shared_dir is bind-mounted into the deploy
        # container at the same path, so the host-side Mkdir in
        # _configure() is actually visible inside the container when
        # `clio_run compose` later opens the file. /tmp in a container
        # is its own ephemeral tmpfs and won't see host mkdirs.
        return [
            (f'{self.shared_dir}/cte_storage/cte_target.bin', '100GB', 0.6)
        ]

    def _validate_and_convert_devices(self, devices):
        validated = []
        for i, device in enumerate(devices):
            try:
                if isinstance(device, (list, tuple)) and len(device) >= 3:
                    path, capacity, score = device[0], device[1], device[2]
                    validated.append((str(path), str(SizeType(capacity)), float(score)))
            except Exception as e:
                self.log(f"Warning: Invalid device {i}: {device} - {e}")
        return validated if validated else self._get_default_devices()

    def _build_compose_config(self, devices):
        storage_config = []
        for path, capacity, score in devices:
            is_ram = path.startswith('ram::')
            bdev_type = 'ram' if is_ram else 'file'
            # CTE FlushData ranks targets by `persistence_level_` (string
            # in StorageDeviceConfig, mapped to enum kVolatile / kTemporary
            # / kLongTerm). The default is "volatile" for every device type
            # — so a file-backed bdev on PFS gets registered as volatile
            # and FlushData with min_persistence_level >= 1 finds no
            # target and the gray-scott write path blocks. Tag file bdevs
            # as "long_term" by default so the tier is correctly chosen.
            persistence_level = 'volatile' if is_ram else 'long_term'
            storage_config.append({
                'path': path, 'bdev_type': bdev_type,
                'capacity_limit': capacity, 'score': score,
                'persistence_level': persistence_level
            })

        compose_list = [{
            'mod_name': 'clio_cte_core',
            'pool_name': self.config.get('pool_name', 'clio_cte_core'),
            'pool_query': self.config.get('pool_query', 'local'),
            'pool_id': self.config.get('pool_id', 512.0),
            'targets': {
                'neighborhood': self.config.get('neighborhood', 4),
                'default_target_timeout_ms': self.config.get('default_target_timeout_ms', 30000),
                'poll_period_ms': self.config.get('poll_period_ms', 5000)
            },
            'storage': storage_config,
            'dpe': {'dpe_type': self.config.get('dpe_type', 'max_bw')},
            'compression': {
                'monitor_interval_ms': self.config.get('monitor_interval_ms', 5),
                'dnn_model_weights_path': self.config.get('dnn_model_weights_path', ''),
                'dnn_samples_before_reinforce': self.config.get('dnn_samples_before_reinforce', 1000),
                'trace_folder_path': self.config.get('trace_folder_path', '')
            }
        }]

        iowarp_compress = self.config.get('iowarp_compress', 'none').lower()
        if iowarp_compress not in ['none', 'off', '']:
            compose_list.append({
                'mod_name': 'clio_cte_compressor',
                'pool_name': 'clio_cte_compressor',
                'pool_query': self.config.get('pool_query', 'local'),
                'pool_id': self.config.get('pool_id', 512.0) + 1
            })

        return {'compose': compose_list}
