"""
IOWarp transparent compression service for the CTE stack.

Generates a clio_run-compose YAML that places the clio_cte_compressor
module at the configured ``pool_id`` (default 512.0 — the CTE entrypoint
that adapters target via CLIO_CTE_CLIENT_INIT) and points it at the
downstream clio_cte_core via ``next_pool_id`` (default 513.0). Pairs
naturally with jarvis_clio_core.clio_cte configured at 513.0.

This package only configures the pipeline-side compose entry. The
underlying chimod (clio_cte_compressor) is built and installed as part of
context-transfer-engine; clio_runtime loads it at daemon start.
"""
from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo
import os
import yaml


class ClioCompress(Service):
    """
    Compressor Service for the CTE I/O stack.

    deploy_mode='default':   runs `clio_run compose` on the host.
    deploy_mode='container': runs `clio_run compose` inside the deploy
                             container (shares clio_runtime's image).
    """

    def _init(self):
        self.compose_config_path = os.path.join(
            self.shared_dir, 'compress_compose.yaml')

    def _configure_menu(self):
        return [
            {
                'name': 'pool_name',
                'msg': 'Name of the compressor pool',
                'type': str,
                'default': 'clio_cte_compressor',
            },
            {
                'name': 'pool_id',
                'msg': ('Pool ID for the compressor (this should be the '
                        'entrypoint pool — i.e., the same ID adapters '
                        'target via CLIO_CTE_CLIENT_INIT, default 512.0)'),
                'type': float,
                'default': 512.0,
            },
            {
                'name': 'next_pool_id',
                'msg': ('Pool ID of the downstream module (the clio_cte_core '
                        'pool that the compressor forwards compressed '
                        'blobs to)'),
                'type': float,
                'default': 513.0,
            },
            {
                'name': 'pool_query',
                'msg': 'Pool query type (local or dynamic)',
                'type': str,
                'choices': ['local', 'dynamic'],
                'default': 'local',
            },
            {
                'name': 'compress_lib',
                'msg': ('Default compression library applied by the '
                        'transparent path. Used for adapters that do '
                        'not set Context.compress_lib_ explicitly. '
                        '"none" disables compression.'),
                'type': str,
                'choices': [
                    'none', 'snappy', 'lz4', 'zstd', 'zlib', 'bzip2',
                    'brotli', 'lzma', 'blosc2', 'fpzip', 'sz3', 'zfp',
                ],
                'default': 'snappy',
            },
            {
                'name': 'compress_preset',
                'msg': ('Compression preset (balanced=0, best=1, '
                        'default=2, fast=3)'),
                'type': str,
                'choices': ['balanced', 'best', 'default', 'fast'],
                'default': 'default',
            },
            {
                'name': 'qtable_model_path',
                'msg': 'Path to Q-table model JSON (empty = disabled)',
                'type': str,
                'default': '',
            },
            {
                'name': 'linreg_model_path',
                'msg': 'Path to LinReg table model JSON (empty = disabled)',
                'type': str,
                'default': '',
            },
            {
                'name': 'distribution_model_path',
                'msg': ('Path to distribution classifier model '
                        '(empty = factory defaults)'),
                'type': str,
                'default': '',
            },
            {
                'name': 'dnn_model_weights_path',
                'msg': 'Path to DNN model weights JSON (empty = disabled)',
                'type': str,
                'default': '',
            },
            {
                'name': 'trace_folder_path',
                'msg': 'Folder to write compression trace logs (empty = disabled)',
                'type': str,
                'default': '',
            },
            # ----------------------------------------------------------
            # NeuroPress neural selection (issue #693)
            # ----------------------------------------------------------
            # neuropress_model_path is the master switch: the compressor
            # only builds the predictor when it is non-empty, so every
            # option below is inert without it. It needs the DIRECTORY
            # holding the trained .nnwt weights, not the file.
            #
            # Selection also requires the caller to ask for dynamic
            # compression -- Context.dynamic_compress_ is 0=skip, 1=static,
            # 2=dynamic, and the NeuroPress gate is `!= 1`. A workload
            # pinning a library gets that library, model or no model.
            {
                'name': 'neuropress_model_path',
                'msg': ('Directory of trained NeuroPress .nnwt weights '
                        '(empty = NeuroPress disabled)'),
                'type': str,
                'default': '',
            },
            # Off by default, mirroring NeuroPress's own
            # g_online_learning_enabled{false}: pointing at weights must
            # give INFERENCE ONLY. A deployment that just wants the trained
            # model should not silently get one whose weights drift.
            {
                'name': 'neuropress_online_learning_enabled',
                'msg': 'Enable NeuroPress online SGD learning',
                'type': bool,
                'default': False,
            },
            {
                'name': 'neuropress_mape_threshold',
                'msg': ('Weighted-cost error above which online SGD fires '
                        '(0.30 = 30%)'),
                'type': float,
                'default': 0.30,
            },
            {
                'name': 'neuropress_learning_rate',
                'msg': 'Online SGD step size',
                'type': float,
                'default': 0.01,
            },
            # Exploration compresses alternatives purely to generate
            # training samples and never stores them. Opt-in, and costly:
            # a full sweep is ~32 compressions per chunk.
            {
                'name': 'neuropress_exploration_enabled',
                'msg': 'Enable K-way exploration (requires online learning)',
                'type': bool,
                'default': False,
            },
            {
                'name': 'neuropress_exploration_threshold',
                'msg': 'Cost error above which a sweep runs (0.50 = 50%)',
                'type': float,
                'default': 0.50,
            },
            {
                'name': 'neuropress_exploration_k',
                'msg': 'Alternatives measured per sweep (31 = whole space)',
                'type': int,
                'default': 3,
            },
            # Measurement mode, not a faster one: every chunk is explored
            # exhaustively and both SGD phases are suppressed, so it
            # establishes the ceiling on selection quality rather than
            # improving throughput.
            {
                'name': 'neuropress_best_mode',
                'msg': 'Exhaustive-search mode (slow; measurement only)',
                'type': bool,
                'default': False,
            },
        ]

    # ------------------------------------------------------------------
    # Container — no separate build needed, shares clio_runtime's image
    # ------------------------------------------------------------------

    def _build_deploy_phase(self) -> str:
        return None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _format_pool_id(pool_id) -> str:
        # clio_run compose expects "<major>.<minor>" — coerce 512 -> "512.0"
        # but preserve user-supplied 513.7 etc.
        if isinstance(pool_id, str):
            return pool_id
        as_float = float(pool_id)
        if as_float.is_integer():
            return f"{int(as_float)}.0"
        return repr(as_float)

    _COMPRESS_LIB_IDS = {
        'brotli': 0,
        'bzip2': 1,
        'blosc2': 2,
        'fpzip': 3,
        'lz4': 4,
        'lzma': 5,
        'snappy': 6,
        'sz3': 7,
        'zfp': 8,
        'zlib': 9,
        'zstd': 10,
    }

    _COMPRESS_PRESET_IDS = {
        'balanced': 0,
        'best': 1,
        'default': 2,
        'fast': 3,
    }

    def _configure(self, **kwargs):
        super()._configure(**kwargs)

        self.compose_config_path = os.path.join(
            self.shared_dir, 'compress_compose.yaml')
        self.log("Configuring transparent compression service (clio_compress)...")

        compose_entry = {
            'mod_name': 'clio_cte_compressor',
            'pool_name': self.config.get('pool_name', 'clio_cte_compressor'),
            'pool_query': self.config.get('pool_query', 'local'),
            'pool_id': self._format_pool_id(self.config.get('pool_id', 512.0)),
            'next_pool_id': self._format_pool_id(
                self.config.get('next_pool_id', 513.0)),
        }

        # Optional model / trace paths — only emit when non-empty so the
        # runtime's path-empty short-circuits keep behaving as before.
        for key in ('qtable_model_path', 'linreg_model_path',
                    'distribution_model_path', 'dnn_model_weights_path',
                    'trace_folder_path', 'neuropress_model_path'):
            value = self.config.get(key, '')
            if value:
                compose_entry[key] = value

        # NeuroPress tunables ride along only when the model is configured.
        # Emitting them on their own would put keys in the compose file that
        # tune a predictor the compressor never builds — parsed, applied to
        # the config struct, and with nothing to act on.
        if self.config.get('neuropress_model_path', ''):
            for key in ('neuropress_online_learning_enabled',
                        'neuropress_mape_threshold',
                        'neuropress_learning_rate',
                        'neuropress_exploration_enabled',
                        'neuropress_exploration_threshold',
                        'neuropress_exploration_k',
                        'neuropress_best_mode'):
                if key in self.config:
                    compose_entry[key] = self.config[key]

        compose_config = {'compose': [compose_entry]}

        with open(self.compose_config_path, 'w') as f:
            f.write('# clio_compress clio_run-compose configuration\n\n')
            yaml.dump(compose_config, f, default_flow_style=False, indent=2)

        # Stash the chosen library/preset as env vars so adapters that
        # honor them (or wrappers that read them) can pick the same
        # default the user requested. The compressor module currently
        # reads compress_lib_ from per-task Context, so this is a
        # forward-compatible pass-through rather than a hard requirement.
        compress_lib = self.config.get('compress_lib', 'snappy').lower()
        compress_preset = self.config.get('compress_preset', 'default').lower()
        self.setenv('CLIO_CTE_COMPRESS_DEFAULT_LIB', compress_lib)
        self.setenv('CLIO_CTE_COMPRESS_DEFAULT_PRESET', compress_preset)
        if compress_lib in self._COMPRESS_LIB_IDS:
            self.setenv('CLIO_CTE_COMPRESS_DEFAULT_LIB_ID',
                        str(self._COMPRESS_LIB_IDS[compress_lib]))
        if compress_preset in self._COMPRESS_PRESET_IDS:
            self.setenv('CLIO_CTE_COMPRESS_DEFAULT_PRESET_ID',
                        str(self._COMPRESS_PRESET_IDS[compress_preset]))

        # Point the HDF5 VOL connector at this compressor pool.
        #
        # This is what makes transparent compression reachable at all for an
        # HDF5 application. clio_adapters sets HDF5_VOL_CONNECTOR=clio and
        # LD_PRELOADs the connector, but the connector only builds a
        # compressor client when it knows which pool to talk to
        # (clio_vol.cc's clio_resolve_compressor_pool reads
        # CLIO_VOL_COMPRESSOR_POOL); with it unset, clio_make_file leaves
        # file->compressor_client null and every write skips
        # AsyncDynamicSchedule -- so the pool is composed, the model is
        # loaded, and nothing is ever compressed.
        #
        # It belongs here rather than in clio_adapters because the pool id is
        # this package's own configuration; the adapter has no way to know it.
        #
        # The VOL is currently the ONLY adapter that calls DynamicSchedule.
        # The PutBlob path the other adapters use (ADIOS2, FUSE, POSIX,
        # MPI-IO) has no dynamic-selection branch, so it reaches static
        # compression at best and never the NeuroPress model.
        self.setenv('CLIO_VOL_COMPRESSOR_POOL', compose_entry['pool_id'])

        self.log(f"clio_compress: compose written to {self.compose_config_path} "
                 f"(pool {compose_entry['pool_id']} -> "
                 f"next {compose_entry['next_pool_id']}, "
                 f"lib={compress_lib}, preset={compress_preset})")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self.log("Starting clio_compress via clio_run compose...")

        if not os.path.exists(self.compose_config_path):
            self.log(f"Error: Compose config not found: {self.compose_config_path}")
            return False

        # Single-shot compose. The jarvis-cd SSH layer prepends
        # ``KEY=VAL`` env vars to the command string; bash only attaches
        # those to a *simple* command, so a wrapping ``for ... do ...
        # done`` retry loop strips the env (notably CLIO_SERVER_CONF) and
        # the clio_run compose client falls back to ~/.clio/clio.yaml,
        # picking up unrelated compose entries that occupy our target
        # pool ID. Keep this a single command so the env prefix reaches
        # clio_run.
        cmd = f'clio_run compose start {self.compose_config_path}'

        Exec(cmd, PsshExecInfo(
            env=self.mod_env,
            hostfile=self.jarvis.hostfile,
            container=self._container_engine,
            container_image=self.deploy_image_name(),
            private_dir=self.private_dir,
            bind_mounts=self.container_mounts,
        )).run()

        self.log("clio_compress started successfully")
        return True

    def stop(self):
        pass

    def kill(self):
        pass

    def clean(self):
        if (self.compose_config_path
                and os.path.exists(self.compose_config_path)):
            os.remove(self.compose_config_path)
