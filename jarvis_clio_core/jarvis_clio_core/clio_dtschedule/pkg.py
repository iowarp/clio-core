"""
DTSchedule policy layer for the CTE stack.

Adds the ``clio_cte_dtschedule`` chimod to the clio_run compose that
``jarvis_clio_core.clio_cte`` writes. dtschedule sits at the TOP of the
interposition chain (DESIGN.md section 1): it must run on the submitter's
node, before the cache's hash-routed write ships the raw bytes.

    client / adapters (CLIO_CTE_POOL=566.0)
      -> dtschedule (566.0)
      -> cache (563.0) | indexer (564.0) | core (512.0)   (next_pool_id)
      -> ... -> core (512.0)

No existing entry is re-pointed. Instead the client-facing entry pool
becomes 566.0: the package exports ``CLIO_CTE_POOL=<pool_id>`` into the
pipeline environment, which ``ClientInit`` (content_transfer_engine.cc)
reads to bind every adapter's CTE client (POSIX LD_PRELOAD, FUSE, HDF5,
ADIOS2) to the dtschedule pool.

Every key of ``context-transfer-engine/dtschedule/docs/DESIGN.md`` section 7
is a package knob; nested ``qos.*`` / ``dag.*`` keys are exposed flat as
``qos_*`` / ``dag_*`` so pipeline-test ``vars:`` can sweep them.

The package only edits compose YAML: the CTE package's ``start`` composes
the whole chain in one ``clio_run compose start``, so ``start``/``stop``
here are no-ops. When no ``clio_cte`` package precedes this one in the
pipeline, the entry is written to this package's own compose file and
composed from ``start`` (the ``clio_compress`` pattern).

``_get_stat`` parses the decision trace CSVs (DESIGN.md section 9) into
counts, ratios and per-library / per-scenario histograms.
"""
import csv
import glob
import os
import pathlib
import re

from jarvis_cd.core.pkg import Service
from jarvis_cd.shell import Exec, PsshExecInfo
import yaml


MOD_NAME = 'clio_cte_dtschedule'
ENTRY_POOL_ENV = 'CLIO_CTE_POOL'
DEFAULT_POOL_ID = 566.0
# Downstream candidates, most preferred first: the old chain top (cache),
# then the indexer, then the core itself.
DOWNSTREAM_MODS = ('clio_cte_cache', 'clio_cte_indexer', 'clio_cte_core')
CTE_PKG_SUFFIX = '.clio_cte'
WFCOMMONS_PKG_SUFFIX = '.clio_wfcommons_dist'
PRODCONS_PKG_SUFFIX = '.clio_prodcons'
DAG_ENV = 'DTSCHEDULE_DAG_PATH'
MODELS_REL = 'context-transfer-engine/dtschedule/models/qtable_v1'

# Decision trace layout: DESIGN.md section 9's 26 columns followed by
# obs_dtime_ms and select_ms (28 columns, header row present). Decision
# (PutBlob) rows fill select_ms and leave obs_dtime_ms empty; decompress
# (GetBlob) rows fill obs_dtime_ms and leave the decision fields empty.
TRACE_COLUMNS = [
    'ts_ms', 'node', 'tag', 'blob', 'size', 'dtype', 'entropy', 'mad', 'd2',
    'producer_cpu', 'consumer_node', 'consumer_cpu', 'owner_node',
    'n_candidates', 'chosen_lib', 'chosen_preset', 'chosen_scenario',
    'chosen_tier', 'pred_ctime_ms', 'pred_dtime_ms', 'pred_ratio',
    'obs_ctime_ms', 'obs_ratio', 'store_ms', 'forced', 'knobs_hash',
    'obs_dtime_ms', 'select_ms',
]
# chosen_lib values of a put stored uncompressed (obs_ratio = 1).
NOT_COMPRESSED = {'', 'none', 'raw', 'skip'}
# Candidate file: <trace_path>.cand.<node>.csv, one row per candidate.
CAND_COLUMNS = ['ts_ms', 'node', 'tag', 'blob', 'lib', 'preset',
                'pred_ctime_ms', 'pred_dtime_ms', 'pred_ratio', 'cost_ms',
                'reason']
CAND_MARKER = '.cand.'

# Device path -> tier name used for the default ``tiers`` table.
TIER_PATTERNS = [
    (r'^ram::', 'ram'), (r'nvme', 'nvme'), (r'ssd', 'ssd'), (r'hdd', 'hdd'),
    (r'^(s3|gcs)://', 's3'),
]


class ClioDtschedule(Service):
    """
    DTSchedule interposer pool for the CTE I/O stack.

    Injects the ``clio_cte_dtschedule`` compose entry at the top of the
    CTE compose chain, exports ``CLIO_CTE_POOL`` so clients enter the
    chain there, and reports trace statistics after a run.
    """

    def _init(self):
        """Set the path of this package's own compose copy."""
        self.compose_config_path = os.path.join(
            self.shared_dir, 'dtschedule_compose.yaml')

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------

    def _configure_menu(self):
        """Describe every DESIGN.md section 7 key as a package knob."""
        return (self._menu_chain() + self._menu_qos() + self._menu_ccm()
                + self._menu_load() + self._menu_workflow()
                + self._menu_decision() + self._menu_trace()
                + self._menu_filesystem())

    @staticmethod
    def _menu_chain():
        """Pool placement knobs (pool ids, names, query)."""
        return [
            {'name': 'pool_name', 'msg': 'Name of the dtschedule pool',
             'type': str, 'default': MOD_NAME},
            {'name': 'pool_id',
             'msg': 'Pool ID of the dtschedule pool; the top of the chain '
                    'and the pool clients enter through (exported as '
                    + ENTRY_POOL_ENV + ')',
             'type': float, 'default': DEFAULT_POOL_ID},
            {'name': 'next_pool_id',
             'msg': 'Pool the dtschedule pool forwards to. Empty = the '
                    'cache pool if the CTE compose has one, else the '
                    'indexer, else the clio_cte_core pool, else 512.0',
             'type': str, 'default': ''},
            {'name': 'pool_query', 'msg': 'Pool query type (local or dynamic)',
             'type': str, 'choices': ['local', 'dynamic'], 'default': 'local'},
        ]

    @staticmethod
    def _menu_qos():
        """QoS knobs (DESIGN.md section 7 ``qos:`` block)."""
        return [
            {'name': 'qos_objective',
             'msg': 'Codec ranking objective (qos.objective)',
             'type': str, 'choices': ['performance', 'ratio'],
             'default': 'performance'},
            {'name': 'qos_max_error',
             'msg': 'Relative error bound for lossy codecs; 0 = lossless '
                    'only (qos.max_error)',
             'type': float, 'default': 1e-3},
            {'name': 'qos_lossy_allowlist',
             'msg': 'Regexes of blob names that may be stored lossily '
                    '(qos.lossy_allowlist)',
             'type': list,
             'default': [r'.*\.(fits|nc|h5|bp|sac|dat)$']},
            {'name': 'qos_compression_preference',
             'msg': 'Candidate codec names; empty = all built-in '
                    '(qos.compression_preference)',
             'type': list, 'default': []},
            {'name': 'qos_resample_error',
             'msg': 'Resample the data statistics when the observed ratio '
                    'misses the prediction by this fraction '
                    '(qos.resample_error)',
             'type': float, 'default': 0.4},
            {'name': 'qos_resample_chance',
             'msg': 'Probability of resampling once triggered '
                    '(qos.resample_chance)',
             'type': float, 'default': 1.0},
            {'name': 'qos_stages',
             'msg': 'Per-stage overrides, list of dicts with "match" '
                    '(regex) plus any qos key, e.g. '
                    '[{"match": ".*/stage2/.*", "max_error": 0}] '
                    '(qos.stages)',
             'type': list, 'default': []},
        ]

    @staticmethod
    def _menu_ccm():
        """Compression-characteristic model knobs."""
        return [
            {'name': 'ccm',
             'msg': 'Codec cost model: qtable | ema | oracle | '
                    'fixed:<lib>[:<preset>]',
             'type': str, 'default': 'qtable'},
            {'name': 'qtable_model_path',
             'msg': 'Directory holding qtable.json + binning_params.json; '
                    'empty = <clio-core>/' + MODELS_REL,
             'type': str, 'default': ''},
            {'name': 'ratio_noise_sigma',
             'msg': 'Multiplicative noise injected into the predicted '
                    'ratio (E9)',
             'type': float, 'default': 0.0},
            {'name': 'min_compress_bytes',
             'msg': 'Blobs smaller than this are stored raw',
             'type': int, 'default': 4096},
        ]

    @staticmethod
    def _menu_load():
        """Load-awareness knobs."""
        return [
            {'name': 'load_aware',
             'msg': 'Weigh producer/consumer CPU load in the placement '
                    'decision',
             'type': bool, 'default': True},
            {'name': 'load_period_ms',
             'msg': 'Node load sampling period (ms)',
             'type': int, 'default': 1000},
            {'name': 'load_cap',
             'msg': 'Cap on the load multiplier applied to predicted '
                    'compute time',
             'type': float, 'default': 4.0},
        ]

    @staticmethod
    def _menu_workflow():
        """Workflow-awareness knobs (DESIGN.md section 7 ``dag:`` block)."""
        return [
            {'name': 'workflow_aware',
             'msg': 'Consumer placement: none | consumer | dag',
             'type': str, 'choices': ['none', 'consumer', 'dag'],
             'default': 'consumer'},
            {'name': 'dag_path',
             'msg': 'DAG placement JSON (placement.json). Empty = the '
                    'pipeline env ' + DAG_ENV + ', else the '
                    'clio_wfcommons_dist package\'s dag_out',
             'type': str, 'default': ''},
            {'name': 'dag_colocate_fanin',
             'msg': 'Place sibling outputs on their common consumer\'s node '
                    '(dag.colocate_fanin)',
             'type': bool, 'default': True},
            {'name': 'dag_replicate_fanout_min',
             'msg': 'Replicate a file consumed by at least this many nodes '
                    '(dag.replicate_fanout_min)',
             'type': int, 'default': 4},
            {'name': 'dag_replicate_max',
             'msg': 'Maximum replicas per file (dag.replicate_max)',
             'type': int, 'default': 8},
        ]

    @staticmethod
    def _menu_decision():
        """Scenario / ordering / tier / network knobs."""
        return [
            {'name': 'force_scenario',
             'msg': 'Force a placement scenario: auto | 1 | 2 | 3 (E5)',
             'type': str, 'choices': ['auto', '1', '2', '3'],
             'default': 'auto'},
            {'name': 'decision_order',
             'msg': 'joint | codec_first | tier_first (E1)',
             'type': str, 'choices': ['joint', 'codec_first', 'tier_first'],
             'default': 'joint'},
            {'name': 'tiers',
             'msg': 'Tier name -> score, must match the CTE device scores; '
                    'empty = derived from the clio_cte package\'s devices',
             'type': dict, 'default': {}},
            {'name': 'net_bw_gbps',
             'msg': 'Inter-node network bandwidth used by the cost model',
             'type': float, 'default': 25.0},
            {'name': 'tier_bw_mbps',
             'msg': 'Tier name -> store bandwidth in MB/s for the cost '
                    'model; empty = derived from the device kind '
                    '(ram 20000, nvme 2000, ssd 500, other 200)',
             'type': dict, 'default': {}},
        ]

    @staticmethod
    def _menu_filesystem():
        """Menu entries for the filesystem chimod composed above dtschedule."""
        return [
            {'name': 'compose_filesystem',
             'msg': ('Compose the clio_cte_filesystem pool (560.0) with '
                     'dtschedule as its downstream so POSIX/FUSE adapter '
                     'writes enter the chain at the top. Without it the '
                     'adapter client creates that pool itself, pointed '
                     'straight at the core, and dtschedule sees no traffic.'),
             'type': bool, 'default': True},
        ]

    @staticmethod
    def _menu_trace():
        """Trace knobs."""
        return [
            {'name': 'trace_path',
             'msg': 'Base path of the per-container decision trace CSV '
                    '(<path>.<container>.csv); "off" disables; empty = '
                    '${HOME}/jarvis-runs/<pipeline>/dtschedule_trace',
             'type': str, 'default': ''},
            {'name': 'trace_candidates',
             'msg': 'Also write every candidate prediction to '
                    '<trace_path>.cand.<node>.csv (E1/E9 replay)',
             'type': bool, 'default': False},
        ]

    # ------------------------------------------------------------------
    # Container — shares clio_runtime's image
    # ------------------------------------------------------------------

    def _build_deploy_phase(self) -> str:
        """No separate container build phase."""
        return None

    # ------------------------------------------------------------------
    # Pipeline discovery helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_pool_id(pool_id) -> str:
        """Coerce a pool id to the ``"<major>.<minor>"`` compose form.

        :param pool_id: float, int or string pool id.
        :return: String such as ``"566.0"``.
        """
        if isinstance(pool_id, str):
            return pool_id if '.' in pool_id else f'{pool_id}.0'
        as_float = float(pool_id)
        if as_float.is_integer():
            return f'{int(as_float)}.0'
        return repr(as_float)

    def _find_pkg_def(self, suffix):
        """Find a package definition in the pipeline by pkg_type suffix.

        :param suffix: Suffix such as ``.clio_cte``.
        :return: The pkg_def dict or None.
        """
        packages = getattr(self.pipeline, 'packages', None) or []
        for pkg_def in packages:
            if str(pkg_def.get('pkg_type', '')).endswith(suffix):
                return pkg_def
        return None

    def _pkg_shared_dir(self, pkg_def):
        """Shared directory of another package in this pipeline.

        :param pkg_def: Package definition dict.
        :return: Absolute path as a string.
        """
        base = self.jarvis.get_pipeline_shared_dir(self.pipeline.name)
        return str(pathlib.Path(base) / pkg_def['pkg_id'])

    def _cte_compose_path(self):
        """Path of the clio_cte package's compose file, or None."""
        cte = self._find_pkg_def(CTE_PKG_SUFFIX)
        if cte is None:
            return None
        return os.path.join(self._pkg_shared_dir(cte), 'cte_compose.yaml')

    @staticmethod
    def _load_compose(path):
        """Load a compose YAML's ``compose`` list.

        :param path: Compose file path.
        :return: List of entries (empty if the file is missing).
        """
        if not path or not os.path.isfile(path):
            return []
        with open(path) as fp:
            data = yaml.safe_load(fp) or {}
        return list(data.get('compose', []) or [])

    def _resolve_next_pool_id(self, chain):
        """Pick the downstream pool: cache, else indexer, else core.

        :param chain: Compose entries of the CTE package.
        :return: Pool id string (512.0 when nothing is known).
        """
        if self.config.get('next_pool_id'):
            return self._format_pool_id(self.config['next_pool_id'])
        by_mod = {entry.get('mod_name'): entry for entry in chain}
        for mod in DOWNSTREAM_MODS:
            if mod in by_mod and 'pool_id' in by_mod[mod]:
                return self._format_pool_id(by_mod[mod]['pool_id'])
        cte = self._find_pkg_def(CTE_PKG_SUFFIX)
        if cte is not None:
            return self._format_pool_id(cte['config'].get('pool_id', 512.0))
        return '512.0'

    def _default_tiers(self):
        """Derive ``tiers`` (name -> score) from the clio_cte devices.

        :return: Dict, empty when no scored devices are found.
        """
        cte = self._find_pkg_def(CTE_PKG_SUFFIX)
        if cte is None:
            return {}
        tiers = {}
        for device in cte['config'].get('devices', []) or []:
            if not isinstance(device, (list, tuple)) or len(device) < 3:
                continue
            name = self._tier_name(str(device[0]))
            while name in tiers:
                name = f'{name}{len(tiers)}'
            tiers[name] = float(device[2])
        return tiers

    def _default_tier_bw(self, tiers):
        """Derive ``tier_bw_mbps`` (name -> MB/s) from the tier names.

        :param tiers: The ``tiers`` map (name -> score) in device order.
        :return: Dict with one entry per tier.
        """
        kinds = (('ram', 20000.0), ('nvme', 2000.0), ('ssd', 500.0))
        bw = {}
        for name in tiers:
            bw[name] = 200.0
            for kind, mbps in kinds:
                if name.startswith(kind):
                    bw[name] = mbps
                    break
        return bw

    @staticmethod
    def _tier_name(path):
        """Map a CTE device path to a tier name.

        :param path: Device path (``ram::x``, ``/mnt/nvme/...``, ...).
        :return: ``ram`` / ``nvme`` / ``ssd`` / ``hdd`` / ``s3`` / ``nfs``.
        """
        for pattern, name in TIER_PATTERNS:
            if re.search(pattern, path, re.IGNORECASE):
                return name
        return 'nfs'

    def _resolve_dag_path(self):
        """DAG path: explicit knob, pipeline env, or the wfcommons package.

        :return: Path string, possibly empty.
        """
        explicit = self.config.get('dag_path', '')
        if explicit:
            return os.path.expandvars(explicit)
        env_path = self.env.get(DAG_ENV, '') or os.environ.get(DAG_ENV, '')
        if env_path:
            return env_path
        pc = self._find_pkg_def(PRODCONS_PKG_SUFFIX)
        if pc is not None:
            out = pc['config'].get('out', '${HOME}/dtschedule-runs/prodcons')
            return os.path.join(os.path.expandvars(out), 'placement.json')
        wf = self._find_pkg_def(WFCOMMONS_PKG_SUFFIX)
        if wf is None:
            return ''
        dag_out = wf['config'].get('dag_out', '')
        if dag_out:
            return os.path.expandvars(dag_out)
        return os.path.join(self._pkg_shared_dir(wf), 'placement.json')

    def _resolve_qtable_dir(self):
        """Q-table model directory: explicit knob or the in-tree model.

        :return: Directory path string.
        """
        explicit = self.config.get('qtable_model_path', '')
        if explicit:
            return os.path.expandvars(explicit)
        repo_root = pathlib.Path(self.pkg_dir).resolve().parents[2]
        return str(repo_root / MODELS_REL)

    def _resolve_trace_path(self):
        """Trace base path: knob, or ${HOME}/jarvis-runs/<pipeline>/...

        :return: Path string; empty string means tracing off.
        """
        knob = self.config.get('trace_path', '')
        if knob.lower() == 'off':
            return ''
        if knob:
            return os.path.expandvars(knob)
        home = os.path.expandvars('${HOME}')
        return os.path.join(home, 'jarvis-runs', self.pipeline.name,
                            'dtschedule_trace')

    # ------------------------------------------------------------------
    # Compose entry
    # ------------------------------------------------------------------

    def _build_entry(self, next_pool_id):
        """Build the DESIGN.md section 7 compose entry.

        :param next_pool_id: Downstream pool id string.
        :return: Dict ready for ``yaml.dump``.
        """
        cfg = self.config
        qos = {
            'objective': cfg['qos_objective'],
            'max_error': float(cfg['qos_max_error']),
            'lossy_allowlist': list(cfg.get('qos_lossy_allowlist') or []),
            'compression_preference':
                list(cfg.get('qos_compression_preference') or []),
            'resample_error': float(cfg['qos_resample_error']),
            'resample_chance': float(cfg['qos_resample_chance']),
        }
        if cfg.get('qos_stages'):
            qos['stages'] = list(cfg['qos_stages'])
        entry = {
            'mod_name': MOD_NAME,
            'pool_name': cfg.get('pool_name', MOD_NAME),
            'pool_query': cfg.get('pool_query', 'local'),
            'pool_id': self._format_pool_id(cfg.get("pool_id", DEFAULT_POOL_ID)),
            'next_pool_id': next_pool_id,
            'qos': qos,
            'ccm': cfg['ccm'],
            'qtable_model_path': cfg['qtable_model_path'],
            'ratio_noise_sigma': float(cfg['ratio_noise_sigma']),
            'load_aware': bool(cfg['load_aware']),
            'load_period_ms': int(cfg['load_period_ms']),
            'load_cap': float(cfg['load_cap']),
            'workflow_aware': cfg['workflow_aware'],
            'dag_path': cfg['dag_path'],
            'dag': {
                'colocate_fanin': bool(cfg['dag_colocate_fanin']),
                'replicate_fanout_min': int(cfg['dag_replicate_fanout_min']),
                'replicate_max': int(cfg['dag_replicate_max']),
            },
            'force_scenario': str(cfg['force_scenario']),
            'decision_order': cfg['decision_order'],
            'tiers': dict(cfg['tiers']),
            'tier_bw_mbps': dict(cfg['tier_bw_mbps'])
                            if cfg.get('tier_bw_mbps')
                            else self._default_tier_bw(cfg['tiers']),
            'net_bw_gbps': float(cfg['net_bw_gbps']),
            'trace_path': cfg['trace_path'],
            'trace_candidates': bool(cfg['trace_candidates']),
            'min_compress_bytes': int(cfg['min_compress_bytes']),
        }
        return entry

    def _splice_into_chain(self, chain, entry):
        """Append the entry on top of the CTE chain.

        Removes a stale dtschedule entry (reconfigure) and any
        ``clio_cte_compressor`` (dtschedule replaces it). No other entry
        is re-pointed: dtschedule is the new chain top and clients reach
        it through ``CLIO_CTE_POOL``. The entry goes last so compose has
        already created the pool its ``next_pool_id`` names.

        :param chain: Compose entries (modified in place and returned).
        :param entry: The dtschedule entry.
        :return: The new chain list.
        """
        kept = []
        for other in chain:
            mod = other.get('mod_name')
            if mod == MOD_NAME:
                continue
            if mod == 'clio_cte_compressor':
                self.log('clio_dtschedule: dropping clio_cte_compressor from '
                         'the CTE compose (dtschedule replaces it)')
                continue
            kept.append(other)
        kept.append(entry)
        return kept

    def _write_compose(self, path, chain, banner):
        """Write a compose list to a YAML file.

        :param path: Destination path.
        :param chain: Compose entries.
        :param banner: Comment line written first.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fp:
            fp.write(f'# {banner}\n\n')
            yaml.dump({'compose': chain}, fp, default_flow_style=False,
                      indent=2, sort_keys=False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        """Resolve defaults and inject the entry into the CTE compose."""
        super()._configure(**kwargs)
        self.compose_config_path = os.path.join(
            self.shared_dir, 'dtschedule_compose.yaml')
        cfg = self.config
        cfg['dag_path'] = self._resolve_dag_path()
        cfg['qtable_model_path'] = self._resolve_qtable_dir()
        cfg['trace_path'] = self._resolve_trace_path()
        if not cfg.get('tiers'):
            cfg['tiers'] = self._default_tiers()
        if cfg['workflow_aware'] == 'dag' and not cfg['dag_path']:
            self.log('clio_dtschedule: workflow_aware=dag but no dag_path '
                     'could be resolved; the chimod will fall back to '
                     'consumer tracking')
        if cfg['trace_path']:
            os.makedirs(os.path.dirname(cfg['trace_path']), exist_ok=True)
            # The chimod appends to <trace_path>.<container>.csv, so a
            # sweep that reconfigures the same pipeline per combination
            # would otherwise fold the previous run's rows into this
            # run's stats. Every configure starts a fresh trace.
            self._remove_trace_files()

        cte_path = self._cte_compose_path()
        chain = self._load_compose(cte_path)
        next_pool_id = self._resolve_next_pool_id(chain)
        cfg['next_pool_id'] = next_pool_id
        entry = self._build_entry(next_pool_id)
        # Clients enter the chain at dtschedule: ClientInit reads this and
        # binds every adapter's CTE client to the named pool.
        self.setenv(ENTRY_POOL_ENV, entry['pool_id'])
        self._write_compose(self.compose_config_path, [entry],
                            'clio_dtschedule clio_run-compose entry (copy)')
        if chain:
            chain = self._splice_into_chain(chain, entry)
            # The POSIX/FUSE adapters write through the filesystem chimod,
            # whose client auto-creates pool 560.0 with the CORE as its
            # downstream when compose did not create it first. Compose it
            # here, pointed at dtschedule, so every adapter write enters
            # the chain at the top. Must come after the dtschedule entry.
            if cfg.get('compose_filesystem', True):
                chain = [e for e in chain
                         if e.get('mod_name') != 'clio_cte_filesystem']
                chain.append({
                    'mod_name': 'clio_cte_filesystem',
                    'pool_name': 'clio_cte_filesystem',
                    'pool_query': 'local',
                    'pool_id': '560.0',
                    'next_pool_id': entry['pool_id'],
                })
            self._write_compose(cte_path, chain,
                                'Content Transfer Engine (CTE) Compose '
                                'Configuration (+ clio_dtschedule)')
            where = f'spliced into {cte_path}'
        else:
            where = (f'standalone at {self.compose_config_path} (no '
                     f'clio_cte compose found; start() composes it)')
        self.log(f"clio_dtschedule: pool {entry['pool_id']} -> "
                 f"{next_pool_id} ({ENTRY_POOL_ENV}={entry['pool_id']}), "
                 f"ccm={cfg['ccm']}, "
                 f"workflow_aware={cfg['workflow_aware']}, "
                 f"force_scenario={cfg['force_scenario']}, "
                 f"trace={cfg['trace_path'] or 'off'}; {where}")

    def _entry_in_cte_compose(self):
        """Whether the CTE compose currently carries the dtschedule entry."""
        return any(e.get('mod_name') == MOD_NAME
                   for e in self._load_compose(self._cte_compose_path()))

    def start(self):
        """No-op when the CTE package composes the chain; else compose."""
        if self._entry_in_cte_compose():
            self.log('clio_dtschedule: composed by the clio_cte package')
            return True
        if not os.path.exists(self.compose_config_path):
            self.log(f'clio_dtschedule: compose file missing: '
                     f'{self.compose_config_path}')
            return False
        Exec(f'clio_run compose start {self.compose_config_path}',
             PsshExecInfo(env=self.mod_env, hostfile=self.jarvis.hostfile,
                          container=self._container_engine,
                          container_image=self.deploy_image_name(),
                          private_dir=self.private_dir,
                          bind_mounts=self.container_mounts)).run()
        return True

    def stop(self):
        """The runtime hosts the pool; nothing to stop."""
        pass

    def kill(self):
        """Nothing to kill."""
        pass

    def _trace_files(self):
        """All trace CSVs of this configuration (decision and candidate).

        :return: Sorted list of paths.
        """
        base = self.config.get('trace_path', '')
        if not base:
            return []
        return sorted(glob.glob(f'{base}.*'))

    def _remove_trace_files(self):
        """Delete every trace CSV under the configured trace_path."""
        for path in self._trace_files():
            try:
                os.remove(path)
            except OSError as err:
                self.log(f'clio_dtschedule: could not remove {path}: {err}')

    def clean(self):
        """Remove the trace files and this package's compose copy."""
        self._remove_trace_files()
        if os.path.exists(self.compose_config_path):
            os.remove(self.compose_config_path)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _as_float(value):
        """Parse a CSV cell as float, or None when empty / malformed."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _decision_files(self):
        """Decision-trace CSVs (candidate files excluded)."""
        return [p for p in self._trace_files() if CAND_MARKER not in p]

    def _candidate_files(self):
        """Candidate-trace CSVs (``<trace_path>.cand.<node>.csv``)."""
        return [p for p in self._trace_files() if CAND_MARKER in p]

    @staticmethod
    def _read_csv(path, columns):
        """Yield one dict per data row of a trace CSV.

        The header row is used when present; a headerless file falls back
        to ``columns``. Short rows are padded so every key exists.

        :param path: CSV path.
        :param columns: Expected column names.
        """
        with open(path, newline='') as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header is None:
                return
            if header[0] != columns[0]:
                fp.seek(0)
                reader = csv.reader(fp)
                header = columns
            for row in reader:
                if not row:
                    continue
                row = row + [''] * (len(header) - len(row))
                yield dict(zip(header, row))

    def _read_trace_rows(self):
        """Yield (record dict, is_get) for every decision-trace row.

        A row is a GetBlob (decompression) row when ``obs_dtime_ms`` is
        non-empty, a PutBlob decision otherwise.
        """
        for path in self._decision_files():
            for record in self._read_csv(path, TRACE_COLUMNS):
                is_get = self._as_float(record.get('obs_dtime_ms')) is not None
                yield record, is_get

    def _candidates_per_decision(self):
        """Mean number of candidate rows per (node, tag, blob) decision.

        :return: Float, or None when no candidate file exists.
        """
        decisions = {}
        for path in self._candidate_files():
            for rec in self._read_csv(path, CAND_COLUMNS):
                key = (rec.get('node'), rec.get('tag'), rec.get('blob'))
                decisions[key] = decisions.get(key, 0) + 1
        if not decisions:
            return None
        return sum(decisions.values()) / len(decisions)

    @staticmethod
    def _count(histogram, key):
        """Increment ``histogram[key]`` when the key is non-empty."""
        if key:
            histogram[key] = histogram.get(key, 0) + 1

    def _summarise_put(self, record, acc):
        """Fold one decision row into the accumulator.

        :param record: Row dict.
        :param acc: Dict of counters / lists built by :meth:`_get_stat`.
        """
        acc['puts'] += 1
        acc['bytes_in'] += int(self._as_float(record.get('size')) or 0)
        lib = (record.get('chosen_lib') or '').strip().lower()
        if lib in NOT_COMPRESSED:
            acc['raw'] += 1
        else:
            acc['compressed'] += 1
            self._count(acc['libs'], lib)
        self._count(acc['scenarios'], record.get('chosen_scenario') or '')
        self._count(acc['tiers'], record.get('chosen_tier') or '')
        for column, bucket in (('obs_ratio', 'ratios'),
                               ('obs_ctime_ms', 'ctimes'),
                               ('select_ms', 'selects')):
            value = self._as_float(record.get(column))
            if value is not None:
                acc[bucket].append(value)

    def _get_stat(self, stat_dict):
        """Summarise the decision and candidate traces.

        :param stat_dict: Flat dict the pipeline test runner serialises.
        """
        pid = self.pkg_id
        cfg = self.config
        for key in ('ccm', 'workflow_aware', 'load_aware', 'force_scenario',
                    'decision_order', 'qos_objective', 'qos_max_error',
                    'ratio_noise_sigma'):
            stat_dict[f'{pid}.{key}'] = cfg.get(key)
        acc = {'puts': 0, 'compressed': 0, 'raw': 0, 'bytes_in': 0,
               'ratios': [], 'ctimes': [], 'selects': [], 'dtimes': [],
               'libs': {}, 'scenarios': {}, 'tiers': {}}
        for record, is_get in self._read_trace_rows():
            if is_get:
                acc['dtimes'].append(self._as_float(record['obs_dtime_ms']))
            else:
                self._summarise_put(record, acc)
        stat_dict[f'{pid}.trace_files'] = len(self._decision_files())
        for key in ('puts', 'compressed', 'raw', 'bytes_in'):
            stat_dict[f'{pid}.{key}'] = acc[key]
        stat_dict[f'{pid}.gets'] = len(acc['dtimes'])
        stat_dict[f'{pid}.mean_ratio'] = self._mean(acc['ratios'])
        stat_dict[f'{pid}.mean_obs_ctime_ms'] = self._mean(acc['ctimes'])
        stat_dict[f'{pid}.mean_obs_dtime_ms'] = self._mean(acc['dtimes'])
        stat_dict[f'{pid}.mean_select_ms'] = self._mean(acc['selects'])
        stat_dict[f'{pid}.candidates_per_decision'] = \
            self._candidates_per_decision()
        for group in ('libs', 'scenarios', 'tiers'):
            prefix = {'libs': 'lib', 'scenarios': 'scenario',
                      'tiers': 'tier'}[group]
            for name, count in sorted(acc[group].items()):
                stat_dict[f'{pid}.{prefix}.{name}'] = count

    @staticmethod
    def _mean(values):
        """Arithmetic mean of a list, or None when empty."""
        if not values:
            return None
        return sum(values) / len(values)
