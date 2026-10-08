"""
Distributed WfCommons/WfBench workflows for DTSchedule evaluations.

Mirrors ``builtin.wfcommons`` but, instead of running the whole DAG on
one node with the sequential ``BashTranslator``, uses the DTSchedule
``dt_translator.py`` to

* give every WfBench file a recipe-typed extension and fill it with
  deterministic, compressible data (``dt_datagen``) instead of
  ``os.urandom``;
* assign tasks to nodes level by level (``round_robin``, ``stage_sets``
  or ``single``) so producer and consumer tasks really sit on different
  nodes;
* emit ``placement.json`` -- the DAG spec the ``dtschedule`` chimod reads
  as ``dag_path`` -- and per-level, per-node scripts that ``run_dist.py``
  drives over ssh.

``_configure`` generates and translates the workflow on the head node
(the output directory lives on the shared filesystem); ``start`` runs
``run_dist.py`` with the interceptor environment (LD_PRELOAD of the CTE
POSIX adapter when ``interceptors: [cte_posix]`` and ``clio_prefix``
are set); ``_get_stat`` reports the makespan from ``makespan.json``.

``api: true`` runs the same workflow over the CTE API instead of WfBench +
POSIX: the translator writes only ``placement.json`` (``--api``: instance
runtimes per task, optionally instance-proportional file sizes), and
``start`` launches ``dtschedule_wfrun`` (tools/wfrun) with one mpirun over
every host, ``ppn`` ranks each, exactly like ``clio_prodcons`` (whose
launch helpers are reused). Files are CTE tags put/get through the
dtschedule pool, so clio-fs's home-node routing does not override
dtschedule's placement. ``dtschedule_wfrun`` writes ``makespan.json``
(same keys plus read/write/stage totals) and ``tasks.csv``.
"""
import json
import os
import pathlib
import shutil

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_cd.shell.process import Mkdir, Rm

from jarvis_clio_core.clio_prodcons.pkg import ClioProdcons


RECIPES = [
    'montage', 'genome', 'cycles', 'blast', 'bwa',
    'srasearch', 'epigenomics', 'seismology', 'soykb', 'rnaseq',
]
PLACEMENTS = ['round_robin', 'stage_sets', 'single']
DATA_CLASSES = ['auto', 'float_field', 'text_seq', 'text_table',
                'mixed_binary']
TOOLS_REL = 'context-transfer-engine/dtschedule/tools/wfcommons'
SIZE_MODES = ['uniform', 'instance']
WFRUN_BIN = 'dtschedule_wfrun'
WFRUN_LOG = 'wfrun.log'


class ClioWfcommonsDist(Application):
    """
    Distributed WfBench workflow runner for DTSchedule.

    Generates a WfFormat workflow from a recipe, translates it with the
    DTSchedule distributed translator, and executes it level by level
    across the pipeline's hostfile (or, with ``api``, with
    ``dtschedule_wfrun`` over the CTE API).
    """

    # CTE-API mode launches exactly like clio_prodcons: one mpirun with
    # launch_agent prted, net_if, and the CLIO_/CTP_ env exported inline.
    _mpirun = ClioProdcons._mpirun
    _rank_wrapper = ClioProdcons._rank_wrapper
    _launch_env = ClioProdcons._launch_env

    def _init(self):
        """Nothing to initialise before configuration."""
        pass

    def _configure_menu(self):
        """Describe the package configuration knobs."""
        return [
            {'name': 'recipe', 'msg': 'WfCommons recipe to generate',
             'type': str, 'choices': RECIPES, 'default': 'montage'},
            {'name': 'num_tasks',
             'msg': 'Number of tasks in the generated workflow',
             'type': int, 'default': 60},
            {'name': 'data_footprint',
             'msg': 'Total workflow data footprint (e.g. 30G); 0 = recipe '
                    'defaults', 'type': str, 'default': '0'},
            {'name': 'cpu_work',
             'msg': 'CPU work units per wfbench task. MUST be > 0; wfbench '
                    'gates its I/O on cpu-benchmark progress, so 0 silently '
                    'disables all reads/writes. 1 = minimal CPU, maximal I/O.',
             'type': int, 'default': 1},
            {'name': 'percent_cpu',
             'msg': 'CPU vs memory thread split (cpu_threads = 10*x, '
                    'mem_threads = 10 - cpu_threads; mem threads need '
                    'stress-ng). 1.0 = no stress-ng.',
             'type': float, 'default': 1.0},
            {'name': 'drop_page_cache',
             'msg': 'fsync + posix_fadvise(DONTNEED) after every read/write '
                    '(sets WFBENCH_DROP_CACHE=1) so NFS vs CTE comparisons '
                    'are not biased by the kernel page cache',
             'type': bool, 'default': False},
            {'name': 'clio_prefix',
             'msg': 'Prefix every task path with "clio::" so the CTE POSIX '
                    'interposer (LD_PRELOAD via interceptors) catches it',
             'type': bool, 'default': False},
            {'name': 'placement',
             'msg': 'Task-to-node policy: round_robin (spread each level '
                    'over all nodes), stage_sets (level i -> node subset '
                    'i mod stage_sets), single (first node only)',
             'type': str, 'choices': PLACEMENTS, 'default': 'stage_sets'},
            {'name': 'stage_sets',
             'msg': 'Number of disjoint node subsets for placement=stage_sets',
             'type': int, 'default': 2},
            {'name': 'data_class',
             'msg': 'Generated data class for every file (auto = by '
                    'extension: .fits/.stf float_field, .fastq text_seq, '
                    '.txt/.vcf text_table, .gz mixed_binary)',
             'type': str, 'choices': DATA_CLASSES, 'default': 'auto'},
            {'name': 'data_noise',
             'msg': 'Noise amplitude [0,1] added to float_field data '
                    '(DT_DATA_NOISE); lower = more compressible',
             'type': float, 'default': 0.05},
            {'name': 'wf_python',
             'msg': 'Python interpreter with wfcommons==1.2 + numpy '
                    'installed (created on demand if missing)',
             'type': str,
             'default': '${HOME}/venv-dtschedule/bin/python'},
            {'name': 'tools_dir',
             'msg': 'Directory holding dt_translator.py (default: derived '
                    'from this package\'s location in the clio-core tree)',
             'type': str, 'default': ''},
            {'name': 'out',
             'msg': 'Output directory on the shared filesystem (receives '
                    'bench/, data/, levels/, logs/, placement.json, '
                    'makespan.json)',
             'type': str, 'default': '${HOME}/dtschedule-runs/wfcommons'},
            {'name': 'dag_out',
             'msg': 'Where to copy placement.json for the dtschedule chimod '
                    '(dag_path); default: <pkg shared_dir>/placement.json',
             'type': str, 'default': ''},
            {'name': 'ssh_cmd',
             'msg': 'ssh command prefix run_dist.py uses to reach nodes',
             'type': str, 'default': 'ssh'},
        ] + self._api_menu()

    @staticmethod
    def _api_menu():
        """Knobs of the CTE-API mode (``api: true``; ignored otherwise).

        :return: Menu entries.
        """
        return [
            {'name': 'api', 'msg': 'Run with dtschedule_wfrun over the CTE '
             'API (dtschedule pool) instead of WfBench over POSIX/clio-fs',
             'type': bool, 'default': False},
            {'name': 'seed', 'msg': 'Seed for the WfChef instance generator '
             '(-1 = unseeded: a new random instance at every configure)',
             'type': int, 'default': -1},
            {'name': 'size_mode', 'msg': 'API: file sizes uniform (WfBench) '
             'or instance (proportional to the WfChef instance, same total)',
             'type': str, 'choices': SIZE_MODES, 'default': 'instance'},
            {'name': 'min_file_kb', 'msg': 'API, size_mode instance: '
             'smallest file (KiB)', 'type': int, 'default': 1024},
            {'name': 'max_file_mb', 'msg': 'API: largest file (MiB; also '
             'caps wfrun); 0 = none', 'type': int, 'default': 256},
            {'name': 'size_scale', 'msg': 'API: factor on every file size '
             'at run time', 'type': float, 'default': 1.0},
            {'name': 'runtime_scale', 'msg': 'API: compute seconds per '
             'instance runtime second (CPU-bound stencil, not sleep)',
             'type': float, 'default': 0.01},
            {'name': 'ppn', 'msg': 'API: wfrun ranks per node (a node\'s '
             'tasks are dealt round-robin over them)', 'type': int,
             'default': 8},
            {'name': 'write_pending', 'msg': 'API: output files a rank leaves '
             'in flight (0 = each write returns once stored)', 'type': int,
             'default': 1},
            {'name': 'payload', 'msg': 'API: directory of real data files '
             'outputs are cut from (on every node); empty = generated by '
             'data class', 'type': str, 'default': ''},
            {'name': 'mpirun', 'msg': 'API: mpirun executable', 'type': str,
             'default': 'mpirun'},
            {'name': 'launch_agent', 'msg': 'API: Open MPI 5 remote daemon '
             '(PRRTE prted); empty = mpirun default', 'type': str,
             'default': ''},
            {'name': 'bin_dir', 'msg': 'API: directory holding '
             'dtschedule_wfrun', 'type': str, 'default': ''},
            {'name': 'net_if', 'msg': 'API: CIDR for MPI traffic; empty = '
             'any', 'type': str, 'default': ''},
            {'name': 'timeout_s', 'msg': 'API: give up waiting for an input '
             'file (s)', 'type': float, 'default': 3600.0},
            {'name': 'input_mem_mb', 'msg': 'API: input bytes a rank holds '
             'at once; wider fan-ins are read and computed in batches',
             'type': int, 'default': 1024},
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _tools_dir(self):
        """Resolve the directory holding the DTSchedule wfcommons tools.

        :return: Absolute path as a string.
        """
        if self.config.get('tools_dir'):
            return os.path.expandvars(self.config['tools_dir'])
        repo_root = pathlib.Path(self.pkg_dir).resolve().parents[2]
        return str(repo_root / TOOLS_REL)

    def _dag_out(self):
        """Resolve where placement.json is copied for the chimod.

        :return: Absolute path as a string.
        """
        if self.config.get('dag_out'):
            return os.path.expandvars(self.config['dag_out'])
        return f'{self.shared_dir}/placement.json'

    def _write_hostfile(self):
        """Write a plain one-node-per-line hostfile for the translator.

        :return: Path of the written hostfile.
        """
        path = f'{self.shared_dir}/hostfile.txt'
        hosts = list(self.hostfile.hosts) if self.hostfile else ['localhost']
        with open(path, 'w') as fp:
            fp.write('\n'.join(hosts) + '\n')
        return path

    def _translator_cmd(self, hostfile_path):
        """Build the dt_translator.py command line.

        :param hostfile_path: Hostfile written by :meth:`_write_hostfile`.
        :return: Command string.
        """
        cfg = self.config
        parts = [
            cfg['wf_python'], f"{self._tools_dir()}/dt_translator.py",
            f"--recipe {cfg['recipe']}", f"--num-tasks {cfg['num_tasks']}",
            f"--data-footprint {cfg['data_footprint']}",
            f"--hostfile {hostfile_path}",
            f"--placement {cfg['placement']}",
            f"--stage-sets {cfg['stage_sets']}",
            f"--out {cfg['out']}", f"--cpu-work {cfg['cpu_work']}",
            f"--percent-cpu {cfg['percent_cpu']}",
            f"--data-class {cfg['data_class']}",
            f"--data-noise {cfg['data_noise']}",
            f"--python {cfg['wf_python']}",
        ]
        if cfg.get('clio_prefix'):
            parts.append('--clio-prefix')
        if int(cfg.get('seed', -1)) >= 0:
            # Set iteration order (string hashing) also shapes the instance.
            parts.insert(0, f"PYTHONHASHSEED={int(cfg['seed'])}")
            parts.append(f"--seed {int(cfg['seed'])}")
        if cfg.get('api'):
            parts += ['--api', f"--size-mode {cfg.get('size_mode', 'instance')}",
                      f"--min-file-kb {int(cfg.get('min_file_kb', 1024))}",
                      f"--max-file-mb {int(cfg.get('max_file_mb', 0))}"]
        return ' '.join(parts)

    def _wfrun_args(self):
        """dtschedule_wfrun arguments for the translated workflow.

        :return: Argument string.
        """
        cfg = self.config
        out = os.path.realpath(os.path.expandvars(cfg['out']))
        args = (f"--placement {out}/placement.json --out {out} "
                f"--runtime-scale {float(cfg['runtime_scale'])} "
                f"--size-scale {float(cfg['size_scale'])} "
                f"--max-file-mb {int(cfg['max_file_mb'])} "
                f"--write-pending {int(cfg['write_pending'])} "
                f"--noise {float(cfg['data_noise'])} "
                f"--timeout-s {float(cfg['timeout_s'])} "
                f"--input-mem-mb {int(cfg['input_mem_mb'])}")
        if cfg.get('payload'):
            args += f" --payload {cfg['payload']}"
        return args

    def _start_api(self):
        """Launch dtschedule_wfrun on every host (blocking) and make sure a
        makespan.json exists afterwards (a failed one if the job died)."""
        cfg = self.config
        out = os.path.realpath(os.path.expandvars(cfg['out']))
        hosts = list(self.hostfile.hosts)
        ppn = int(cfg['ppn'])
        hostfile = os.path.join(out, 'wfrun_hosts.txt')
        with open(hostfile, 'w') as fh:
            fh.writelines(f'{h} slots={ppn}\n' for h in hosts)
        log = os.path.join(out, WFRUN_LOG)
        for name in (WFRUN_LOG, 'makespan.json', 'tasks.csv'):
            if os.path.exists(os.path.join(out, name)):
                os.remove(os.path.join(out, name))
        cmd = self._mpirun(hostfile, ppn * len(hosts), ppn, WFRUN_BIN,
                           self._wfrun_args(), log)
        self.log(f'wfrun: {cmd}')
        Exec(cmd, LocalExecInfo(env=self._launch_env())).run()
        path = os.path.join(out, 'makespan.json')
        if not os.path.isfile(path):
            with open(path, 'w') as fp:
                json.dump({'total_ms': None, 'status': 'failed',
                           'levels': []}, fp)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _configure(self, **kwargs):
        """Generate + translate the workflow on the head node."""
        super()._configure(**kwargs)
        cfg = self.config
        if cfg['recipe'] not in RECIPES:
            raise ValueError(f"recipe '{cfg['recipe']}' not in {RECIPES}")
        if cfg['placement'] not in PLACEMENTS:
            raise ValueError(
                f"placement '{cfg['placement']}' not in {PLACEMENTS}")
        if int(cfg.get('cpu_work', 0)) <= 0:
            self.log('cpu_work was 0; coercing to 1 so wfbench performs I/O')
            cfg['cpu_work'] = 1
        cfg['wf_python'] = os.path.expandvars(cfg['wf_python'])
        cfg['out'] = os.path.expandvars(cfg['out'])
        if cfg.get('drop_page_cache'):
            self.setenv('WFBENCH_DROP_CACHE', '1')
        self.setenv('DT_DATA_NOISE', str(cfg['data_noise']))
        if cfg['data_class'] != 'auto':
            self.setenv('DT_DATA_CLASS', cfg['data_class'])

        tools = self._tools_dir()
        if not os.path.isfile(f'{tools}/dt_translator.py'):
            raise FileNotFoundError(
                f'dt_translator.py not found under {tools}; set tools_dir')
        Exec(f"bash {self.pkg_dir}/ensure_venv.sh {cfg['wf_python']}",
             LocalExecInfo(env=self.env)).run()
        Mkdir(cfg['out'], PsshExecInfo(hostfile=self.hostfile,
                                       env=self.env)).run()
        hostfile_path = self._write_hostfile()
        Exec(self._translator_cmd(hostfile_path),
             LocalExecInfo(env=self.env)).run()

        placement = f"{cfg['out']}/placement.json"
        if not os.path.isfile(placement):
            raise RuntimeError(f'translator did not produce {placement}')
        dag_out = self._dag_out()
        if os.path.abspath(dag_out) != os.path.abspath(placement):
            os.makedirs(os.path.dirname(dag_out), exist_ok=True)
            shutil.copy(placement, dag_out)
        self.setenv('DTSCHEDULE_DAG_PATH', dag_out)
        self.log(f'placement written to {dag_out}')

    def start(self):
        """Run the workflow level by level with the interceptor env (or
        with dtschedule_wfrun over the CTE API when ``api`` is set)."""
        cfg = self.config
        if cfg.get('api'):
            self._start_api()
            return
        cmd = (f"{cfg['wf_python']} {cfg['out']}/run_dist.py "
               f"--out {cfg['out']} --ssh-cmd '{cfg['ssh_cmd']}'")
        Exec(cmd, LocalExecInfo(env=self.mod_env)).run()

    def stop(self):
        """Kill straggling wfbench_dt / cpu-benchmark processes on all nodes."""
        Exec(f'bash {self.pkg_dir}/kill_stragglers.sh',
             PsshExecInfo(hostfile=self.hostfile, env=self.env)).run()

    def clean(self):
        """Remove the output directory and the copied placement.json."""
        out = self.config.get('out')
        if out:
            Rm(os.path.expandvars(out),
               PsshExecInfo(hostfile=self.hostfile, env=self.env)).run()
        dag_out = self._dag_out()
        if os.path.isfile(dag_out):
            os.remove(dag_out)

    def _get_stat(self, stat_dict):
        """Report configuration and the measured makespan (ms)."""
        cfg = self.config
        pid = self.pkg_id
        stat_dict[f'{pid}.recipe'] = cfg['recipe']
        stat_dict[f'{pid}.num_tasks'] = cfg['num_tasks']
        stat_dict[f'{pid}.placement'] = cfg['placement']
        stat_dict[f'{pid}.stage_sets'] = cfg['stage_sets']
        stat_dict[f'{pid}.runtime'] = self.runtime
        path = f"{os.path.expandvars(cfg['out'])}/makespan.json"
        if not os.path.isfile(path):
            stat_dict[f'{pid}.status'] = 'no makespan.json'
            return
        with open(path) as fp:
            makespan = json.load(fp)
        stat_dict[f'{pid}.status'] = makespan.get('status')
        stat_dict[f'{pid}.makespan_ms'] = makespan.get('total_ms')
        stat_dict[f'{pid}.levels'] = len(makespan.get('levels', []))
        for level in makespan.get('levels', []):
            key = f"{pid}.level{level['level']}_ms"
            stat_dict[key] = level['wall_ms']
        for key in ('stage_ms', 'stage_gb', 'read_gb', 'write_gb', 'wait_s',
                    'read_s', 'compute_s', 'gen_s', 'write_s', 'errors'):
            if key in makespan:
                stat_dict[f'{pid}.{key}'] = makespan[key]
