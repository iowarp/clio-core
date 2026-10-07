"""
DTSchedule producer-consumer workload (the paper's headline case).

The first ``producer_nodes`` hosts of the pipeline run
``dtschedule_heat_producer`` (MPI 2D heat solver, one rank per hardware
thread, so it holds every core) and write a snapshot per step into CLIO
through the POSIX interposer. The remaining hosts run
``dtschedule_checksum_consumer`` (fewer ranks, lighter compute) which waits
for each step and checksums it. With a RAM tier smaller than the raw
footprint, the open question per chunk is where (if anywhere) to compress:
on the saturated producer, or on the idle consumer when RAM fills.

Writes ``<out>/placement.json`` (every file produced on a producer node and
consumed on a consumer node; clio_dtschedule picks it up as its DAG) and
``<out>/makespan.json`` (``total_ms`` = producer start to consumer done),
which sweep.py reads.

Stats: ``<pkg>.producer_*`` / ``<pkg>.consumer_*`` from the two summary
lines, ``<pkg>.makespan_ms``.
"""
import json
import os
import re
import time

from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo

PROD_RE = re.compile(r'heat_producer (.*)$')
CONS_RE = re.compile(r'checksum_consumer (.*)$')
# prterun's banner when a rank exits non-zero (the job is torn down).
LAUNCH_FAIL_RE = re.compile(r'prterun detected|exited with non-zero')
# Variables the ranks need to reach the CTE through the interposer.
FORWARD_PREFIXES = ('CLIO_', 'CTP_', 'HSHM_')
FORWARD_KEYS = ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'PATH')


class ClioProdcons(Application):
    """MPI producer -> CLIO -> MPI checksum consumer on separate nodes."""

    def _init(self):
        """No state beyond the config."""

    def _configure_menu(self):
        """Menu: launcher, placement, and the workload shape."""
        return [
            {'name': 'mpirun', 'msg': 'mpirun executable', 'type': str,
             'default': 'mpirun'},
            {'name': 'launch_agent', 'msg': 'Open MPI 5 remote daemon '
             '(PRRTE prted); empty = mpirun default', 'type': str,
             'default': ''},
            {'name': 'bin_dir', 'msg': 'Directory holding '
             'dtschedule_heat_producer / dtschedule_checksum_consumer',
             'type': str, 'default': ''},
            {'name': 'out', 'msg': 'Output dir (logs, placement.json, '
             'makespan.json); shared by all nodes', 'type': str,
             'default': '${HOME}/dtschedule-runs/prodcons'},
            {'name': 'run', 'msg': 'Run name (CTE files are '
             '/clio::<run>__step<s>_rank<r>.dat)', 'type': str,
             'default': 'prodcons'},
            {'name': 'producer_nodes', 'msg': 'First N hosts produce, the '
             'rest consume', 'type': int, 'default': 1},
            {'name': 'ppn_producer', 'msg': 'Producer ranks per node (all '
             'hardware threads)', 'type': int, 'default': 40},
            {'name': 'ppn_consumer', 'msg': 'Consumer ranks per node',
             'type': int, 'default': 20},
            {'name': 'steps', 'msg': 'Output steps', 'type': int,
             'default': 10},
            {'name': 'nx', 'msg': 'Columns per producer rank', 'type': int,
             'default': 2048},
            {'name': 'ny', 'msg': 'Rows per producer rank', 'type': int,
             'default': 2048},
            {'name': 'iters_per_step', 'msg': 'Jacobi sweeps per step',
             'type': int, 'default': 200},
            {'name': 'noise', 'msg': 'Relative noise in the output field',
             'type': float, 'default': 0.02},
            {'name': 'passes', 'msg': 'Consumer passes over each file',
             'type': int, 'default': 4},
            {'name': 'net_if', 'msg': 'CIDR for MPI traffic; empty = any',
             'type': str, 'default': ''},
            {'name': 'timeout_s', 'msg': 'Give up waiting for the consumer',
             'type': float, 'default': 3600.0},
        ]

    def _configure(self, **kwargs):
        """Store the config."""
        super()._configure(**kwargs)

    def _out(self):
        """Expanded output directory (created on demand)."""
        # realpath: the head node's $HOME may be a symlink the compute
        # nodes do not have (/home/llogan -> /mnt/common/llogan).
        out = os.path.realpath(os.path.expandvars(self.config['out']))
        os.makedirs(out, exist_ok=True)
        return out

    def _split_hosts(self):
        """Return (producer hosts, consumer hosts)."""
        hosts = list(self.hostfile.hosts)
        n = max(1, min(int(self.config['producer_nodes']), len(hosts) - 1))
        return hosts[:n], hosts[n:]

    def _write_placement(self, producers, consumers):
        """Write the DAG: every rank file produced on a producer host and
        consumed on every consumer host (clio_dtschedule's dag_path)."""
        c = self.config
        nprod = int(c['ppn_producer']) * len(producers)
        per = int(c['ppn_producer'])
        size = int(c['nx']) * int(c['ny']) * 8
        files = {}
        for step in range(int(c['steps'])):
            for r in range(nprod):
                files[f'step{step}_rank{r}.dat'] = {
                    'producer': f'heat_rank{r}',
                    'producer_node': producers[r // per],
                    'consumers': ['checksum'],
                    'consumer_nodes': list(consumers),
                    'size': size}
        doc = {'recipe': 'prodcons', 'nodes': producers + consumers,
               'path_prefix': 'clio::', 'files': files}
        with open(os.path.join(self._out(), 'placement.json'), 'w') as fh:
            json.dump(doc, fh)

    def _write_hostfile(self, name, hosts, ppn):
        """Open MPI hostfile with ``slots=ppn`` per host; returns its path."""
        path = os.path.join(self._out(), name)
        with open(path, 'w') as fh:
            for h in hosts:
                fh.write(f'{h} slots={ppn}\n')
        return path

    def _mpirun(self, hostfile, nranks, ppn, binary, args, log):
        """Single-line mpirun command forwarding the interposer env."""
        c = self.config
        agent = (f'--launch-agent {c["launch_agent"]} '
                 if c.get('launch_agent') else '')
        net = ''
        if c.get('net_if'):
            net = (f'--mca btl_tcp_if_include {c["net_if"]} '
                   f'--mca oob_tcp_if_include {c["net_if"]} ')
        exe = os.path.join(c['bin_dir'], binary) if c['bin_dir'] else binary
        return (f'{c["mpirun"]} {agent}-np {nranks} --hostfile {hostfile} '
                f'--map-by ppr:{ppn}:node:HWTCPUS --bind-to hwthread {net}'
                f'{self._rank_wrapper()} {exe} {args} > {log} 2>&1')

    def _rank_wrapper(self):
        """Inline per-rank wrapper that exports the interposer env.

        ``bash -c '<exports>; exec "$0" "$@"'`` runs on each node, so values
        such as CLIO_MEMFD_DIR=${HOME}/..._${HOSTNAME} expand per node. It is
        inline rather than a script file because a file written on the head
        node just before launch was not yet visible to the compute nodes
        over NFS.

        :return: Wrapper prefix for the rank command.
        """
        exports = []
        for key, val in sorted(self.mod_env.items()):
            if key in FORWARD_KEYS or key.startswith(FORWARD_PREFIXES):
                val = str(val)
                if "'" in val:
                    raise ValueError(f'clio_prodcons: {key} contains a quote')
                exports.append(f'{key}="{val}"')
        return (f"bash -c 'export {' '.join(exports)}; "
                f'exec "$0" "$@"\'')

    def _launch_env(self):
        """Env for the local mpirun: the pipeline env minus LD_PRELOAD (the
        launcher runs on the head node, which has no runtime)."""
        env = dict(self.mod_env)
        env.pop('LD_PRELOAD', None)
        return env

    def start(self):
        """Run consumer (async) and producer (blocking); wait for consumer."""
        c = self.config
        out = self._out()
        producers, consumers = self._split_hosts()
        self._write_placement(producers, consumers)
        nprod = int(c['ppn_producer']) * len(producers)
        ncons = int(c['ppn_consumer']) * len(consumers)
        shape = (f'--run {c["run"]} --steps {int(c["steps"])} '
                 f'--nx {int(c["nx"])} --ny {int(c["ny"])}')
        cons_log = os.path.join(out, 'consumer.log')
        prod_log = os.path.join(out, 'producer.log')
        for path in (cons_log, prod_log):
            if os.path.exists(path):
                os.remove(path)
        cons_cmd = self._mpirun(
            self._write_hostfile('consumer_hosts.txt', consumers,
                                 int(c['ppn_consumer'])),
            ncons, int(c['ppn_consumer']), 'dtschedule_checksum_consumer',
            f'{shape} --producers {nprod} --passes {int(c["passes"])}',
            cons_log)
        prod_cmd = self._mpirun(
            self._write_hostfile('producer_hosts.txt', producers,
                                 int(c['ppn_producer'])),
            nprod, int(c['ppn_producer']), 'dtschedule_heat_producer',
            f'{shape} --iters-per-step {int(c["iters_per_step"])} '
            f'--noise {float(c["noise"])}', prod_log)
        self.log(f'prodcons consumer: {cons_cmd}')
        self.log(f'prodcons producer: {prod_cmd}')
        t0 = time.time()
        Exec(cons_cmd, LocalExecInfo(env=self._launch_env(),
                                     exec_async=True)).run()
        Exec(prod_cmd, LocalExecInfo(env=self._launch_env())).run()
        cons = self._wait_summary(cons_log, CONS_RE, float(c['timeout_s']))
        self._write_makespan(t0, time.time(), prod_log, cons)

    @staticmethod
    def _wait_summary(log, regex, timeout_s):
        """Poll log for regex; return the match, or None on timeout or when
        the MPI launcher reported a failed job."""
        end = time.time() + timeout_s
        while time.time() < end:
            m = ClioProdcons._find(log, regex)
            if m is not None:
                return m
            if ClioProdcons._find(log, LAUNCH_FAIL_RE) is not None:
                return None
            time.sleep(1)
        return None

    @staticmethod
    def _find(log, regex):
        """First match of regex in log, or None."""
        if not os.path.exists(log):
            return None
        with open(log) as fh:
            for line in fh:
                m = regex.search(line)
                if m:
                    return m
        return None

    @staticmethod
    def _fields(match):
        """Parse ``k=v`` pairs of a summary match into a dict."""
        out = {}
        if match is None:
            return out
        for kv in match.group(1).split():
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    out[k] = float(v) if k != 'checksum' else v
                except ValueError:
                    out[k] = v
        return out

    def _write_makespan(self, t0, t1, prod_log, cons_match):
        """Write makespan.json in the format sweep.py reads."""
        prod = self._fields(self._find(prod_log, PROD_RE))
        cons = self._fields(cons_match)
        ok = (cons_match is not None and prod.get('rc') == 0.0 and
              cons.get('rc') == 0.0 and cons.get('bad') == 0.0)
        doc = {'total_ms': round((t1 - t0) * 1e3, 1),
               'status': 'ok' if ok else 'failed',
               'producer': prod, 'consumer': cons}
        with open(os.path.join(self._out(), 'makespan.json'), 'w') as fh:
            json.dump(doc, fh, indent=1)
        self.log(f'prodcons makespan: {doc}')

    def stop(self):
        """Nothing persistent: both MPI jobs exit on their own."""

    def clean(self):
        """Remove the generated logs and json files."""
        for name in ('consumer.log', 'producer.log', 'placement.json',
                     'makespan.json', 'producer_hosts.txt',
                     'consumer_hosts.txt'):
            path = os.path.join(self._out(), name)
            if os.path.exists(path):
                os.remove(path)

    def _get_stat(self, stat_dict):
        """Report makespan and the two summaries."""
        pid = self.pkg_id
        path = os.path.join(self._out(), 'makespan.json')
        if not os.path.exists(path):
            stat_dict[f'{pid}.status'] = 'no_makespan'
            return
        with open(path) as fh:
            doc = json.load(fh)
        stat_dict[f'{pid}.makespan_ms'] = doc.get('total_ms')
        stat_dict[f'{pid}.status'] = doc.get('status')
        for side in ('producer', 'consumer'):
            for k, v in (doc.get(side) or {}).items():
                stat_dict[f'{pid}.{side}_{k}'] = v
