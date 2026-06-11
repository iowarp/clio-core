"""
Mofka Benchmark Application — runs producer and/or consumer benchmarks
against a running Mofka server and collects performance metrics.

Ported from the archived ``jarvis_iowarp.mofka_bench`` (branch
``claude/mofka-multinode``) into ``jarvis_clio_core``. Deltas from the
archived file, both required to run under the Slurm ``scheduler:`` block
(jarvis ``dev``), which seeds the hostfile with *all* allocated nodes:

  1. Multi-node dispatch keys on ``self.hostfile`` and requires
     ``len(hosts) > 1`` — so a 1-node Slurm allocation (one real, non-
     ``localhost`` host) runs producer/consumer locally, co-located with
     bedrock, exactly like the archived single-node flow.
  2. ``_client_hostfile()`` drops the bedrock head node
     (``MOFKA_SERVER_HOST``, set by mofka_server) from the PSSH target
     set so only the N-1 client nodes run producers/consumers. Idempotent
     — a no-op when the head is already absent (e.g. Pattern-A bash-
     stripped hostfile). See pipelines/mofka/architecture_decisions.md.
"""
from jarvis_cd.core.pkg import Application
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_cd.util.hostfile import Hostfile
import glob
import os
import re
import time as time_mod


class MofkaBench(Application):
    """
    Runs Mofka producer/consumer benchmarks and collects throughput,
    event-rate, and latency statistics.
    """

    _FOLD_SUM = {
        'throughput_mbps',
        'events_per_sec',
        'total_data_mb',
        'events_count',
    }
    _FOLD_MAX = {
        'elapsed_ms',
    }

    def _init(self):
        self.start_time = None

    def _configure_menu(self):
        return [
            {
                'name': 'mode',
                'msg': 'Benchmark mode',
                'type': str,
                'default': 'both',
                'choices': ['producer', 'consumer', 'both'],
            },
            {
                'name': 'num_events',
                'msg': 'Number of events to produce/consume',
                'type': int,
                'default': 1000,
            },
            {
                'name': 'data_size',
                'msg': 'Data payload size in bytes',
                'type': int,
                'default': 1024,
            },
            {
                'name': 'metadata_size',
                'msg': 'Metadata size in bytes',
                'type': int,
                'default': 64,
            },
            {
                'name': 'batch_size',
                'msg': 'Producer/consumer batch size',
                'type': int,
                'default': 16,
            },
            {
                'name': 'num_threads',
                'msg': 'Number of threads',
                'type': int,
                'default': 1,
            },
            {
                'name': 'data_selectivity',
                'msg': 'Consumer data selectivity (0.0-1.0)',
                'type': float,
                'default': 1.0,
            },
            {
                'name': 'use_progress_thread',
                'msg': 'Use Mercury progress thread',
                'type': bool,
                'default': True,
            },
            {
                'name': 'nprocs',
                'msg': 'Number of MPI/SSH processes (1 = single-node local)',
                'type': int,
                'default': 1,
            },
            {
                'name': 'ppn',
                'msg': 'Processes per node',
                'type': int,
                'default': 1,
            },
        ]

    def _configure(self, **kwargs):
        # Resolve group file and topic from upstream mofka_server env
        group_file = self.env.get(
            'MOFKA_GROUP_FILE', '/tmp/mofka-bench/mofka.json'
        )
        topic = self.env.get('MOFKA_TOPIC', 'benchmark_topic')

        self.setenv('MOFKA_GROUP_FILE', group_file)
        self.setenv('MOFKA_TOPIC', topic)
        self.setenv('MOFKA_NUM_EVENTS', str(self.config['num_events']))
        self.setenv('MOFKA_DATA_SIZE', str(self.config['data_size']))
        self.setenv('MOFKA_METADATA_SIZE', str(self.config['metadata_size']))
        self.setenv('MOFKA_BATCH_SIZE', str(self.config['batch_size']))
        self.setenv('MOFKA_NUM_THREADS', str(self.config['num_threads']))
        self.setenv('MOFKA_DATA_SELECTIVITY',
                     str(self.config['data_selectivity']))

        self.log('Mofka benchmark configured')

    def start(self):
        t0 = time_mod.time()

        group_file = self.env.get(
            'MOFKA_GROUP_FILE', '/tmp/mofka-bench/mofka.json'
        )
        topic = self.env.get('MOFKA_TOPIC', 'benchmark_topic')
        scripts_dir = os.path.join(self.pkg_dir, 'scripts')

        # Clear previous output files (and any per-host siblings from a
        # prior multi-node run) so _parse_output can't match stale data.
        for fname in ('producer_output.txt', 'consumer_output.txt'):
            path = os.path.join(self.shared_dir, fname)
            if os.path.exists(path):
                os.remove(path)
            for stale in glob.glob(path + '.*'):
                os.remove(stale)

        progress_flag = ('--use-progress-thread'
                         if self.config['use_progress_thread'] else '')

        # --- Producer ---
        if self.config['mode'] in ('producer', 'both'):
            cmd = (
                f'python3 {scripts_dir}/producer.py'
                f' --group-file {group_file}'
                f' --topic {topic}'
                f' --num-events {self.config["num_events"]}'
                f' --data-size {self.config["data_size"]}'
                f' --metadata-size {self.config["metadata_size"]}'
                f' --batch-size {self.config["batch_size"]}'
                f' --num-threads {self.config["num_threads"]}'
                f' {progress_flag}'
            )
            producer_out = os.path.join(self.shared_dir, 'producer_output.txt')
            self.log('Running producer benchmark')
            self._run_role(cmd, producer_out, 'producer')
            self._check_output_freshness(producer_out, 'producer')

        # --- Consumer ---
        if self.config['mode'] in ('consumer', 'both'):
            cmd = (
                f'python3 {scripts_dir}/consumer.py'
                f' --group-file {group_file}'
                f' --topic {topic}'
                f' --num-events {self.config["num_events"]}'
                f' --data-size {self.config["data_size"]}'
                f' --data-selectivity {self.config["data_selectivity"]}'
                f' --batch-size {self.config["batch_size"]}'
                f' --num-threads {self.config["num_threads"]}'
                f' {progress_flag}'
            )
            consumer_out = os.path.join(
                self.shared_dir, 'consumer_output.txt'
            )
            self.log('Running consumer benchmark')
            self._run_role(cmd, consumer_out, 'consumer')
            self._check_output_freshness(consumer_out, 'consumer')

        self.start_time = time_mod.time() - t0
        self.log(f'Benchmark completed in {self.start_time:.2f}s')

    def _is_multinode(self):
        """Decide whether producer/consumer dispatch via PSSH (multi-node)
        or locally (single-node).

        Returns True only for a genuine multi-host allocation
        (``len(hosts) > 1``). Three cases:
          - default localhost hostfile (no allocation) -> False (local)
          - 1-node Slurm allocation: the scheduler writes one real host
            (e.g. ``ares-comp-14``, not ``localhost``) -> is_local() is
            False, but ``len(hosts) == 1`` -> False (local; bedrock and
            the producer/consumer share the one node, matching the
            archived single-node flow)
          - 2+ node allocation -> True (PSSH to the N-1 client nodes)

        The ``len(hosts) > 1`` gate (vs the archived ``len(hf) > 0``) is
        what keeps single-node Slurm runs on the local branch now that the
        scheduler always materialises a non-localhost hostfile.
        """
        hf = self.hostfile
        if hf is None:
            return False
        try:
            if hf.is_local():
                return False
            return len(hf.hosts) > 1
        except (AttributeError, TypeError):
            return False

    def _client_hostfile(self):
        """Hostfile of client-only nodes for PSSH (bedrock head removed).

        The Slurm scheduler block seeds jarvis's hostfile with every
        allocated node, head included. Bedrock runs on the head node (the
        batch node, published as ``MOFKA_SERVER_HOST`` by mofka_server);
        producers/consumers must run only on the remaining N-1 client
        nodes. Compared on short hostnames so a domain suffix
        (``ares-comp-14.localdomain`` vs ``ares-comp-14``) still matches.

        Idempotent: if ``MOFKA_SERVER_HOST`` is unset or already absent
        from the hostfile (e.g. a Pattern-A bash-pre-stripped client
        hostfile), the original hostfile is returned unchanged. If the
        server is somehow the only host, fall back to the original so
        PSSH still has a target (``_is_multinode`` gates this off anyway).
        """
        hf = self.hostfile
        server_host = (self.env.get('MOFKA_SERVER_HOST', '') or '').split(
            '.')[0]
        if not server_host:
            return hf
        has_ip = len(hf.hosts_ip) == len(hf.hosts)
        keep, keep_ip = [], []
        for i, host in enumerate(hf.hosts):
            if host.split('.')[0] == server_host:
                continue
            keep.append(host)
            if has_ip:
                keep_ip.append(hf.hosts_ip[i])
        if not keep:
            return hf
        return Hostfile(hosts=keep, hosts_ip=keep_ip if keep_ip else None)

    def _run_role(self, cmd, output_path, role):
        """Dispatch a producer/consumer command either locally or across
        client hosts via PsshExecInfo. The dispatch signal is
        ``_is_multinode()`` (a >1-host allocation), not nprocs.

        Multi-node uses a client-only hostfile (``_client_hostfile``) so
        the bedrock head node never runs a producer/consumer. PsshExecInfo
        doesn't accept pipe_stdout, so the redirect is embedded with
        double-quoted bash -c and an escaped \\$(hostname) so each node
        writes to a unique file.

        Local branch captures both stdout and stderr into the output file
        so a Python traceback from producer.py/consumer.py reaches the
        log instead of vanishing. Non-zero exit codes raise RuntimeError
        so jarvis flips the iteration's status to failed.
        """
        if self._is_multinode():
            wrapped = (
                f'bash -c "{cmd} > {output_path}.\\$(hostname) 2>&1"'
            )
            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self._client_hostfile(),
                nprocs=self.config['nprocs'],
                ppn=self.config['ppn'],
            )
            result = Exec(wrapped, exec_info).run()
        else:
            result = Exec(cmd, LocalExecInfo(
                env=self.mod_env,
                pipe_stdout=output_path,
                pipe_stderr=output_path,
            )).run()

        exit_codes = getattr(result, 'exit_code', {}) or {}
        nonzero = {h: c for h, c in exit_codes.items() if c != 0}
        if nonzero:
            self._log_output_tail(output_path)
            raise RuntimeError(
                f'{role} benchmark exited with non-zero code(s): {nonzero}')

    def stop(self):
        pass

    def clean(self):
        for fname in ('producer_output.txt', 'consumer_output.txt'):
            path = os.path.join(self.shared_dir, fname)
            if os.path.exists(path):
                os.remove(path)
            for per_host in glob.glob(path + '.*'):
                os.remove(per_host)

    # ------------------------------------------------------------------
    # Statistics collection
    # ------------------------------------------------------------------

    def _resolve_output_path(self, path):
        """Return the canonical {role}_output.txt if it exists (single-node),
        else any one of the per-host files written in multi-node mode.
        Returns None if neither shape is on disk. Single-node runs always
        hit the first branch; the glob branch handles the multi-node case
        where each host writes its own file."""
        if os.path.exists(path):
            return path
        per_host = sorted(glob.glob(path + '.*'))
        return per_host[0] if per_host else None

    def _log_output_tail(self, path, n_lines=100):
        """Emit the tail of {role}_output.txt into the Jarvis log. Falls back
        to a per-host file (multi-node) if the canonical path is absent."""
        resolved = self._resolve_output_path(path)
        if resolved is None:
            self.log(f'(no output file at {path} or {path}.*)')
            return
        try:
            with open(resolved, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            self.log(f'failed to read {resolved}: {e}')
            return
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        self.log(
            f'--- {os.path.basename(resolved)} tail ({len(tail)} lines) ---')
        for line in tail:
            self.log(line.rstrip())
        self.log(f'--- end {os.path.basename(resolved)} tail ---')

    def _check_output_freshness(self, path, role):
        """Raise if output is missing, empty, or lacks the RESULTS header.

        Both producer.py and consumer.py print a `RESULTS` block on the
        success path; a crash inside the push/flush/pull loop leaves an
        output file with the early config banner but no RESULTS, which is
        the silent-failure mode that produced empty CSV columns on the
        original Ares sweep.
        """
        resolved = self._resolve_output_path(path)
        if resolved is None:
            raise RuntimeError(
                f'{role} benchmark produced no output file at '
                f'{path} (or {path}.*)')
        with open(resolved, 'r') as f:
            content = f.read()
        if not content.strip():
            raise RuntimeError(
                f'{role} benchmark output file is empty: {resolved}')
        content_stripped = re.sub(r'\033\[[0-9;]*m', '', content)
        if not re.search(r'\bRESULTS\b', content_stripped):
            self._log_output_tail(path)
            raise RuntimeError(
                f'{role} benchmark output lacks RESULTS header: {resolved}')

    def _fold_host_stats(self, per_host_stats, stat_dict):
        """Aggregate per-host stat dicts into stat_dict.

        SUM for cumulative metrics (throughput, events, data volume);
        MAX for wall-clock (the run is only over when the slowest
        host finishes). Unknown metrics fall back to mean — a safe
        default if a new metric is added without updating the fold
        rules, though anything load-bearing should be explicitly
        classified in _FOLD_SUM / _FOLD_MAX.
        """
        from collections import defaultdict
        buckets = defaultdict(list)
        for host_stat in per_host_stats:
            for key, value in host_stat.items():
                buckets[key].append(value)
        for key, values in buckets.items():
            metric = key.rsplit('.', 1)[-1]
            if metric in self._FOLD_SUM:
                stat_dict[key] = sum(values)
            elif metric in self._FOLD_MAX:
                stat_dict[key] = max(values)
            else:
                stat_dict[key] = sum(values) / len(values)

    def _get_stat(self, stat_dict):
        """Parse benchmark output files and populate stat_dict.

        Single-node: one canonical {role}_output.txt is parsed via
        _resolve_output_path (same code path as before this change).
        Multi-node: glob all per-host {role}_output.txt.<hostname>
        files, parse each into its own dict, then fold via
        _fold_host_stats.
        """
        for role in ('producer', 'consumer'):
            canonical = os.path.join(
                self.shared_dir, f'{role}_output.txt'
            )
            if self._is_multinode():
                files = sorted(glob.glob(canonical + '.*'))
            else:
                resolved = self._resolve_output_path(canonical)
                files = [resolved] if resolved else []
            if not files:
                continue
            per_host_stats = []
            for fpath in files:
                with open(fpath, 'r') as f:
                    output = f.read()
                host_stat = {}
                self._parse_output(output, role, host_stat)
                if host_stat:
                    per_host_stats.append(host_stat)
            if not per_host_stats:
                continue
            if len(per_host_stats) == 1:
                stat_dict.update(per_host_stats[0])
            else:
                self._fold_host_stats(per_host_stats, stat_dict)

    def _parse_output(self, output, role, stat_dict):
        """Extract metrics from producer or consumer stdout.

        Uses findall and takes the last match to be resilient against
        output files that contain data from multiple appended runs.
        """
        patterns = {
            'throughput_mbps': r'Throughput:\s+([\d.]+)\s+MB/s',
            'events_per_sec': r'Events/sec:\s+([\d.]+)',
            'elapsed_ms': r'Elapsed time:\s+([\d.]+)\s+ms',
            'total_data_mb': r'Total data:\s+([\d.]+)\s+MB',
            'events_count': r'Events (?:produced|consumed):\s+(\d+)',
        }
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, output)
            if matches:
                value = float(matches[-1])
                if metric == 'events_count':
                    value = int(value)
                stat_dict[f'{self.pkg_id}.{role}.{metric}'] = value

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _plot(self, results_dir):
        """
        Generate performance visualisation plots from pipeline test
        results stored in *results_dir*/results.csv.

        Produces four PNG figures:
          1. Throughput (MB/s) vs Data Size
          2. Events/sec vs Data Size
          3. Throughput (MB/s) vs Num Threads
          4. Events/sec vs Num Threads
        """
        import csv
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            self.log('matplotlib not available — skipping plots')
            return

        csv_path = os.path.join(results_dir, 'results.csv')
        if not os.path.exists(csv_path):
            self.log(f'No results.csv found at {csv_path}')
            return

        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            self.log('results.csv is empty')
            return

        # Identify the stat column names (they use the pkg_name prefix)
        # Try to find any column matching *.producer.throughput_mbps
        pkg_prefix = None
        for key in rows[0]:
            if key.endswith('.producer.throughput_mbps'):
                pkg_prefix = key.rsplit('.producer.throughput_mbps', 1)[0]
                break
        if pkg_prefix is None:
            # Fall back: try consumer
            for key in rows[0]:
                if key.endswith('.consumer.throughput_mbps'):
                    pkg_prefix = key.rsplit('.consumer.throughput_mbps', 1)[0]
                    break
        if pkg_prefix is None:
            self.log('Could not identify stat columns in results.csv')
            return

        def _col(role, metric):
            return f'{pkg_prefix}.{role}.{metric}'

        def _safe_float(row, col):
            val = row.get(col, '')
            if val == '':
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        # Detect which sweep variable columns exist
        data_size_col = None
        threads_col = None
        for key in rows[0]:
            if key.endswith('.data_size'):
                data_size_col = key
            if key.endswith('.num_threads'):
                threads_col = key

        # --- Helper: aggregate rows by a sweep variable ---
        def _aggregate(rows, sweep_col, role, metric_col):
            """Return sorted (x_values, y_means) averaging over repeats."""
            from collections import defaultdict
            buckets = defaultdict(list)
            for row in rows:
                x = _safe_float(row, sweep_col)
                y = _safe_float(row, _col(role, metric_col))
                if x is not None and y is not None:
                    buckets[x].append(y)
            xs = sorted(buckets.keys())
            ys = [sum(buckets[x]) / len(buckets[x]) for x in xs]
            return xs, ys

        # --- Figure 1: Throughput vs Data Size ---
        if data_size_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            for role, colour, label in [
                ('producer', '#2196F3', 'Producer'),
                ('consumer', '#FF9800', 'Consumer'),
            ]:
                xs, ys = _aggregate(rows, data_size_col, role,
                                    'throughput_mbps')
                if xs:
                    ax.bar(
                        [str(int(x)) for x in xs], ys,
                        alpha=0.7, color=colour, label=label,
                        width=0.35,
                    )
            ax.set_xlabel('Data Size (bytes)')
            ax.set_ylabel('Throughput (MB/s)')
            ax.set_title('Mofka Throughput vs Data Size')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir,
                                     'throughput_vs_data_size.png'),
                        dpi=150)
            plt.close(fig)

        # --- Figure 2: Events/sec vs Data Size ---
        if data_size_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            for role, colour, label in [
                ('producer', '#2196F3', 'Producer'),
                ('consumer', '#FF9800', 'Consumer'),
            ]:
                xs, ys = _aggregate(rows, data_size_col, role,
                                    'events_per_sec')
                if xs:
                    ax.bar(
                        [str(int(x)) for x in xs], ys,
                        alpha=0.7, color=colour, label=label,
                        width=0.35,
                    )
            ax.set_xlabel('Data Size (bytes)')
            ax.set_ylabel('Events / sec')
            ax.set_title('Mofka Event Rate vs Data Size')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir,
                                     'events_vs_data_size.png'),
                        dpi=150)
            plt.close(fig)

        # --- Figure 3: Throughput vs Num Threads ---
        if threads_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            for role, marker, label in [
                ('producer', 'o', 'Producer'),
                ('consumer', 's', 'Consumer'),
            ]:
                xs, ys = _aggregate(rows, threads_col, role,
                                    'throughput_mbps')
                if xs:
                    ax.plot(xs, ys, marker=marker, label=label, linewidth=2)
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Throughput (MB/s)')
            ax.set_title('Mofka Throughput vs Thread Count')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir,
                                     'throughput_vs_threads.png'),
                        dpi=150)
            plt.close(fig)

        # --- Figure 4: Events/sec vs Num Threads ---
        if threads_col:
            fig, ax = plt.subplots(figsize=(8, 5))
            for role, marker, label in [
                ('producer', 'o', 'Producer'),
                ('consumer', 's', 'Consumer'),
            ]:
                xs, ys = _aggregate(rows, threads_col, role,
                                    'events_per_sec')
                if xs:
                    ax.plot(xs, ys, marker=marker, label=label, linewidth=2)
            ax.set_xlabel('Number of Threads')
            ax.set_ylabel('Events / sec')
            ax.set_title('Mofka Event Rate vs Thread Count')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(results_dir,
                                     'events_vs_threads.png'),
                        dpi=150)
            plt.close(fig)

        self.log(f'Plots saved to {results_dir}')
