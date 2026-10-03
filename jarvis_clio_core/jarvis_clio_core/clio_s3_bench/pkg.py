from jarvis_cd.shell import Exec, PsshExecInfo, LocalExecInfo
import os
import sys

# The repo root is on sys.path when jarvis imports this module, so the shared
# base and parser resolve as namespace-package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_clio_core.bench_base import S3BenchBase  # noqa: E402
from jarvis_clio_core.bench_parse import parse_time_v  # noqa: E402


class ClioS3Bench(S3BenchBase):
    """
    CLIO S3 Benchmark Application -- both directions, selected by `mode`.

    mode: read
        Drives `clio_s3_read_bench`, which reads N objects from S3 through the
        CAE assimilator (ParseOmni -> S3FileAssimilator -> fork+exec
        cae_s3_tool get -> CTE PutBlob) with a sliding window of K in-flight
        transfers.

    mode: write
        Drives `clio_s3_write_bench`, which writes N objects into CTE with a
        sliding window of K in-flight AsyncPutBlob calls. CTE places them on an
        S3-backed bdev tier whose WriteBlocks issues signed PUTs from the
        runtime daemon.

    Pairs with `zarr_s3_bench` in the same mode (baseline) and, on the write
    side, `s3_raw_put_bench` (wire floor). All emit the same two results blocks
    so one parser serves every stack.

    WHY ONE PACKAGE. Every other direction-carrying benchmark in the jarvis
    ecosystem selects direction from config rather than from the package name:
    builtin.ior and builtin.fio take `read`/`write` booleans, clio_cte_bench
    and clio_redis_bench take `test_case: Put|Get|PutGet`, mofka_bench takes
    `mode: producer|consumer|both`. An enum rather than ior's two booleans
    because the two directions are two distinct binaries here -- `read: true,
    write: true` would be a configuration neither can satisfy.

    IMPORTANT -- environment ownership: neither direction signs its own
    requests in THIS process. Reading, the runtime forks cae_s3_tool; writing,
    clio_run itself signs. Both are launched by the clio_runtime package with
    an environment jarvis builds from `EnvironmentManager.COMMON_ENV_VARS`, a
    fixed list carrying no AWS_* entry -- so exporting credentials in the
    pipeline's pre_cmds reaches jarvis and this process but never the daemon.
    The pipeline must name the AWS variables in clio_runtime's `forward_env`.
    """

    #: One executable per mode. Both are installed to bin/ by the same build,
    #: but under different cmake gates (+cae +s3_cae for read, +s3_bdev for write),
    #: which is why the pipelines gate on `command -v` for the one they need.
    EXECUTABLES = {
        'read': 'clio_s3_read_bench',
        'write': 'clio_s3_write_bench',
    }

    #: Emitted as --label, and the namespace results.csv columns land under.
    #: Keeping these capitalized matches clio_bench::PrintResults; bench_parse
    #: lowercases them into `clio_s3.read.*` / `clio_s3.write.*`.
    LABELS = {'read': 'Read', 'write': 'Write'}

    def _init(self):
        """Initialize instance state."""
        self.output_path = None
        self.rss_path = None
        self.objects_path = None
        self.purged_count = None

    def _configure_menu(self):
        """
        Configure the application menu.

        Returns:
            List[Dict]: Configuration menu options for the benchmark.
        """
        return [
            {
                'name': 'mode',
                'msg': 'Transfer direction',
                'type': str,
                'default': 'read',
                'choices': ['read', 'write'],
                'help': 'Selects the driver binary, the results label, and '
                        'which mode-specific options below apply. Set it in '
                        'the pipeline YAML -- there is no safe default for a '
                        'sweep, and read against a write pipeline silently '
                        'measures the wrong path.'
            },
            {
                'name': 'bucket',
                'msg': 'S3 bucket',
                'type': str,
                'default': '',
                'help': 'read: required, holds the staged objects (see '
                        'scripts/stage_s3_read_bench_data.py). write: enables '
                        'the objects_measured column and the pre-run purge; '
                        'empty disables both and the run still succeeds.'
            },
            {
                'name': 'key_prefix',
                'msg': 'Key prefix for the object set',
                'type': str,
                'default': '',
                'help': 'read: required, e.g. clio-s3-read-bench/raw/33554432. '
                        'write: the clio_cte device path WITHOUT the '
                        's3://bucket/ part and without the _node<N> suffix CTE '
                        'appends -- listing that stem covers every node.'
            },
            {
                'name': 'object_size',
                'msg': 'Bytes per object',
                'type': str,
                'default': '32m',
                'help': 'Suffixes k/m/g. read: must match the staged set '
                        '(512k, 4m, 32m, 256m). write: one object becomes one '
                        'S3 object -- the bdev issues one PutObject per '
                        'allocator block and an unfragmented request gets a '
                        'single block.'
            },
            {
                'name': 'num_objects',
                'msg': 'Number of objects to transfer',
                'type': int,
                'default': 64,
                'help': 'num_objects * object_size is the total bytes moved.'
            },
            {
                'name': 'concurrency',
                'msg': 'In-flight transfers (K)',
                'type': int,
                'default': 8,
                'help': 'Effective concurrency is capped by the runtime worker '
                        'count in BOTH directions: reading, the assimilator '
                        'blocks a worker on waitpid for the whole GET; '
                        'writing, the S3 PUT blocks one for its whole '
                        'duration. Sweep clio_runtime.num_threads with this.'
            },
            {
                'name': 'worker_threads',
                'msg': 'Runtime worker threads, for the fairness report',
                'type': int,
                'default': 0,
                'help': 'Reported verbatim into results.csv. Keep in sync with '
                        'clio_runtime.num_threads; 0 means unknown.'
            },
            {
                'name': 'tag_prefix',
                'msg': 'CTE tag prefix',
                'type': str,
                'default': 's3b',
                'help': 'read: one tag PER OBJECT -- the assimilator restarts '
                        'blob naming at chunk_0 per transfer, so a shared tag '
                        'would overwrite earlier objects. write: all blobs '
                        'share one tag, since blob names are explicit there.'
            },
            {
                'name': 'verify',
                'msg': 'Verify after the timed run',
                'type': bool,
                'default': False,
                'help': 'read: re-reads tag sizes from CTE (2 round trips per '
                        'object). write: reads every blob back and compares '
                        'bytes, which is what proves data round-tripped '
                        'through S3. Enable for one-off validation, not sweeps.'
            },
            {
                'name': 'aws_region',
                'msg': 'AWS region',
                'type': str,
                'default': 'us-east-1',
                'help': 'Sets AWS_DEFAULT_REGION. SigV4 is region-scoped: a '
                        "mismatch against the bucket's real region comes back "
                        'as an HTTP 301, not a 403.'
            },
            {
                'name': 'aws_profile',
                'msg': 'AWS profile name from ~/.aws/credentials',
                'type': str,
                'default': '',
                'help': 'read: resolved by cae_s3_tool through the AWS SDK '
                        'chain. write: recorded for provenance only -- the '
                        'Poco signer reads env vars, so pre_cmds must resolve '
                        'the profile to raw keys. Never put secrets in YAML.'
            },

            # --- read only ---
            {
                'name': 'preflight',
                'msg': '[read] Do a 1-byte ranged GET before timing starts',
                'type': bool,
                'default': True,
                'help': 'Fails fast on bad credentials/bucket/prefix instead '
                        'of issuing thousands of doomed GETs.'
            },
            {
                'name': 'tmpdir',
                'msg': '[read] Node-local staging dir for downloaded objects',
                'type': str,
                'default': '/tmp/clio_s3_bench',
                'help': 'Peak usage is concurrency * object_size (32 x 256 MiB '
                        '= 8 GiB), so this must be on local disk with headroom.'
            },
            {
                'name': 's3_tool',
                'msg': '[read] Path to the cae_s3_tool helper',
                'type': str,
                'default': 'cae_s3_tool',
                'help': 'Exported as CAE_S3_TOOL; bare name resolves via PATH.'
            },
            {
                'name': 'nprocs',
                'msg': '[read] Number of benchmark processes',
                'type': int,
                'default': 1,
                'help': 'Fallback for when one process cannot saturate the '
                        'link: M processes partition the key space via '
                        '--object-stride/--object-offset.'
            },
            {
                'name': 'ppn',
                'msg': '[read] Processes per node',
                'type': int,
                'default': 1,
                'help': 'Only used when nprocs > 1'
            },

            # --- write only ---
            {
                'name': 'purge_prefix',
                'msg': '[write] Delete stale objects under key_prefix first',
                'type': bool,
                'default': True,
                'help': 'Every row of a sweep shares one key prefix, so '
                        'without this objects_measured also counts what '
                        'earlier rows left behind and stops being an exact '
                        'check. Only the bdev prefix is deleted -- the zarr '
                        "store sits one level up and the raw-PUT floor's keys "
                        'in a sibling, so neither is in range. Needs bucket, '
                        'key_prefix and venv, and is skipped without them.'
            },
            {
                'name': 'venv',
                'msg': '[write] Python venv providing botocore',
                'type': str,
                'default': '',
                'help': 'Ares has no AWS CLI and no system botocore; reuse '
                        'the zarr venv. Empty falls back to sys.executable. '
                        'Used only for the purge and the object count.'
            },
        ]

    # ------------------------------------------------------------------
    # Paths. Named by mode so a read row and a write row can share one
    # shared_dir without either scraping the other's output.
    # ------------------------------------------------------------------

    def _paths(self):
        """
        Resolve this mode's output paths from ``self.shared_dir``.

        Called from both ``start()`` and ``_get_stat()`` rather than assigned
        once in ``_configure``: the sweep runner reloads a fresh instance and
        calls each without ``_configure`` in between.

        Returns:
            tuple: (output_path, rss_path, objects_path). The last is used by
            write mode only, but is returned unconditionally so ``clean()``
            can remove it without branching.
        """
        mode = self._mode()
        base = os.path.join(self.shared_dir, f'clio_s3_{mode}')
        return f'{base}_output.txt', f'{base}_time.txt', f'{base}_objects.txt'

    def _configure(self, **kwargs):
        """
        Validate configuration and export the S3/AWS environment.

        Note this only covers THIS process. The S3 I/O happens in the runtime
        daemon, so the pipeline must also forward the AWS variables to it --
        see the class docstring.
        """
        mode = self._mode()
        if int(self.config['num_objects']) <= 0:
            raise ValueError('clio_s3_bench: num_objects must be > 0')
        if int(self.config['concurrency']) <= 0:
            raise ValueError('clio_s3_bench: concurrency must be > 0')
        if mode == 'read':
            # Write mode can run without them (they only gate the optional
            # object count); read mode has nothing to read without them.
            if not self.config['bucket']:
                raise ValueError('clio_s3_bench: bucket is required in read '
                                 'mode')
            if not self.config['key_prefix']:
                raise ValueError('clio_s3_bench: key_prefix is required in '
                                 'read mode')

        self.output_path, self.rss_path, self.objects_path = self._paths()

        self._apply_aws_env()
        if mode == 'read':
            self.setenv('CAE_S3_TOOL', self.config['s3_tool'])
            self.setenv('TMPDIR', self.config['tmpdir'])
            os.makedirs(self.config['tmpdir'], exist_ok=True)

        self.log(f"CLIO S3 {mode} benchmark: {self.config['num_objects']} "
                 f"objects of {self.config['object_size']}, "
                 f"K={self.config['concurrency']}")

    def _build_cmd(self):
        """
        Assemble the driver command line for the configured mode.

        The two binaries take the same flags apart from their object-count and
        size spellings, which differ because they predate this merge.

        Returns:
            str: The command, without any output redirection.
        """
        mode = self._mode()
        count_flag = '--num-objects' if mode == 'read' else '--num-blobs'
        size_flag = '--object-size' if mode == 'read' else '--blob-size'
        # _paths() rather than self.rss_path: the attribute is None until
        # start() (or _configure) assigns it, and a None here becomes
        # "sequence item 3: expected str instance" inside the join below.
        _, rss_path, _ = self._paths()
        cmd = self._time_prefix(rss_path) + [
            self.EXECUTABLES[mode],
            count_flag, str(self.config['num_objects']),
            size_flag, str(self.config['object_size']),
            '--concurrency', str(self.config['concurrency']),
            '--tag-prefix', str(self.config['tag_prefix']),
            '--worker-threads', str(self.config['worker_threads']),
            '--label', self.LABELS[mode],
        ]
        if mode == 'read':
            # The write driver takes neither: its destination is the CTE tier
            # from the compose config, not a bucket it addresses itself.
            cmd += ['--bucket', str(self.config['bucket']),
                    '--key-prefix', str(self.config['key_prefix'])]
        if self.config['verify']:
            cmd.append('--verify')
        if mode == 'read' and not self.config['preflight']:
            cmd.append('--no-preflight')
        return ' '.join(cmd)

    def start(self):
        """
        Run the benchmark, capturing stdout+stderr for _get_stat.

        The on-disk output file is the contract between start() and
        _get_stat(): the sweep runner reloads a fresh package instance before
        collecting stats, so an in-memory buffer would be lost.
        """
        # The sweep runner reloads a fresh instance and calls start() WITHOUT
        # re-running _configure(), so the paths it sets are still None here.
        # Resolve them from framework attributes rather than trusting
        # _configure -- otherwise _time_prefix() feeds a None into ' '.join()
        # ("sequence item 3: expected str instance").
        mode = self._mode()
        self.output_path, self.rss_path, self.objects_path = self._paths()
        # Same reasoning as the paths: never trust _configure to have run.
        self.purged_count = None

        self._remove_stale(self.output_path, self.rss_path, self.objects_path)

        if mode == 'write':
            # Stale objects in the bucket are the same failure mode one level
            # out: objects_measured lists the prefix, so a previous row's
            # leftovers would be counted as this row's work.
            self._purge_prefix()

        cmd = self._build_cmd()
        nprocs = int(self.config['nprocs']) if mode == 'read' else 1
        if nprocs > 1:
            # PsshExecInfo does not accept pipe_stdout; embed the redirect.
            cmd += f' --object-stride {nprocs}'
            exec_info = PsshExecInfo(
                env=self.mod_env,
                hostfile=self.hostfile,
                nprocs=nprocs,
                ppn=self.config['ppn'],
            )
            cmd += f' > {self.output_path} 2>&1'
        else:
            exec_info = LocalExecInfo(
                env=self.mod_env,
                pipe_stdout=self.output_path,
                pipe_stderr=self.output_path,
            )

        self.log(f'Executing: {cmd}')
        result = Exec(cmd, exec_info).run()

        what = self.EXECUTABLES[mode]
        self._check_exit_codes(result, what, self.output_path)
        self._check_output_freshness(self.output_path, what)
        if mode == 'write':
            # Count what actually landed in the bucket while the runtime is
            # still up. It must happen here, not in _get_stat: the runtime's
            # teardown frees blocks, and FreeBlocks issues a DeleteObject per
            # block (s3_bdev_transport.cc), so by stat-collection time the
            # prefix may be empty and the count would read zero.
            self._measure_objects()
        self.log(f'Benchmark completed. Output: {self.output_path}')

    # ------------------------------------------------------------------
    # Write-mode bucket accounting. The botocore plumbing these sit on
    # (_bucket_target / _botocore_env / _run_botocore) lives in S3BenchBase.
    # ------------------------------------------------------------------

    def _purge_prefix(self):
        """
        Delete every object under the bdev key prefix before the run.

        WHY THIS EXISTS. All rows of a sweep share one key prefix, and
        `objects_measured` is a listing of that prefix -- so it counts
        whatever earlier rows left behind alongside what this row wrote. The
        36-row sweep of 2026-08-26 hit exactly that: every 4 MiB row reported
        448 objects against `num_objects: 256`, the excess being precisely 192
        x 1 MiB orphans that the 1 MiB rows' teardown never deleted. Throughput
        was unaffected -- those columns come from logical_bytes/wall_time_us,
        which the benchmark owns -- but the guard was blunted into a lower
        bound, and a row that wrote nothing would still have listed its
        predecessors' objects and looked plausible.

        Purging here rather than subtracting a baseline in the count is
        deliberate: it restores `objects_measured == num_objects` as an exact
        equality instead of leaving a comparison against a moving reference.

        WHY DELETING HERE IS SAFE. At start() this row has written nothing, so
        every object under the prefix belongs to a row that has already
        finished. That rests on the runtime tearing down between rows, which
        the sweep evidence supports: the orphan count held at exactly 192
        across all eighteen 4 MiB rows rather than accumulating, so each row
        does free its own blocks. If that ever stops holding, the failure is
        loud rather than silent -- purging live blocks shows up immediately as
        objects_measured < num_objects, which the gate already catches.

        Only the bdev prefix is in range. The zarr baseline's store sits one
        level up and the raw-PUT floor's keys in a sibling prefix, so neither
        can be reached; `_bucket_target` refuses an empty prefix outright, and
        the snippet asserts it again on the far side.

        NOT A ROOT-CAUSE FIX. Something in the bdev teardown path is still
        orphaning blocks -- FreeBlocks issues a DeleteObject per block, and 192
        of them did not happen. This makes the measurement honest; it does not
        explain the leak. `objects_purged` is recorded per row precisely so
        the leak stays visible: a nonzero value names the row whose
        predecessor leaked, and how much.

        Never raises: a purge problem must not fail a row that can still run.
        """
        if not self.config.get('purge_prefix'):
            return
        bucket, prefix = self._bucket_target('purge_prefix')
        if not bucket:
            return

        # Keys are collected across all pages BEFORE any delete: deleting
        # while the paginator is mid-listing invalidates its continuation
        # token and can silently skip a page.
        script = (
            'import os, sys, botocore.session\n'
            'b, p = sys.argv[1], sys.argv[2]\n'
            'assert p, "refusing to purge an empty prefix"\n'
            'c = botocore.session.get_session().create_client(\n'
            '    "s3", region_name=os.environ.get("AWS_DEFAULT_REGION"))\n'
            'keys = []\n'
            'for page in c.get_paginator("list_objects_v2").paginate(\n'
            '        Bucket=b, Prefix=p):\n'
            '    for o in page.get("Contents", []):\n'
            '        keys.append({"Key": o["Key"]})\n'
            'n = 0\n'
            'for i in range(0, len(keys), 1000):\n'
            '    batch = keys[i:i + 1000]\n'
            '    r = c.delete_objects(Bucket=b, Delete={"Objects": batch})\n'
            '    n += len(r.get("Deleted", []))\n'
            '    for e in r.get("Errors", []):\n'
            '        print("ERROR", e.get("Key"), e.get("Message"),\n'
            '              file=sys.stderr)\n'
            'print(n)\n')
        out = self._run_botocore(script, [bucket, prefix], 'purge_prefix')
        if out is None:
            self.log('purge_prefix: FAILED. objects_measured is now a lower '
                     'bound, not an exact count -- compare it against '
                     'put_count rather than num_objects for this row.')
            return
        try:
            self.purged_count = int(out.split()[0])
        except Exception:
            self.log(f'purge_prefix: unparseable output: {out!r}')
            return
        if self.purged_count:
            self.log(f'purge_prefix: deleted {self.purged_count} stale '
                     f'object(s) under s3://{bucket}/{prefix} left by an '
                     f'earlier row')
        else:
            self.log(f'purge_prefix: s3://{bucket}/{prefix} was already clean')

    def _measure_objects(self):
        """
        List the bdev's key prefix and record how many objects exist.

        WHY THIS IS NOT DERIVED. The benchmark reports `PUT count` as the
        object count, which is right only while the allocator satisfies each
        request with one block. It previously derived ceil(object_size /
        block_size) from a knob that configured nothing, and overstated the
        count 4x. A derived number that cannot be wrong-detected is worse than
        no number, so this lists the bucket and reports ground truth alongside
        it. When objects_measured != put_count the allocator fragmented -- or,
        if it is zero, nothing reached S3 at all and the throughput column is
        fiction.

        The count is exact only because `_purge_prefix` emptied the prefix
        before the run. With the purge disabled or failed it is a lower bound
        that includes earlier rows' leftovers.

        Never raises: a listing problem must not fail a row whose measurement
        already succeeded.
        """
        bucket, prefix = self._bucket_target('objects_measured')
        if not bucket:
            return
        script = (
            'import os, sys, botocore.session\n'
            'b, p = sys.argv[1], sys.argv[2]\n'
            'c = botocore.session.get_session().create_client(\n'
            '    "s3", region_name=os.environ.get("AWS_DEFAULT_REGION"))\n'
            'n = tot = 0\n'
            'for page in c.get_paginator("list_objects_v2").paginate(\n'
            '        Bucket=b, Prefix=p):\n'
            '    for o in page.get("Contents", []):\n'
            '        n += 1; tot += o["Size"]\n'
            'print(n, tot)\n')
        out = self._run_botocore(script, [bucket, prefix], 'objects_measured',
                                 timeout=180)
        if out is None:
            return
        try:
            count, total = out.split()[:2]
            int(count), int(total)
        except Exception:
            self.log(f'objects_measured: unparseable listing output: {out!r}')
            return
        with open(self.objects_path, 'w') as f:
            f.write(f'objects_measured {count}\nbytes_measured {total}\n')
            # Diagnostic, and the reason the purge is not silent: a nonzero
            # value in results.csv names the row whose predecessor leaked.
            if self.purged_count is not None:
                f.write(f'objects_purged {self.purged_count}\n')
        self.log(f'objects_measured: {count} objects, {total} bytes under '
                 f's3://{bucket}/{prefix}')
        if int(count) == 0:
            self.log('WARNING: the bdev prefix is EMPTY after a run the '
                     'benchmark called successful. Nothing reached S3; treat '
                     'the throughput columns as invalid.')

    def clean(self):
        """
        Remove benchmark output, and in read mode the staged temp files.

        The S3 objects themselves are NOT purged here: the bucket prefix is
        the pipeline's to manage (post_cmds), and clean() runs per-package
        without knowing whether another row still needs the data.
        """
        # An unusable mode must not abort the pipeline's clean sweep: the
        # files are named by mode, so remove both spellings and move on.
        try:
            mode = self._mode()
            self._remove_quietly(self._paths())
        except ValueError as e:
            self.log(f'clean: {e}; removing output for every mode')
            self._remove_globs(('clio_s3_read_*.txt', 'clio_s3_write_*.txt'))
            return
        if mode != 'read':
            return
        try:
            Exec(f"rm -rf {self.config['tmpdir']}/cae_s3_*",
                 PsshExecInfo(hostfile=self.hostfile)).run()
        except Exception as e:
            self.log(f'clean: temp sweep failed: {e}')

    def _get_stat(self, stat_dict):
        """
        Scrape the benchmark output into results.csv columns.

        Keys are `<pkg_id>.<label>.<metric>`, so a pipeline keeping the same
        pkg_name across modes gets `clio_s3.read.*` in a read sweep and
        `clio_s3.write.*` in a write sweep. This must never raise: jarvis calls
        it inside a try/except that logs a warning and continues, so an
        exception silently drops every column this package contributes.

        Args:
            stat_dict (dict): Collected statistics, modified in place.
        """
        # _mode() validates, and validation raises. Nothing in this method may
        # reach jarvis's bare `except Exception: warn`, which would silently
        # drop every column instead of naming the problem.
        try:
            mode = self._mode()
            output_path, rss_path, objects_path = self._paths()
        except ValueError as e:
            self.log(f'_get_stat: {e}; no columns collected')
            return
        found = self._scrape(output_path, stat_dict)
        parse_time_v(rss_path, self.pkg_id, mode, stat_dict)
        # Ground truth from the bucket, written by start(). Compare against
        # <pkg>.write.put_count: equal means one object per request as
        # expected, zero means nothing reached S3 and the row is fiction.
        if mode == 'write' and os.path.exists(objects_path):
            try:
                with open(objects_path, 'r') as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2:
                            stat_dict[f'{self.pkg_id}.write.{parts[0]}'] = \
                                int(parts[1])
            except Exception as e:
                self.log(f'Could not read {objects_path}: {e}')
        self._warn_if_empty(found, output_path)
