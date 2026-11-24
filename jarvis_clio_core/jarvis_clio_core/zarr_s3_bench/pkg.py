from jarvis_cd.shell import Exec, LocalExecInfo
import glob
import os
import sys

# The repo root is on sys.path when jarvis imports this module, so the shared
# base and parser resolve as namespace-package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_clio_core.bench_base import S3BenchBase  # noqa: E402
from jarvis_clio_core.bench_parse import parse_time_v  # noqa: E402


class ZarrS3Bench(S3BenchBase):
    """
    Zarr S3 Benchmark Application -- both directions, selected by `mode`.

    mode: read
        Drives scripts/zarr_s3_read.py, which reads a Zarr v3 store from S3
        via zarr-python + s3fs.

    mode: write
        Drives scripts/zarr_s3_write.py, which writes one via the same stack.

    Either way it reports throughput in the same format as `clio_s3_bench`, so
    one parser serves both stacks. It is the baseline half of the comparison;
    see the class docstring there for why direction is a config option rather
    than a package name.

    Runs once PER COMPRESSION VARIANT within a single sweep row, emitting
    labels `Read`/`ReadZstd` or `Write`/`WriteZstd`. bench_parse lowercases
    them, so the columns are `<pkg_id>.read.*` / `<pkg_id>.readzstd.*`. Both
    land in the same results.csv row, so the compression comparison costs no
    extra sweep combinations and no extra CLIO runs.

    Compression is the biggest confound in either direction and it favours
    zarr: a zstd store moves fewer bytes than CLIO does for the same logical
    payload. Both S3 pipelines therefore pin `variants: ["none"]` -- CLIO gets
    its own compression mechanism later this year, and until it does the
    compressed comparison measures zstd rather than either system. The
    machinery stays because that is a temporary state; `compressibility` makes
    the source entropy an explicit input rather than an artifact of the test
    data, and matters again the moment a codec is switched back on -- see the
    script's module docstring.
    """

    #: One driver script per mode, resolved under ``pkg_dir/scripts``.
    SCRIPTS = {'read': 'zarr_s3_read.py', 'write': 'zarr_s3_write.py'}

    #: Prefix of the results label; the capitalized variant is appended for
    #: compressed runs, giving Read/ReadZstd and Write/WriteZstd. bench_parse
    #: lowercases the label, so the columns read `readzstd` / `writezstd`.
    LABELS = {'read': 'Read', 'write': 'Write'}

    def _init(self):
        """Initialize instance state."""
        self.script_path = None

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
                'help': 'Selects the driver script, the results label prefix, '
                        'and which mode-specific options below apply. Set it '
                        'in the pipeline YAML alongside clio_s3_bench.mode; '
                        'the two must agree or the row compares a read '
                        'against a write.'
            },
            {
                'name': 'bucket',
                'msg': 'S3 bucket holding the Zarr stores',
                'type': str,
                'default': '',
                'help': 'Required. read: staged by '
                        'scripts/stage_s3_read_bench_data.py. write: the '
                        'stores are OVERWRITTEN on every run.'
            },
            {
                'name': 'store_prefix',
                'msg': 'Key prefix containing the zarr/ directory',
                'type': str,
                'default': 'clio-s3-read-bench',
                'help': 'Stores resolve to <store_prefix>/zarr/'
                        'bench_c<chunk_edge>_<variant>.zarr when reading, and '
                        '<store_prefix>/zarr/wbench_<variant>.zarr when '
                        'writing.'
            },
            {
                'name': 'variants',
                'msg': 'Compression variants to run in this row',
                'type': list,
                'default': ['none'],
                'help': "Each runs once; 'none' emits the bare label, 'zstd' "
                        'emits <Label>Zstd (the readzstd/writezstd columns), '
                        'both into the same results row. Defaults to '
                        "uncompressed ONLY: comparing a compressed zarr "
                        'against an uncompressed CLIO measures the codec, so '
                        'a codec belongs here only once CLIO has one too.'
            },
            {
                'name': 'async_concurrency',
                'msg': 'zarr async.concurrency (parallel chunk transfers)',
                'type': int,
                'default': 8,
                'help': "The Zarr-side concurrency knob. The reference suite's "
                        'default of 10 is far too low for WAN S3.'
            },
            {
                'name': 'venv',
                'msg': 'Path to the venv providing zarr, s3fs and numpy',
                'type': str,
                'default': '${HOME}/zarr-venv',
                'help': 'Prepended to PATH/PYTHONPATH. Do NOT use conda '
                        'activate or spack load: they install shell functions '
                        'that misbehave under the job script.'
            },
            {
                'name': 'aws_region',
                'msg': 'AWS region',
                'type': str,
                'default': 'us-east-1',
                'help': "Passed to the s3fs client; must match the bucket's "
                        'real region.'
            },
            {
                'name': 'aws_profile',
                'msg': 'AWS profile name from ~/.aws/credentials',
                'type': str,
                'default': '',
                'help': 'Resolved through the standard botocore chain -- '
                        'unlike the CLIO side, zarr can use a profile '
                        'directly. Never put secrets in the pipeline YAML.'
            },

            # --- read only ---
            {
                'name': 'chunk_edge',
                'msg': '[read] Cubic chunk edge selecting the staged store',
                'type': int,
                'default': 256,
                'help': 'One of 64/128/256/512 -> 512 KiB/4 MiB/32 MiB/256 MiB '
                        'per chunk. NOT a spelling of object_size: it is a '
                        'component of the store KEY, naming which pre-staged '
                        'store to open, and the chunk count comes from that '
                        'store rather than from configuration. Sweep it in '
                        'lockstep with clio_s3.object_size.'
            },
            {
                'name': 'anon',
                'msg': '[read] Read anonymously (public buckets only)',
                'type': bool,
                'default': False,
                'help': 'Skips credential resolution entirely'
            },

            # --- write only ---
            {
                'name': 'num_objects',
                'msg': '[write] Number of chunks to write',
                'type': int,
                'default': 64,
                'help': 'One zarr chunk is one S3 object, so this is the same '
                        "unit as clio_s3.num_objects. Set both from the same "
                        'sweep variable: they are what makes the two stacks '
                        'move the same data.'
            },
            {
                'name': 'object_size',
                'msg': '[write] Bytes per chunk',
                'type': str,
                'default': '4m',
                'help': 'Suffixes k/m/g, same spelling as '
                        'clio_s3.object_size. The script takes --total-bytes '
                        'and --chunk-bytes; both are computed from this pair '
                        '(total = num_objects * object_size), which is what '
                        'used to be a hand-maintained equality in the YAML.'
            },
            {
                'name': 'compressibility',
                'msg': '[write] Source data entropy, 0.0 random .. 1.0 constant',
                'type': float,
                'default': 0.5,
                'help': 'Decides whether zstd sends fewer bytes than CLIO. At '
                        '0.0 zstd cannot compress at all and slightly EXPANDS '
                        'the data, so the zstd row would measure only encode '
                        'overhead; at 1.0 it compresses to nothing. Neither '
                        'resembles real scientific arrays.'
            },
        ]

    def _configure(self, **kwargs):
        """Validate configuration and put the zarr venv on PATH/PYTHONPATH."""
        mode = self._mode()
        if not self.config['bucket']:
            raise ValueError('zarr_s3_bench: bucket is required')
        if mode == 'write':
            # Read mode takes its extent from the staged store, so these are
            # meaningless there and are not validated.
            if int(self.config['num_objects']) <= 0:
                raise ValueError('zarr_s3_bench: num_objects must be > 0')
            self._parse_size(self.config['object_size'])

        self.script_path = self._script_path()
        if not os.path.exists(self.script_path):
            raise ValueError(
                f'{self.SCRIPTS[mode]} not found at {self.script_path}')

        self._activate_venv(self.config['venv'])
        # Not _apply_aws_env's endpoint clearing: the zarr scripts accept an
        # S3-compatible endpoint deliberately (--endpoint-url, or S3_ENDPOINT
        # from the environment), which is how they run against a local stand-in.
        self._apply_aws_env(clear_endpoint=False)

        if mode == 'read':
            self.log(f"Zarr S3 read benchmark: chunk_edge="
                     f"{self.config['chunk_edge']} variants="
                     f"{self.config['variants']} "
                     f"concurrency={self.config['async_concurrency']}")
        else:
            self.log(f"Zarr S3 write benchmark: "
                     f"{self.config['num_objects']} chunks of "
                     f"{self.config['object_size']} variants="
                     f"{self.config['variants']} "
                     f"concurrency={self.config['async_concurrency']}")

    # ------------------------------------------------------------------
    # Paths. Named by mode so a read row and a write row can share one
    # shared_dir without either scraping the other's output.
    # ------------------------------------------------------------------

    def _script_path(self):
        """
        Resolve this mode's driver script under ``pkg_dir``.

        Returns:
            str: Absolute path to the script.
        """
        return os.path.join(self.pkg_dir, 'scripts', self.SCRIPTS[self._mode()])

    def _output_prefix(self):
        """
        Shared-dir filename stem for this mode.

        Returns:
            str: e.g. `<shared_dir>/zarr_s3_write`.
        """
        return os.path.join(self.shared_dir, f'zarr_s3_{self._mode()}')

    def _output_path(self, variant):
        """
        Per-variant output file.

        Args:
            variant (str): Compression variant.

        Returns:
            str: Path under the shared dir.
        """
        return f'{self._output_prefix()}_output_{variant}.txt'

    def _rss_path(self, variant):
        """
        Per-variant /usr/bin/time report.

        Args:
            variant (str): Compression variant.

        Returns:
            str: Path under the shared dir.
        """
        return f'{self._output_prefix()}_time_{variant}.txt'

    def _variant_label(self, variant):
        """
        Results namespace for a compression variant.

        Args:
            variant (str): 'none' or a codec name such as 'zstd'.

        Returns:
            str: The bare mode label for uncompressed, else `<Label><Variant>`.
        """
        base = self.LABELS[self._mode()]
        return base if variant == 'none' else f'{base}{variant.capitalize()}'

    def _build_cmd(self, python, variant, rss_path):
        """
        Assemble the driver command line for one compression variant.

        Args:
            python (str): Interpreter to run the script with.
            variant (str): Compression variant.
            rss_path (str): Where time(1) should write its report.

        Returns:
            str: The command, without any output redirection.
        """
        mode = self._mode()
        # _script_path() rather than self.script_path: the attribute is None
        # until start() (or _configure) assigns it, and a None here becomes
        # "sequence item 1: expected str instance" inside the join below.
        cmd = self._time_prefix(rss_path) + [python, self._script_path()]
        if mode == 'read':
            store_key = (f"{self.config['store_prefix']}/zarr/"
                         f"bench_c{self.config['chunk_edge']}_{variant}.zarr")
        else:
            store_key = (f"{self.config['store_prefix']}/zarr/"
                         f'wbench_{variant}.zarr')
        cmd += ['--bucket', str(self.config['bucket']),
                '--store-key', store_key,
                '--async-concurrency', str(self.config['async_concurrency']),
                '--region', str(self.config['aws_region']),
                '--label', self._variant_label(variant)]
        if mode == 'read':
            if self.config['anon']:
                cmd.append('--anon')
        else:
            # The script's inputs are total/chunk bytes; the package's are the
            # object count and unit it shares with clio_s3_bench. One chunk is
            # one object, so n_chunks = ceil(total/chunk) == num_objects.
            chunk_bytes = self._parse_size(self.config['object_size'])
            total_bytes = chunk_bytes * int(self.config['num_objects'])
            cmd += ['--total-bytes', str(total_bytes),
                    '--chunk-bytes', str(chunk_bytes),
                    '--compressor', variant,
                    '--compressibility', str(self.config['compressibility'])]
        return ' '.join(cmd)

    def start(self):
        """Run the driver once per compression variant."""
        # The sweep runner reloads a fresh instance and calls start() WITHOUT
        # re-running _configure(), so self.script_path set there is still None
        # here. Resolve it from self.pkg_dir (a framework attribute set on
        # every instance) rather than trusting _configure -- otherwise the
        # command becomes "python None ..." and can't open the script.
        mode = self._mode()
        self.script_path = self._script_path()
        python = self._venv_python(self.config['venv'])
        what = self.SCRIPTS[mode]

        for variant in self.config['variants']:
            out = self._output_path(variant)
            rss = self._rss_path(variant)
            # Stale output from a previous combination is the blank-column
            # failure mode -- a crash here would otherwise be scored with old
            # numbers.
            self._remove_stale(out, rss)

            cmd = self._build_cmd(python, variant, rss)
            self.log(f'Executing: {cmd}')
            result = Exec(cmd, LocalExecInfo(
                env=self.mod_env, pipe_stdout=out, pipe_stderr=out)).run()

            self._check_exit_codes(result, f'{what} ({variant})', out)
            self._check_output_freshness(out, f'{what} ({variant})')

        self.log(f'Zarr {mode} benchmark completed for variants '
                 f"{self.config['variants']}")

    def clean(self):
        """
        Remove per-variant output and timing files.

        The S3 stores themselves are NOT purged here: the bucket prefix is the
        pipeline's to manage (post_cmds).
        """
        self._remove_globs(('zarr_s3_read_output_*.txt',
                            'zarr_s3_read_time_*.txt',
                            'zarr_s3_write_output_*.txt',
                            'zarr_s3_write_time_*.txt'))

    def _get_stat(self, stat_dict):
        """
        Scrape every variant's output into results.csv columns.

        Keys are `<pkg_id>.<label>.<metric>`, with e.g. `write` and
        `writezstd` sharing one row. Must never raise: jarvis calls this inside
        a try/except that logs a warning and continues, so an exception
        silently drops every column this package contributes.

        Args:
            stat_dict (dict): Collected statistics, modified in place.
        """
        # _mode() validates, and validation raises -- which here would mean
        # losing every column rather than reporting the problem.
        try:
            prefix = self._output_prefix()
        except ValueError as e:
            self.log(f'_get_stat: {e}; no columns collected')
            return
        total = 0
        # Variants are recovered from the filenames rather than from config:
        # a run that was interrupted after one variant should still file the
        # columns it did produce.
        paths = sorted(glob.glob(f'{prefix}_output_*.txt'))
        if not paths:
            self.log(f'No zarr output files found matching {prefix}_output_*')
            return
        marker = f'{os.path.basename(prefix)}_output_'
        for path in paths:
            total += self._scrape(path, stat_dict)
            variant = os.path.basename(path)[len(marker):-len('.txt')]
            parse_time_v(self._rss_path(variant), self.pkg_id,
                         self._variant_label(variant).lower(), stat_dict)
        self._warn_if_empty(total, str(paths))
