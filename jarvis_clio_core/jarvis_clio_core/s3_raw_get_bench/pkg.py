from jarvis_cd.shell import Exec, LocalExecInfo
import os
import sys

# The repo root is on sys.path when jarvis imports this module, so the shared
# base and parser resolve as namespace-package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_clio_core.bench_base import S3BenchBase  # noqa: E402
from jarvis_clio_core.bench_parse import parse_time_v  # noqa: E402


class S3RawGetBench(S3BenchBase):
    """
    Raw S3 GET wire-speed floor.

    Drives scripts/s3_raw_get.py, which downloads N objects with K
    concurrent `cae_s3_tool get` processes and reports in the same format as
    clio_s3_bench and zarr_s3_bench.

    This is the row that makes the other two interpretable. Without a floor,
    a poor CLIO number cannot be attributed: it may be CLIO's block layer, or
    simply what this host can push to this bucket at this concurrency. Read it
    as a bound, not a competitor -- it does no chunking, no metadata, and no
    compression, so nothing in the comparison should beat it.

    WHY THIS ONE HAS NO `mode`. It is read-only by construction: the write
    counterpart is s3_raw_put_bench. `MODES` is narrowed to ('read',) so the
    shared helpers name the right label without offering a direction that does
    not exist. See jarvis_clio_core/pipelines/ares/docs/CLIO_S3_BENCH.md.
    """

    #: Read-only. Narrowing this (rather than adding a `mode` option) is what
    #: keeps _mode() usable here without implying a write path exists.
    MODES = ('read',)

    def _init(self):
        """Initialize instance state."""
        self.script_path = None
        self.output_path = None
        self.rss_path = None

    def _mode(self):
        """
        Always 'read'.

        Overridden rather than backed by a config option: the package exposes
        no direction to choose, so reading one out of config would invite a
        pipeline to set a value that cannot be honoured.

        Returns:
            str: 'read'.
        """
        return 'read'

    def _configure_menu(self):
        """
        Configure the application menu.

        Returns:
            List[Dict]: Configuration menu options for the benchmark.
        """
        return [
            {
                'name': 'bucket',
                'msg': 'S3 bucket to download from',
                'type': str,
                'default': '',
                'help': 'Required.'
            },
            {
                'name': 'key_prefix',
                'msg': 'Key prefix of the objects to read',
                'type': str,
                'default': 'clio-s3-read-bench/raw',
                'help': 'Keys resolve to <key_prefix>/obj_%06d.bin -- the same '
                        'keys the CLIO read row reads. Match clio_s3.key_prefix.'
            },
            {
                'name': 'num_objects',
                'msg': 'Number of objects to download',
                'type': int,
                'default': 64,
                'help': "Match the CLIO row's num_objects."
            },
            {
                'name': 'object_size',
                'msg': 'Bytes per object',
                'type': str,
                'default': '4m',
                'help': "Suffixes k/m/g, same spelling as clio_s3.object_size "
                        '-- the underlying script takes a byte count, which '
                        'this package computes. Match the CLIO row.'
            },
            {
                'name': 'concurrency',
                'msg': 'Concurrent cae_s3_tool processes (K)',
                'type': int,
                'default': 8,
                'help': "One process per in-flight GET. Match the CLIO row's "
                        'concurrency.'
            },
            {
                'name': 's3_tool',
                'msg': 'Path to the cae_s3_tool helper',
                'type': str,
                'default': 'cae_s3_tool',
                'help': 'Built under CAE_ENABLE_S3 / spack +s3_cae. '
                        'Resolved on PATH when left as the bare name.'
            },
            {
                'name': 'tmpdir',
                'msg': 'Staging directory for the source files',
                'type': str,
                'default': '/tmp',
                'help': 'Peak usage is concurrency * object_size (one '
                        'destination file per slot, overwritten each GET).'
            },
            {
                'name': 'aws_region',
                'msg': 'AWS region',
                'type': str,
                'default': 'us-east-1',
                'help': "Must match the bucket's real region."
            },
            {
                'name': 'aws_profile',
                'msg': 'AWS profile name',
                'type': str,
                'default': '',
                'help': 'Resolved by cae_s3_tool through the AWS SDK chain.'
            },
        ]

    def _paths(self):
        """
        Resolve the script and output paths from framework attributes.

        Called from ``start()`` and ``_get_stat()`` rather than assigned once
        in ``_configure``: the sweep runner reloads a fresh instance and calls
        each without ``_configure`` in between.

        Returns:
            tuple: (script_path, output_path, rss_path).
        """
        return (
            os.path.join(self.pkg_dir, 'scripts', 's3_raw_get.py'),
            os.path.join(self.shared_dir, 's3_raw_get_output.txt'),
            os.path.join(self.shared_dir, 's3_raw_get_time.txt'),
        )

    def _configure(self, **kwargs):
        """Validate configuration and export the AWS environment."""
        if not self.config['bucket']:
            raise ValueError('s3_raw_get_bench: bucket is required')
        if int(self.config['num_objects']) <= 0:
            raise ValueError('s3_raw_get_bench: num_objects must be > 0')
        if int(self.config['concurrency']) <= 0:
            raise ValueError('s3_raw_get_bench: concurrency must be > 0')
        object_size = self._parse_size(self.config['object_size'])

        self.script_path, self.output_path, self.rss_path = self._paths()
        if not os.path.exists(self.script_path):
            raise ValueError(f's3_raw_get.py not found at {self.script_path}')

        self._apply_aws_env()
        self.setenv('CAE_S3_TOOL', self.config['s3_tool'])
        os.makedirs(self.config['tmpdir'], exist_ok=True)

        self.log(f"Raw S3 GET floor: {self.config['num_objects']} objects of "
                 f"{object_size} B, K={self.config['concurrency']}")

    def _build_cmd(self):
        """
        Assemble the s3_raw_get.py command line.

        Returns:
            str: The command, without any output redirection.
        """
        # _paths() rather than the attributes: they are None until start()
        # (or _configure) assigns them, and a None here becomes "sequence item
        # N: expected str instance" inside the join below.
        script_path, _, rss_path = self._paths()
        cmd = self._time_prefix(rss_path) + [
            'python3', script_path,
            '--bucket', str(self.config['bucket']),
            '--key-prefix', str(self.config['key_prefix']),
            '--num-objects', str(self.config['num_objects']),
            # The script takes a byte count; the package takes the same k/m/g
            # spelling every other package in the sweep uses.
            '--object-size', str(self._parse_size(self.config['object_size'])),
            '--concurrency', str(self.config['concurrency']),
            '--s3-tool', str(self.config['s3_tool']),
            '--tmpdir', str(self.config['tmpdir']),
            '--label', 'Rawget',
        ]
        return ' '.join(cmd)

    def start(self):
        """Run the raw-GET floor, capturing stdout+stderr for _get_stat."""
        # The sweep runner reloads a fresh instance and calls start() WITHOUT
        # re-running _configure(), so the paths set there are still None here.
        # Resolve them from framework attributes rather than trusting
        # _configure -- otherwise the command becomes "python None ...".
        self.script_path, self.output_path, self.rss_path = self._paths()
        self._remove_stale(self.output_path, self.rss_path)

        cmd = self._build_cmd()
        self.log(f'Executing: {cmd}')
        result = Exec(cmd, LocalExecInfo(
            env=self.mod_env,
            pipe_stdout=self.output_path,
            pipe_stderr=self.output_path)).run()

        # The script exits non-zero when any GET failed: a partial download
        # timed fewer bytes than it reports, so the row must fail rather than
        # publish a flattering number.
        self._check_exit_codes(result, 's3_raw_get.py', self.output_path)
        self._check_output_freshness(self.output_path, 's3_raw_get.py')
        self.log(f'Raw GET floor completed. Output: {self.output_path}')

    def clean(self):
        """
        Remove benchmark output and any orphaned staging directories.

        The read objects are inputs staged by the pipeline; this floor never
        creates or deletes S3 objects.
        """
        _, output_path, rss_path = self._paths()
        self._remove_quietly((output_path, rss_path))
        try:
            Exec(f"rm -rf {self.config['tmpdir']}/s3_raw_get_*",
                 LocalExecInfo()).run()
        except Exception as e:
            self.log(f'clean: temp sweep failed: {e}')

    def _get_stat(self, stat_dict):
        """
        Scrape the benchmark output into results.csv columns.

        Keys are `<pkg_id>.rawget.<metric>`. Must never raise: jarvis calls
        this inside a try/except that logs a warning and continues, so an
        exception silently drops every column this package contributes.

        Args:
            stat_dict (dict): Collected statistics, modified in place.
        """
        _, output_path, rss_path = self._paths()
        found = self._scrape(output_path, stat_dict)
        parse_time_v(rss_path, self.pkg_id, 'rawget', stat_dict)
        self._warn_if_empty(found, output_path)
