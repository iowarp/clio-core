from jarvis_cd.shell import Exec, LocalExecInfo
import os
import sys

# The repo root is on sys.path when jarvis imports this module, so the shared
# base and parser resolve as namespace-package imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_clio_core.bench_base import S3BenchBase  # noqa: E402
from jarvis_clio_core.bench_parse import parse_time_v  # noqa: E402


class S3RawPutBench(S3BenchBase):
    """
    Raw S3 PUT wire-speed floor.

    Drives scripts/s3_raw_put.py, which uploads N pre-staged files with K
    concurrent `cae_s3_tool put` processes and reports in the same format as
    clio_s3_bench and zarr_s3_bench.

    This is the row that makes the other two interpretable. Without a floor,
    a poor CLIO number cannot be attributed: it may be CLIO's block layer, or
    simply what this host can push to this bucket at this concurrency. Read it
    as a bound, not a competitor -- it does no chunking, no metadata, and no
    compression, so nothing in the comparison should beat it.

    WHY THIS ONE HAS NO `mode`. It is write-only by construction: there is no
    raw-GET counterpart, which is the same interpretability gap in the other
    direction. `MODES` is narrowed to ('write',) so the shared helpers still
    name the right label without offering a direction that does not exist.
    See jarvis_clio_core/pipelines/ares/docs/CLIO_S3_BENCH.md.
    """

    #: Write-only. Narrowing this (rather than adding a `mode` option) is what
    #: keeps _mode() usable here without implying a read path exists.
    MODES = ('write',)

    def _init(self):
        """Initialize instance state."""
        self.script_path = None
        self.output_path = None
        self.rss_path = None

    def _mode(self):
        """
        Always 'write'.

        Overridden rather than backed by a config option: the package exposes
        no direction to choose, so reading one out of config would invite a
        pipeline to set a value that cannot be honoured.

        Returns:
            str: 'write'.
        """
        return 'write'

    def _configure_menu(self):
        """
        Configure the application menu.

        Returns:
            List[Dict]: Configuration menu options for the benchmark.
        """
        return [
            {
                'name': 'bucket',
                'msg': 'S3 bucket to upload into',
                'type': str,
                'default': '',
                'help': 'Required.'
            },
            {
                'name': 'key_prefix',
                'msg': 'Key prefix for the uploaded objects',
                'type': str,
                'default': 'clio-s3-write-bench/rawput',
                'help': 'Keys resolve to <key_prefix>/raw_%06d.bin'
            },
            {
                'name': 'num_objects',
                'msg': 'Number of objects to upload',
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
                'help': "One process per in-flight PUT. Match the CLIO row's "
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
                'help': 'Peak usage is concurrency * object_size. Staging '
                        'happens before timing starts.'
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
            os.path.join(self.pkg_dir, 'scripts', 's3_raw_put.py'),
            os.path.join(self.shared_dir, 's3_raw_put_output.txt'),
            os.path.join(self.shared_dir, 's3_raw_put_time.txt'),
        )

    def _configure(self, **kwargs):
        """Validate configuration and export the AWS environment."""
        if not self.config['bucket']:
            raise ValueError('s3_raw_put_bench: bucket is required')
        if int(self.config['num_objects']) <= 0:
            raise ValueError('s3_raw_put_bench: num_objects must be > 0')
        if int(self.config['concurrency']) <= 0:
            raise ValueError('s3_raw_put_bench: concurrency must be > 0')
        object_size = self._parse_size(self.config['object_size'])

        self.script_path, self.output_path, self.rss_path = self._paths()
        if not os.path.exists(self.script_path):
            raise ValueError(f's3_raw_put.py not found at {self.script_path}')

        self._apply_aws_env()
        self.setenv('CAE_S3_TOOL', self.config['s3_tool'])
        os.makedirs(self.config['tmpdir'], exist_ok=True)

        self.log(f"Raw S3 PUT floor: {self.config['num_objects']} objects of "
                 f"{object_size} B, K={self.config['concurrency']}")

    def _build_cmd(self):
        """
        Assemble the s3_raw_put.py command line.

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
            '--label', 'Rawput',
        ]
        return ' '.join(cmd)

    def start(self):
        """Run the raw-PUT floor, capturing stdout+stderr for _get_stat."""
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

        # The script exits non-zero when any PUT failed: a partial upload
        # timed fewer bytes than it reports, so the row must fail rather than
        # publish a flattering number.
        self._check_exit_codes(result, 's3_raw_put.py', self.output_path)
        self._check_output_freshness(self.output_path, 's3_raw_put.py')
        self.log(f'Raw PUT floor completed. Output: {self.output_path}')

    def clean(self):
        """
        Remove benchmark output and any orphaned staging directories.

        The uploaded S3 objects are NOT purged here: the bucket prefix is the
        pipeline's to manage (post_cmds).
        """
        _, output_path, rss_path = self._paths()
        self._remove_quietly((output_path, rss_path))
        try:
            Exec(f"rm -rf {self.config['tmpdir']}/s3_raw_put_*",
                 LocalExecInfo()).run()
        except Exception as e:
            self.log(f'clean: temp sweep failed: {e}')

    def _get_stat(self, stat_dict):
        """
        Scrape the benchmark output into results.csv columns.

        Keys are `<pkg_id>.rawput.<metric>`. Must never raise: jarvis calls
        this inside a try/except that logs a warning and continues, so an
        exception silently drops every column this package contributes.

        Args:
            stat_dict (dict): Collected statistics, modified in place.
        """
        _, output_path, rss_path = self._paths()
        found = self._scrape(output_path, stat_dict)
        parse_time_v(rss_path, self.pkg_id, 'rawput', stat_dict)
        self._warn_if_empty(found, output_path)
