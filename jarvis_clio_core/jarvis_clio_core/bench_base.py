"""Shared plumbing for the S3 benchmark packages.

Three packages drive the S3 sweeps -- ``clio_s3_bench`` (a C++ driver),
``zarr_s3_bench`` (zarr-python + s3fs) and ``s3_raw_put_bench`` (concurrent
``cae_s3_tool put``). They measure different stacks, but they all run a command
under ``/usr/bin/time -v``, redirect its output to a file under ``shared_dir``,
assert a results banner landed in that file, and scrape it in ``_get_stat``.
That envelope is what lives here.

WHY A BASE CLASS AND NOT ONE PACKAGE. The three remain separate because they
invoke genuinely different things. What they share is the envelope, not the
measurement -- and the envelope has repeatedly been where the bugs were. The
``start()``-time path re-resolution below was fixed once per package (five
times) before this file existed.

TWO RULES FOR SUBCLASSES.

1. The concrete package class MUST be *defined* in ``<pkg>/pkg.py`` -- inherit
   from this, never alias it. ``Pkg._detect_pkg_dir`` sets ``self.pkg_dir``
   from ``inspect.getfile(self.__class__)``, so a module-level alias such as
   ``ZarrS3Bench = S3BenchBase`` would silently point ``pkg_dir`` at *this*
   file's directory, and every package that loads a helper out of
   ``pkg_dir/scripts/`` would fail to find it.

2. Re-resolve every path from ``self.shared_dir`` / ``self.pkg_dir`` inside
   ``start()``. The sweep runner reloads a fresh instance and calls ``start()``
   WITHOUT ``_configure()``, so anything assigned there is still None. (``_init``
   runs from ``Pkg._ensure_directories``, ``_configure`` does not.)

NOTHING IN ``_get_stat``'S PATH MAY RAISE. jarvis calls ``_get_stat`` inside a
bare ``except Exception: warn``, so one exception silently drops EVERY column
the package would have contributed -- a green row with a blank throughput
column rather than a visible error.
"""

import glob
import os
import re
import subprocess
import sys

from jarvis_cd.core.pkg import Application

# The repo root is on sys.path when jarvis imports a package module (load_class
# inserts it), so the shared parser resolves as a namespace-package import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jarvis_clio_core.bench_parse import parse_bench_output  # noqa: E402

#: Emitted by ``clio_bench::PrintResults`` and mirrored by the Python drivers.
#: Its absence in an output file is the blank-column failure mode.
RESULTS_BANNER = re.compile(r'=== \w+ Benchmark Results ===')

#: The C++ drivers emit through HLOG, which colours its level tags.
_ANSI = re.compile(r'\033\[[0-9;]*m')

#: Endpoint overrides flip both cae_s3_tool and the Poco signer to path-style
#: addressing against a host that does not exist on real AWS.
_ENDPOINT_VARS = ('S3_ENDPOINT', 'AWS_ENDPOINT_URL')


class S3BenchBase(Application):
    """Common envelope for the S3 benchmark driver packages."""

    # ------------------------------------------------------------------
    # Mode
    # ------------------------------------------------------------------

    #: Modes a subclass accepts. Override to restrict (s3_raw_put_bench is
    #: write-only, so it does not expose a mode at all).
    MODES = ('read', 'write')

    def _mode(self):
        """
        Resolve the read/write mode, validated against ``MODES``.

        Read from ``self.config`` on every call rather than cached in
        ``_init``: ``_get_stat`` runs on a fresh instance whose ``_configure``
        never ran, and a cached value would be None there.

        Returns:
            str: 'read' or 'write'.

        Raises:
            ValueError: If the configured mode is not in ``MODES``.
        """
        mode = str(self.config.get('mode') or '').strip().lower()
        if mode not in self.MODES:
            raise ValueError(
                f'{type(self).__name__}: mode must be one of '
                f'{list(self.MODES)}, got {mode!r}')
        return mode

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _time_prefix(self, rss_path):
        """
        Build the GNU time(1) prefix used to capture peak RSS.

        Degrades to no prefix when /usr/bin/time is absent: peak RSS is a
        secondary metric, and hard-requiring the binary would turn a missing
        package into a total failure of every sweep row rather than one blank
        column. The shell builtin `time` cannot substitute -- no -v, no -o.

        Args:
            rss_path (str): Where time(1) should write its report.

        Returns:
            list: Command tokens to prepend (empty when time(1) is missing).
        """
        if os.path.exists('/usr/bin/time'):
            return ['/usr/bin/time', '-v', '-o', rss_path]
        self.log('WARNING: /usr/bin/time not found; peak RSS will not be '
                 'recorded (throughput columns are unaffected)')
        return []

    def _parse_size(self, value):
        """
        Parse a k/m/g size string into bytes.

        The C++ drivers take these suffixes natively (``clio_bench::ParseSize``);
        the Python drivers take plain integers. Parsing here is what lets every
        package in the sweep accept the SAME ``object_size: "4m"`` in the
        pipeline YAML instead of one stack spelling it 4194304.

        Args:
            value: A size string ('4m', '512k', '1g') or a plain number.

        Returns:
            int: Size in bytes.

        Raises:
            ValueError: On an unparseable value. Raised rather than defaulted:
                a silently-wrong size produces a plausible number for the
                wrong workload, which is worse than a failed row.
        """
        text = str(value).strip().lower()
        if not text:
            raise ValueError('empty size')
        mult = {'k': 1024, 'm': 1024 ** 2, 'g': 1024 ** 3}.get(text[-1])
        if mult is not None:
            text = text[:-1].strip()
        else:
            mult = 1
        try:
            scaled = float(text) * mult
        except ValueError:
            raise ValueError(f'unparseable size {value!r} '
                             f'(expected e.g. 4m, 512k, 1g, or a byte count)')
        if scaled <= 0:
            raise ValueError(f'size must be > 0, got {value!r}')
        return int(scaled)

    def _venv_python(self, venv):
        """
        Resolve the interpreter inside a venv, falling back to ``python3``.

        Args:
            venv (str): Venv root, possibly containing ``${VARS}``.

        Returns:
            str: Path to an interpreter, or the bare name 'python3'.
        """
        venv = os.path.expandvars(str(venv or ''))
        python = os.path.join(venv, 'bin', 'python3') if venv else ''
        return python if python and os.path.exists(python) else 'python3'

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def _apply_aws_env(self, clear_endpoint=True):
        """
        Export the AWS region and profile for THIS process.

        Note what this does not cover: the S3 I/O is performed by the runtime
        daemon (or a process it forks), which jarvis launches with an
        environment built from ``EnvironmentManager.COMMON_ENV_VARS`` -- a
        fixed list carrying no AWS_* entry. The pipeline must name the AWS
        variables in clio_runtime's ``forward_env`` for them to reach it.

        Args:
            clear_endpoint (bool): Drop S3_ENDPOINT / AWS_ENDPOINT_URL. Real
                AWS needs them unset; an override selects path-style
                addressing against a nonexistent host.
        """
        self.setenv('AWS_DEFAULT_REGION', self.config['aws_region'])
        if self.config.get('aws_profile'):
            self.setenv('AWS_PROFILE', self.config['aws_profile'])
        if not clear_endpoint:
            return
        for key in _ENDPOINT_VARS:
            for env in (self.env, self.mod_env):
                if isinstance(env, dict):
                    env.pop(key, None)

    def _activate_venv(self, venv):
        """
        Put a venv on PATH and PYTHONPATH.

        Deliberately not `conda activate` or `spack load`: both install shell
        functions that misbehave under a non-interactive job script.

        Args:
            venv (str): Venv root, possibly containing ``${VARS}``. A missing
                directory is a warning, not an error -- the interpreter on
                PATH may already carry the packages.
        """
        venv = os.path.expandvars(str(venv or ''))
        if not venv or not os.path.isdir(venv):
            self.log(f'Warning: venv not found at {venv}; relying on python3 '
                     f'already having the required packages')
            return
        self.prepend_env('PATH', os.path.join(venv, 'bin'))
        for site in glob.glob(
                os.path.join(venv, 'lib', 'python3.*', 'site-packages')):
            self.prepend_env('PYTHONPATH', site)

    # ------------------------------------------------------------------
    # Output files
    # ------------------------------------------------------------------

    def _remove_stale(self, *paths):
        """
        Delete output files left by a previous sweep combination.

        Stale output is the blank-column failure mode one level in: a crash in
        this row would otherwise be scored with the previous row's numbers.

        Args:
            *paths (str): Files to remove; None and absent paths are skipped.
        """
        for path in paths:
            if path and os.path.exists(path):
                os.remove(path)

    def _log_output_tail(self, path, n_lines=100):
        """
        Emit the tail of an output file into the Jarvis log.

        Args:
            path (str): File to tail.
            n_lines (int): How many trailing lines to log.
        """
        if not path or not os.path.exists(path):
            self.log(f'(no output file at {path})')
            return
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            self.log(f'failed to read {path}: {e}')
            return
        tail = lines[-n_lines:] if len(lines) > n_lines else lines
        self.log(f'--- {os.path.basename(path)} tail ({len(tail)} lines) ---')
        for line in tail:
            self.log(line.rstrip())
        self.log('--- end tail ---')

    def _check_output_freshness(self, path, what):
        """
        Raise unless the output carries the results banner.

        A crash partway through the transfer loop leaves a file with the
        startup banner but no results. That is the silent failure this guards:
        without it the row is scored green with a blank throughput column.

        Args:
            path (str): Output file to inspect.
            what (str): Driver name, for the error message.

        Raises:
            RuntimeError: If the file is missing, empty, or bannerless.
        """
        if not os.path.exists(path):
            raise RuntimeError(f'{what} produced no output at {path}')
        with open(path, 'r') as f:
            content = f.read()
        if not content.strip():
            raise RuntimeError(f'{what} output is empty: {path}')
        if not RESULTS_BANNER.search(_ANSI.sub('', content)):
            self._log_output_tail(path)
            raise RuntimeError(
                f'{what} output lacks the results banner: {path}')

    def _check_exit_codes(self, result, what, path):
        """
        Raise if any host's process exited non-zero, tailing the output first.

        Args:
            result: Return value of ``Exec(...).run()``.
            what (str): Driver name, for the error message.
            path (str): Output file to tail on failure.

        Raises:
            RuntimeError: If any exit code is non-zero.
        """
        exit_codes = getattr(result, 'exit_code', {}) or {}
        nonzero = {h: c for h, c in exit_codes.items() if c != 0}
        if nonzero:
            self._log_output_tail(path)
            raise RuntimeError(
                f'{what} exited with non-zero code(s): {nonzero}')

    def _remove_quietly(self, paths):
        """
        Delete files for ``clean()``, logging rather than raising on failure.

        Args:
            paths (iterable): Paths to remove; None and absent are skipped.
        """
        for path in paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                self.log(f'clean: could not remove {path}: {e}')

    def _remove_globs(self, patterns):
        """
        Delete every ``shared_dir`` file matching any glob pattern.

        Args:
            patterns (iterable): Globs relative to ``self.shared_dir``.
        """
        for pattern in patterns:
            self._remove_quietly(
                glob.glob(os.path.join(self.shared_dir, pattern)))

    # ------------------------------------------------------------------
    # Stat collection
    # ------------------------------------------------------------------

    def _scrape(self, path, stat_dict):
        """
        Read one output file and scrape its results blocks into ``stat_dict``.

        Never raises -- see the module docstring.

        Args:
            path (str): Output file written by ``start()``.
            stat_dict (dict): Collected statistics, modified in place.

        Returns:
            int: Number of metrics extracted; 0 when the file is missing,
            unreadable, or carries no results block.
        """
        if not os.path.exists(path):
            self.log(f'No output file found at {path}')
            return 0
        try:
            with open(path, 'r') as f:
                output = f.read()
        except Exception as e:
            self.log(f'Could not read {path}: {e}')
            return 0
        return parse_bench_output(output, self.pkg_id, stat_dict)

    def _warn_if_empty(self, found, where):
        """
        Log the blank-column warning when nothing was scraped.

        Args:
            found (int): Metric count returned by ``_scrape``.
            where (str): What was scraped, for the message.
        """
        if found == 0:
            self.log(f'Warning: no metrics extracted from {where}. A green '
                     f'row with a blank throughput column is a FAILURE.')

    # ------------------------------------------------------------------
    # S3 control plane
    #
    # Ares has no AWS CLI and no system botocore, so every control-plane call
    # shells into the configured venv. Callers are diagnostics around a
    # measurement that has its own success criteria, so none of this raises.
    # ------------------------------------------------------------------

    def _bucket_target(self, what):
        """
        Resolve (bucket, prefix) from the config, or (None, None).

        The prefix is required, not defaulted. An empty prefix widens every
        caller to the whole bucket -- harmless for a listing, but a purge would
        then reach the sibling stacks' data, which shares the bucket.

        Args:
            what (str): Label for the skip message.

        Returns:
            tuple: (bucket, prefix), or (None, None) when either is unset.
        """
        bucket = str(self.config.get('bucket') or '').strip()
        prefix = str(self.config.get('key_prefix') or '').strip().strip('/')
        if not bucket or not prefix:
            self.log(f'{what}: skipped (bucket/key_prefix not set)')
            return None, None
        return bucket, prefix

    def _botocore_env(self):
        """
        Build the environment for a botocore subprocess.

        AWS_PROFILE + HOME resolve through botocore's shared credentials file.
        mod_env carries both (jarvis copies HOME out of COMMON_ENV_VARS), and
        the job script's raw keys are inherited by this process, so those are
        passed through too for the profile-less case.

        Returns:
            dict: Environment for the subprocess.
        """
        env = dict(self.mod_env) if isinstance(self.mod_env, dict) \
            else dict(os.environ)
        for name in ('HOME', 'AWS_PROFILE', 'AWS_ACCESS_KEY_ID',
                     'AWS_SECRET_ACCESS_KEY', 'AWS_SESSION_TOKEN'):
            if not env.get(name) and os.environ.get(name):
                env[name] = os.environ[name]
        env['AWS_DEFAULT_REGION'] = str(self.config.get('aws_region')
                                        or env.get('AWS_DEFAULT_REGION')
                                        or 'us-east-1')
        return env

    def _run_botocore(self, script, args, what, timeout=300):
        """
        Run a botocore snippet in the configured venv.

        Never raises: every caller is a diagnostic around a measurement that
        has its own success criteria, and none should be able to fail a row on
        its own.

        Args:
            script (str): Python source to run with `-c`.
            args (list): Positional arguments appended after the script.
            what (str): Label for log messages.
            timeout (int): Seconds before the subprocess is killed.

        Returns:
            str: stdout on success, or None.
        """
        # Deliberately NOT _venv_python: that falls back to a bare `python3`
        # on PATH, which on Ares exists but has no botocore. Skipping loudly
        # beats a subprocess failure that looks like an S3 problem.
        venv = str(self.config.get('venv') or '').strip()
        python = os.path.join(venv, 'bin', 'python3') if venv \
            else sys.executable
        if not os.path.exists(python):
            self.log(f'{what}: skipped ({python} not found)')
            return None
        try:
            proc = subprocess.run([python, '-c', script] + list(args),
                                  env=self._botocore_env(),
                                  capture_output=True, timeout=timeout,
                                  text=True)
        except Exception as e:
            self.log(f'{what}: failed ({e})')
            return None
        if proc.returncode != 0:
            self.log(f'{what}: failed rc={proc.returncode}: '
                     f'{(proc.stderr or "").strip()[-400:]}')
            return None
        return proc.stdout

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def stop(self):
        """Nothing to stop: these benchmarks run to completion."""
        return True
