"""
This module provides classes and methods to deploy JuiceFS as a Jarvis
service package.

JuiceFS is a POSIX-compatible distributed filesystem that stores file
*data* in an object store (here a local directory via ``--storage file``)
and file *metadata* in a transactional engine (here Redis). This package
formats the filesystem once, mounts it via FUSE, and unmounts it on stop --
mirroring the lifecycle style of the ``redis`` service package.

The expected deployment order is: redis -> juicefs -> <benchmark driver>,
so the Redis metadata engine is already accepting connections before
``juicefs format`` runs.
"""
import os
import time
from jarvis_cd.core.pkg import Application
from jarvis_cd.util.logger import Color
from jarvis_cd.shell import Exec, LocalExecInfo, PsshExecInfo
from jarvis_clio_core.container_utils import (
    container_kwargs, flatten_stdout, eff_hostfile, single_instance_menu_opt)


class Juicefs(Application):
    """
    Format and FUSE-mount a JuiceFS filesystem (Redis metadata +
    local-file object backend) for single-node benchmarking.
    """

    def _init(self):
        """
        Initialize paths. Concrete (env-expanded) paths are resolved in
        ``_configure`` and recomputed on demand via ``_paths``.
        """
        pass

    def _configure_menu(self):
        """
        Create a CLI menu for the configurator method.

        :return: List(dict)
        """
        return [
            {
                'name': 'meta_url',
                'msg': 'Metadata engine URL (Redis) for format/mount',
                'type': str,
                'default': 'redis://127.0.0.1:6379/1',
            },
            {
                'name': 'storage',
                'msg': "Object storage backend type ('file' local dir, or 'gs' for Google Cloud Storage)",
                'type': str,
                'default': 'file',
            },
            {
                'name': 'bucket',
                'msg': "Object bucket URL for non-file storage, e.g. gs://my-bucket (required when storage='gs')",
                'type': str,
                'default': '',
            },
            {
                'name': 'gcs_credentials',
                'msg': 'Path to a GCS service-account JSON key (exported as GOOGLE_APPLICATION_CREDENTIALS; never stored in config)',
                'type': str,
                'default': '',
            },
            {
                'name': 'data_dir',
                'msg': "Local object-store dir (used as --bucket only when storage='file')",
                'type': str,
                'default': '${HOME}/juicefs_data',
            },
            {
                'name': 'mountpoint',
                'msg': 'FUSE mountpoint directory',
                'type': str,
                'default': '${HOME}/juicefs_mnt',
            },
            {
                'name': 'name',
                'msg': 'JuiceFS volume name (passed to juicefs format)',
                'type': str,
                'default': 'jfsbench',
            },
            {
                'name': 'cache_dir',
                'msg': 'Local cache directory for the mount',
                'type': str,
                'default': '${HOME}/juicefs_cache',
            },
            {
                'name': 'cache_size_mb',
                'msg': 'Local cache size cap in MiB (juicefs --cache-size). '
                       'JuiceFS defaults to 100 GiB, which can exhaust a small '
                       'node-local scratch device; bound it well under free '
                       'space on the cache_dir filesystem.',
                'type': int,
                'default': 512,
            },
            {
                'name': 'juicefs_bin',
                'msg': 'Path to the juicefs binary',
                'type': str,
                'default': 'juicefs',
            },
            {
                'name': 'format_fresh',
                'msg': "Wipe the local bucket dir before formatting (storage='file' only; no-op for cloud)",
                'type': bool,
                'default': True,
            },
            {
                'name': 'mount_wait',
                'msg': 'Seconds to wait for the mount to become ready',
                'type': int,
                'default': 20,
            },
            {
                'name': 'extra_mount_opts',
                'msg': 'Extra options appended to juicefs mount',
                'type': str,
                'default': '',
            },
            # Head-node-only pinning: JuiceFS is a single mount here (its redis
            # metadata store lives only on the first host), so on a multi-node
            # pipeline it must NOT fan format/mount/rm out to every node.
            single_instance_menu_opt(),
        ]

    def _paths(self):
        """
        Resolve env-expanded, user-expanded absolute paths from config.

        :return: tuple(data_dir, mountpoint, cache_dir)
        """
        def fix(p):
            return os.path.expanduser(os.path.expandvars(str(p)))
        return (fix(self.config['data_dir']),
                fix(self.config['mountpoint']),
                fix(self.config['cache_dir']))

    def _bucket_arg(self):
        """
        The value passed to ``juicefs format --bucket``. For storage='file'
        this is the local data_dir; for cloud backends (e.g. 'gs') it is the
        configured bucket URL (gs://...).

        :return: str
        """
        if self.config['storage'] == 'file':
            data_dir, _, _ = self._paths()
            return data_dir
        return str(self.config['bucket'])

    def _inject_cloud_env(self):
        """
        Populate ``self.mod_env`` with object-store credentials for the
        spawned juicefs subprocesses. Secrets are never read from config: the
        JSON key stays on disk and only its *path* (``gcs_credentials``) is
        turned into ``GOOGLE_APPLICATION_CREDENTIALS``. Any ``GCS_*`` already
        in the parent environment is passed through verbatim (env-based
        credential chain).

        Must run in ``start``/``stop`` -- ``_configure`` runs in a separate
        process, so mod_env mutations there would not survive to the Exec
        (the pipeline rebuilds mod_env from the persisted env each invocation).

        :return: None
        """
        if self.config['storage'] != 'gs':
            return
        cred = os.path.expanduser(os.path.expandvars(
            str(self.config.get('gcs_credentials', ''))))
        if cred:
            self.mod_env['GOOGLE_APPLICATION_CREDENTIALS'] = cred
        # Pass through pre-set GCS_* (e.g. a short-lived GCS_ACCESS_TOKEN, or
        # GCS_PROJECT_ID) that the env allowlist would otherwise drop.
        for key in ('GCS_ACCESS_TOKEN', 'GCS_PROJECT_ID'):
            if key not in self.mod_env and key in os.environ:
                self.mod_env[key] = os.environ[key]

    def _configure(self, **kwargs):
        """
        Validate config. The mount/cache/bucket directories are NOT created
        here: ``_configure`` runs host-side in the jarvis process BEFORE the
        apptainer instance exists, so a host ``os.makedirs`` of an in-container
        path like ``/mnt/jfs_mnt`` would hit the host's root-owned ``/mnt`` and
        fail (Errno 13). The dirs are instead created in ``start`` via a
        (container-wrapped) Exec — mirroring clio_cte_libfuse's ``Mkdir`` in
        ``start``. Harmless host mkdir in a bare-metal deploy.

        :param kwargs: Configuration parameters for this pkg.
        :return: None
        """
        if not self.config['meta_url']:
            raise ValueError('juicefs: meta_url must be set')

        storage = self.config['storage']
        if storage == 'gs':
            if not str(self.config.get('bucket', '')).startswith('gs://'):
                raise ValueError(
                    "juicefs: storage='gs' requires bucket=gs://<bucket>")
            if not str(self.config.get('gcs_credentials', '')) \
                    and 'GCS_ACCESS_TOKEN' not in os.environ:
                self.log("juicefs: storage='gs' with no gcs_credentials and no "
                         "GCS_ACCESS_TOKEN; relying on Google's ADC chain (e.g. "
                         "`gcloud auth application-default login`, or a GCE "
                         "attached service account). Valid if ADC is configured; "
                         "fails only if the chain resolves no credentials.",
                         color=Color.YELLOW)
        elif storage != 'file':
            self.log(f"juicefs: storage='{storage}' is configured but only "
                     f"'file' and 'gs' are exercised by this package",
                     color=Color.YELLOW)

    def _is_mounted(self, mountpoint):
        """
        Report whether ``mountpoint`` has a filesystem mounted *in the
        deployment context* (the apptainer instance's mount namespace for a
        container deploy, or the host for bare-metal). Uses a container-wrapped
        Exec probe rather than host-side ``os.path.ismount``: the JuiceFS FUSE
        mount lives inside the instance, which the host can't see.

        :param mountpoint: Absolute mountpoint path.
        :return: bool
        """
        probe = Exec(
            f'mountpoint -q {mountpoint} && echo __JFS_MOUNTED__ || true',
            PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                         collect_output=True, **container_kwargs(self))).run()
        return '__JFS_MOUNTED__' in flatten_stdout(probe)

    def start(self):
        """
        Format (idempotent) and FUSE-mount the JuiceFS filesystem, then
        block until the mount is ready.

        :return: None
        """
        jfs = self.config['juicefs_bin']
        data_dir, mountpoint, cache_dir = self._paths()
        meta_url = self.config['meta_url']

        # Make object-store credentials available to BOTH subprocesses below
        # (same process as the Exec -- see _inject_cloud_env docstring).
        self._inject_cloud_env()

        # Create the mount/cache (and, for storage='file', bucket) dirs in the
        # deployment context. Under a non-setuid apptainer these are container
        # paths (e.g. /mnt/jfs_mnt), so they must be made via a container-wrapped
        # Exec, NOT host-side os.makedirs (which would hit the host's root-owned
        # /mnt). Bare-metal: a harmless local mkdir -p.
        mkdirs = [mountpoint, cache_dir]
        if self.config['storage'] == 'file':
            mkdirs.append(data_dir)
        Exec(f'mkdir -p {" ".join(mkdirs)}',
             PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()

        # Fresh slate: drop any orphaned chunks from a previous run. This is a
        # LOCAL operation, only meaningful for storage='file' (the Redis
        # metadata is wiped per-run by the redis package, so stale chunks here
        # would only be dead bytes). For cloud backends, wiping the bucket is
        # destructive re-init and is out of scope -- log and skip.
        if self.config['format_fresh']:
            if self.config['storage'] == 'file':
                self.log(f'Clearing bucket dir {data_dir}', color=Color.YELLOW)
                # Container-wrapped (see mkdir rationale above): host-side
                # shutil.rmtree on /mnt/jfs_data would fail / hit the wrong fs.
                Exec(f'rm -rf {data_dir} && mkdir -p {data_dir}',
                     PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()
            else:
                self.log(f"format_fresh: skipping bucket wipe for "
                         f"storage='{self.config['storage']}' (no-op; "
                         f"destructive bucket re-init is out of scope)",
                         color=Color.YELLOW)

        # Format the volume (safe to re-run against fresh metadata).
        fmt = [
            jfs, 'format',
            '--storage', self.config['storage'],
            '--bucket', self._bucket_arg(),
            meta_url,
            self.config['name'],
        ]
        self.log(f"Formatting JuiceFS: {' '.join(fmt)}", color=Color.YELLOW)
        Exec(' '.join(fmt), PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()

        # Mount via FUSE. --background daemonizes and returns once the
        # mount is registered.
        mnt = [
            jfs, 'mount',
            meta_url,
            mountpoint,
            '--cache-dir', cache_dir,
            # Cap the local cache: JuiceFS defaults to 100 GiB, which can fill a
            # small node-local scratch device (e.g. Ares compute /tmp lives on a
            # ~2 GiB-free root fs) and tip a high-thread combo into ENOSPC.
            '--cache-size', str(self.config['cache_size_mb']),
            '--background',
        ]
        if self.config['extra_mount_opts']:
            mnt.append(self.config['extra_mount_opts'])
        self.log(f"Mounting JuiceFS: {' '.join(mnt)}", color=Color.YELLOW)
        Exec(' '.join(mnt), PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()

        # Wait for the mount to actually be ready before downstream I/O.
        deadline = int(self.config['mount_wait'])
        for _ in range(deadline):
            if self._is_mounted(mountpoint):
                self.log(f'JuiceFS mounted at {mountpoint}', color=Color.GREEN)
                return
            time.sleep(1)
        raise RuntimeError(
            f'juicefs: mountpoint {mountpoint} not ready after '
            f'{deadline}s (check redis connectivity and /dev/fuse access)')

    def stop(self):
        """
        Unmount the JuiceFS filesystem.

        :return: None
        """
        _, mountpoint, _ = self._paths()
        jfs = self.config['juicefs_bin']
        Exec(f'{jfs} umount {mountpoint}',
             PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()
        # Fallback for a wedged mount; ignore failure if already gone.
        if self._is_mounted(mountpoint):
            Exec(f'fusermount -u {mountpoint}',
                 PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()

    def clean(self):
        """
        Unmount (if needed) and delete the local cache (and, for
        storage='file', the local bucket) directories. A cloud bucket is
        never deleted here -- teardown is left to the provider / lifecycle
        rules (see scripts/smoke_juicefs_gcs.sh for manual cleanup).

        :return: None
        """
        data_dir, mountpoint, cache_dir = self._paths()
        if self._is_mounted(mountpoint):
            self.stop()
        dirs = [cache_dir]
        if self.config['storage'] == 'file':
            dirs.append(data_dir)
        # Container-wrapped removal (see start()'s mkdir rationale): host-side
        # shutil.rmtree can't reach the in-container paths.
        Exec(f'rm -rf {" ".join(dirs)}',
             PsshExecInfo(env=self.mod_env, hostfile=eff_hostfile(self),
                          **container_kwargs(self))).run()
