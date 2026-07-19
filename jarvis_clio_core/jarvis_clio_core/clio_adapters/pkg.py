"""
WRP CTE adapters interceptor.

This interceptor injects WRP (IoWarp) CTE adapter libraries via
LD_PRELOAD so that the target package's file I/O is rerouted through
the Content Transfer Engine.

Path selection is now driven entirely by the ``clio::`` filename prefix:
anything an application opens whose path starts with ``clio::`` gets
intercepted; everything else passes straight through to the kernel.
There is no include/exclude regex list, no CAE YAML config, and the
adapter page size is hard-coded to 1 MiB inside the adapter itself.

Supported adapters:
- POSIX  (read/write/open/close/stat/...)
- MPI-IO (MPI_File_*)
- STDIO  (fread/fwrite/fopen/...)
- HDF5 VFD / VOL
- NVIDIA GDS
- ADIOS2 plugin engine
"""
from jarvis_cd.core.pkg import Interceptor
import pathlib


class ClioAdapters(Interceptor):
    """
    WRP CTE Adapters Interceptor.

    Set ``posix: true`` (or ``mpiio`` / ``stdio`` / ``vfd`` / ``vol`` /
    ``nvidia_gds`` / ``adios2``) on this package in a pipeline YAML to
    LD_PRELOAD-inject the matching adapter into the next package's
    process. Applications opt their I/O calls into interception by
    prefixing paths with ``clio::`` — for example::

        fopen("clio::/data/out.bin", "w")

    Paths without the prefix are not intercepted, so an app can mix
    intercepted and pass-through I/O freely in the same process.
    """

    def _init(self):
        pass

    def _configure_menu(self):
        return [
            {'name': 'posix',
             'msg': 'Intercept POSIX I/O operations',
             'type': bool, 'default': False,
             'help': 'Intercepts read, write, open, close, lseek, stat, ...'},
            {'name': 'mpiio',
             'msg': 'Intercept MPI-IO operations',
             'type': bool, 'default': False,
             'help': 'Intercepts MPI_File_* operations'},
            {'name': 'stdio',
             'msg': 'Intercept STDIO operations',
             'type': bool, 'default': False,
             'help': 'Intercepts fread, fwrite, fopen, fclose, ...'},
            {'name': 'vfd',
             'msg': 'Intercept HDF5 I/O via VFD',
             'type': bool, 'default': False,
             'help': 'Enables HDF5 Virtual File Driver for CTE'},
            {'name': 'vol',
             'msg': 'Intercept HDF5 I/O via VOL connector',
             'type': bool, 'default': False,
             'help': 'Enables HDF5 VOL connector for CTE'},
            {'name': 'nvidia_gds',
             'msg': 'Intercept NVIDIA GDS I/O',
             'type': bool, 'default': False,
             'help': 'Intercepts NVIDIA GPUDirect Storage operations'},
            {'name': 'adios2',
             'msg': 'Enable ADIOS2 IOWarp engine plugin',
             'type': bool, 'default': False,
             'help': 'Sets ADIOS2_PLUGIN_PATH so ADIOS2 discovers the '
                     'iowarp_engine plugin'},
        ]

    def _find_adapter_lib(self, name):
        """Look up an adapter shared library on the host, or use the
        in-container `/usr/local/lib/` location when the pipeline runs
        containerized."""
        if hasattr(self, 'pipeline') and self.pipeline:
            if self.pipeline._has_containerized_packages():
                container_path = f'/usr/local/lib/lib{name}.so'
                self.log(f'Using container library path: {container_path}')
                return container_path
        return self.find_library(name)

    def _configure(self, **kwargs):
        """Locate each enabled adapter library and stash its absolute path
        in ``self.env`` for use by ``modify_env``. Raises if no adapter
        is enabled or a requested adapter library can't be found."""
        has_one = False

        if self.config['posix']:
            lib = self._find_adapter_lib('clio_cte_posix')
            if lib is None:
                raise Exception('Could not find clio_cte_posix library')
            self.env['CLIO_CTE_POSIX'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_cte_posix.so at {lib}')
            has_one = True

        if self.config['mpiio']:
            lib = self._find_adapter_lib('clio_cte_mpiio')
            if lib is None:
                raise Exception('Could not find clio_cte_mpiio library')
            self.env['CLIO_CTE_MPIIO'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_cte_mpiio.so at {lib}')
            has_one = True

        if self.config['stdio']:
            lib = self._find_adapter_lib('clio_cte_stdio')
            if lib is None:
                raise Exception('Could not find clio_cte_stdio library')
            self.env['CLIO_CTE_STDIO'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_cte_stdio.so at {lib}')
            has_one = True

        if self.config['vfd']:
            lib = self._find_adapter_lib('clio_cte_vfd')
            if lib is None:
                raise Exception('Could not find clio_cte_vfd library')
            self.env['CLIO_CTE_VFD'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_cte_vfd.so at {lib}')
            has_one = True

        if self.config['vol']:
            lib = self._find_adapter_lib('clio_hdf5_vol')
            if lib is None:
                raise Exception('Could not find clio_hdf5_vol library')
            self.env['CLIO_CTE_VOL'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_hdf5_vol.so at {lib}')
            has_one = True

        if self.config['nvidia_gds']:
            lib = self._find_adapter_lib('clio_cte_nvidia_gds')
            if lib is None:
                raise Exception('Could not find clio_cte_nvidia_gds library')
            self.env['CLIO_CTE_NVIDIA_GDS'] = lib
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libclio_cte_nvidia_gds.so at {lib}')
            has_one = True

        if self.config['adios2']:
            lib = self.find_library('iowarp_engine')
            if lib is None:
                raise Exception('Could not find iowarp_engine library')
            self.env['ADIOS2_PLUGIN_PATH'] = str(pathlib.Path(lib).parent)
            self.env['CLIO_CTE_ROOT'] = str(pathlib.Path(lib).parent.parent)
            self.log(f'Found libiowarp_engine.so at {lib}')
            has_one = True

        if not has_one:
            raise Exception(
                'No WRP CTE adapter selected. Enable at least one of: '
                'posix, mpiio, stdio, vfd, vol, nvidia_gds, adios2.'
            )

    def modify_env(self):
        """Modify the target package's environment to LD_PRELOAD each
        enabled adapter. Jarvis shares ``mod_env`` between this
        interceptor and the next package, so mutations here propagate
        directly to that package's process."""
        if self.config['posix']:
            self.prepend_env('LD_PRELOAD', self.env['CLIO_CTE_POSIX'])
            self.log("Added POSIX adapter to LD_PRELOAD")

        if self.config['mpiio']:
            self.prepend_env('LD_PRELOAD', self.env['CLIO_CTE_MPIIO'])
            self.log("Added MPI-IO adapter to LD_PRELOAD")

        if self.config['stdio']:
            self.prepend_env('LD_PRELOAD', self.env['CLIO_CTE_STDIO'])
            self.log("Added STDIO adapter to LD_PRELOAD")

        if self.config['vfd']:
            plugin_dir = str(pathlib.Path(self.env['CLIO_CTE_VFD']).parent)
            self.setenv('HDF5_PLUGIN_PATH', plugin_dir)
            self.setenv('HDF5_DRIVER', 'clio_cte_vfd')
            self.log(f"Configured HDF5 VFD plugin path: {plugin_dir}")

        if self.config['vol']:
            self.prepend_env('LD_PRELOAD', self.env['CLIO_CTE_VOL'])
            self.setenv('HDF5_VOL_CONNECTOR', 'clio')
            self.log("Added HDF5 VOL connector + set HDF5_VOL_CONNECTOR=clio")

        if self.config['nvidia_gds']:
            self.prepend_env('LD_PRELOAD', self.env['CLIO_CTE_NVIDIA_GDS'])
            self.log("Added NVIDIA GDS adapter to LD_PRELOAD")

        if self.config['adios2']:
            self.setenv('ADIOS2_PLUGIN_PATH', self.env['ADIOS2_PLUGIN_PATH'])
            self.log(f"Set ADIOS2_PLUGIN_PATH to "
                     f"{self.env['ADIOS2_PLUGIN_PATH']}")

    def clean(self):
        """Adapters don't write durable state; nothing to clean."""
        pass
