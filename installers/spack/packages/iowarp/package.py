# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Iowarp(CMakePackage):
    """IOWarp Core: Unified repository containing runtime (Chimaera),
    context-transport-primitives, context-transfer-engine,
    context-assimilation-engine, and context-exploration-engine."""

    homepage = "https://github.com/iowarp/clio-core"
    git = "https://github.com/iowarp/clio-core.git"

    # Branch versions
    version('main', branch='main', submodules=True, preferred=True)
    version('dev', branch='74-fix-context-transport-primitives-for-the-gpu', submodules=True)

    # Build variants
    variant('debug', default=False, description='Build in Debug mode')
    variant('shared', default=True, description='Build shared libraries')
    variant('test', default=True, description='Enable tests for all components')
    variant('benchmark', default=True, description='Enable benchmarks for all components')

    # Component enable/disable variants
    variant('runtime', default=True, description='Enable Chimaera runtime component')
    variant('cte', default=True, description='Enable context-transfer-engine component')
    variant('cae', default=True, description='Enable context-assimilation-engine component')
    variant('cee', default=True, description='Enable context-exploration-engine component')

    # Feature variants
    variant('posix', default=True, description='Enable POSIX adapter')
    variant('mpiio', default=True, description='Enable MPI I/O adapter')
    variant('stdio', default=True, description='Enable STDIO adapter')
    variant('hdf5', default=True, description='Enable HDF5')
    variant('ares', default=False, description='Enable full libfabric install')
    variant('mochi', default=False, description='Build with mochi-thallium support')
    variant('encrypt', default=False, description='Include encryption libraries')
    variant('compress', default=False, description='Include compression libraries')
    variant('python', default=False, description='Install python bindings')
    variant('elf', default=True, description='Build elf toolkit')
    variant('zmq', default=True, description='Build ZeroMQ support')
    variant('cuda', default=False, description='Enable CUDA support')
    variant('rocm', default=False, description='Enable ROCm support')
    variant('adios2', default=False, description='Build with ADIOS2 support')

    # Core dependencies (always required)
    depends_on('cmake@3.25:')
    depends_on('catch2@3.0.1')
    depends_on('yaml-cpp')
    depends_on('doxygen')
    depends_on('cereal')
    depends_on('msgpack-c@6:')
    depends_on('libaio')
    depends_on('libzmq', when='+zmq')

    # Python dependencies
    depends_on('python')
    depends_on('py-pip')
    depends_on('py-setuptools')

    # Conditional core dependencies
    depends_on('libelf', when='+elf')
    depends_on('mpi', when='+mpiio')
    depends_on('hdf5', when='+hdf5')
    depends_on('adios2', when='+adios2')

    # Networking libraries
    depends_on('libfabric fabrics=sockets,tcp,udp,verbs,mlx,rxm,rxd,shm', when='+ares')
    depends_on('mochi-thallium+cereal', when='+mochi')
    depends_on('argobots@1.1+affinity', when='+mochi')

    # Compression libraries (conditional on +compress)
    depends_on('lzo', when='+compress')
    depends_on('bzip2', when='+compress')
    depends_on('zstd', when='+compress')
    depends_on('lz4', when='+compress')
    depends_on('zlib', when='+compress')
    depends_on('xz', when='+compress')
    depends_on('brotli', when='+compress')
    depends_on('snappy', when='+compress')
    depends_on('c-blosc2', when='+compress')

    # Encryption libraries (conditional on +encrypt)
    depends_on('openssl', when='+encrypt')

    # GPU support (conditional)
    depends_on('cuda', when='+cuda')
    depends_on('rocm-core', when='+rocm')

    def cmake_args(self):
        args = []

        # Build type
        if '+debug' in self.spec:
            args.append(self.define('CMAKE_BUILD_TYPE', 'Debug'))
        else:
            args.append(self.define('CMAKE_BUILD_TYPE', 'Release'))

        # Shared/static libraries
        args.append(self.define_from_variant('BUILD_SHARED_LIBS', 'shared'))

        # Component enable/disable (using the naming from CMakeLists.txt)
        args.append(self.define_from_variant('WRP_CORE_ENABLE_RUNTIME', 'runtime'))
        args.append(self.define_from_variant('WRP_CORE_ENABLE_CTE', 'cte'))
        args.append(self.define_from_variant('WRP_CORE_ENABLE_CAE', 'cae'))
        args.append(self.define_from_variant('WRP_CORE_ENABLE_CEE', 'cee'))

        # Context-transport-primitives (HSHM) options
        if '+hdf5' in self.spec:
            args.append(self.define('HSHM_ENABLE_VFD', 'ON'))
        if '+compress' in self.spec:
            args.append(self.define('HSHM_ENABLE_COMPRESS', 'ON'))
        if '+encrypt' in self.spec:
            args.append(self.define('HSHM_ENABLE_ENCRYPT', 'ON'))
        if '+mochi' in self.spec:
            args.append(self.define('HSHM_RPC_THALLIUM', 'ON'))
        if '+zmq' in self.spec:
            args.append(self.define('HSHM_ENABLE_ZMQ_TESTS', 'ON'))
        if '+elf' in self.spec:
            args.append(self.define('HSHM_ENABLE_ELF', 'ON'))
        if '+cuda' in self.spec:
            args.append(self.define('HSHM_ENABLE_CUDA', 'ON'))
        if '+rocm' in self.spec:
            args.append(self.define('HSHM_ENABLE_ROCM', 'ON'))
        if '+adios2' in self.spec:
            args.append(self.define('WRP_CTE_ENABLE_ADIOS2_ADAPTER', 'ON'))
            args.append(self.define('WRP_CORE_ENABLE_GRAY_SCOTT', 'ON'))

        # Tests and benchmarks
        if '+test' in self.spec:
            args.append(self.define('WRP_CORE_ENABLE_TESTS', 'ON'))
            args.append(self.define('HSHM_ENABLE_TESTS', 'ON'))
            args.append(self.define('CHIMAERA_ENABLE_TESTS', 'ON'))
            args.append(self.define('WRP_CTE_ENABLE_TESTS', 'ON'))
            args.append(self.define('WRP_CAE_ENABLE_TESTS', 'ON'))
            args.append(self.define('WRP_CEE_ENABLE_TESTS', 'ON'))
        else:
            args.append(self.define('WRP_CORE_ENABLE_TESTS', 'OFF'))
            args.append(self.define('HSHM_ENABLE_TESTS', 'OFF'))
            args.append(self.define('CHIMAERA_ENABLE_TESTS', 'OFF'))
            args.append(self.define('WRP_CTE_ENABLE_TESTS', 'OFF'))
            args.append(self.define('WRP_CAE_ENABLE_TESTS', 'OFF'))
            args.append(self.define('WRP_CEE_ENABLE_TESTS', 'OFF'))

        if '+benchmark' in self.spec:
            args.append(self.define('WRP_CORE_ENABLE_BENCHMARKS', 'ON'))
            args.append(self.define('HSHM_ENABLE_BENCHMARKS', 'ON'))
            args.append(self.define('CHIMAERA_ENABLE_BENCHMARKS', 'ON'))
            args.append(self.define('WRP_CTE_ENABLE_BENCHMARKS', 'ON'))
            args.append(self.define('WRP_CAE_ENABLE_BENCHMARKS', 'ON'))
            args.append(self.define('WRP_CEE_ENABLE_BENCHMARKS', 'ON'))
        else:
            args.append(self.define('WRP_CORE_ENABLE_BENCHMARKS', 'OFF'))
            args.append(self.define('HSHM_ENABLE_BENCHMARKS', 'OFF'))
            args.append(self.define('CHIMAERA_ENABLE_BENCHMARKS', 'OFF'))
            args.append(self.define('WRP_CTE_ENABLE_BENCHMARKS', 'OFF'))
            args.append(self.define('WRP_CAE_ENABLE_BENCHMARKS', 'OFF'))
            args.append(self.define('WRP_CEE_ENABLE_BENCHMARKS', 'OFF'))

        # Chimaera runtime options (if enabled)
        if '+runtime' in self.spec:
            if '+cuda' in self.spec:
                args.append(self.define('CHIMAERA_ENABLE_CUDA', 'ON'))
            if '+rocm' in self.spec:
                args.append(self.define('CHIMAERA_ENABLE_ROCM', 'ON'))

        # Context-transfer-engine (CTE) options (if enabled)
        if '+cte' in self.spec:
            if '+posix' in self.spec:
                args.append(self.define('CTE_ENABLE_POSIX_ADAPTER', 'ON'))
            if '+mpiio' in self.spec:
                args.append(self.define('CTE_ENABLE_MPIIO_ADAPTER', 'ON'))
                if 'openmpi' in self.spec:
                    args.append(self.define('CTE_OPENMPI', 'ON'))
                elif 'mpich' in self.spec:
                    args.append(self.define('CTE_MPICH', 'ON'))
            if '+stdio' in self.spec:
                args.append(self.define('CTE_ENABLE_STDIO_ADAPTER', 'ON'))
            if '+hdf5' in self.spec:
                args.append(self.define('CTE_ENABLE_VFD', 'ON'))
            if '+compress' in self.spec:
                args.append(self.define('CTE_ENABLE_COMPRESS', 'ON'))
            if '+encrypt' in self.spec:
                args.append(self.define('CTE_ENABLE_ENCRYPT', 'ON'))
            if '+python' in self.spec:
                args.append(self.define('CTE_ENABLE_PYTHON', 'ON'))
            if '+cuda' in self.spec:
                args.append(self.define('CTE_ENABLE_CUDA', 'ON'))
            if '+rocm' in self.spec:
                args.append(self.define('CTE_ENABLE_ROCM', 'ON'))

        # Context-assimilation-engine (CAE) options (if enabled)
        if '+cae' in self.spec:
            if '+posix' in self.spec:
                args.append(self.define('CAE_ENABLE_POSIX_ADAPTER', 'ON'))
            if '+mpiio' in self.spec:
                args.append(self.define('CAE_ENABLE_MPIIO_ADAPTER', 'ON'))
                if 'openmpi' in self.spec:
                    args.append(self.define('CAE_OPENMPI', 'ON'))
                elif 'mpich' in self.spec:
                    args.append(self.define('CAE_MPICH', 'ON'))
            if '+stdio' in self.spec:
                args.append(self.define('CAE_ENABLE_STDIO_ADAPTER', 'ON'))
            if '+hdf5' in self.spec:
                args.append(self.define('CAE_ENABLE_VFD', 'ON'))
            if '+cuda' in self.spec:
                args.append(self.define('CAE_ENABLE_CUDA', 'ON'))
            if '+rocm' in self.spec:
                args.append(self.define('CAE_ENABLE_ROCM', 'ON'))

        return args

    def setup_run_environment(self, env):
        # Set up library and module paths
        env.prepend_path('LD_LIBRARY_PATH', self.prefix.lib)
        env.prepend_path('CMAKE_MODULE_PATH', self.prefix.cmake)
        env.prepend_path('CMAKE_PREFIX_PATH', self.prefix.cmake)

        # Add Python paths if Python bindings are enabled
        if '+python' in self.spec:
            env.prepend_path('PYTHONPATH', self.prefix.lib)
