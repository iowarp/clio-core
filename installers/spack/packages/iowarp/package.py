# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import glob
import os

from spack.package import *


class Iowarp(CMakePackage, CudaPackage, ROCmPackage):
    """IOWarp Core: Unified repository containing runtime (Clio),
    context-transport-primitives, context-transfer-engine,
    context-assimilation-engine, and context-exploration-engine."""

    homepage = "https://github.com/iowarp/clio-core"
    # This package is a source build of the whole tree: every optional feature
    # the CMake build has (GPU memory expansion, the I/O adapters, clio-ml, the
    # dashboard, cloud tiers) is a variant here. .github/workflows/spack.yml
    # installs the feature variants on every change to this file.
    git = "https://github.com/iowarp/clio-core.git"

    # Branch versions
    version('main', branch='main', submodules=True, preferred=True)
    version('dev', branch='dev', submodules=True)

    # Build variants
    variant('debug', default=False, description='Build in Debug mode')
    variant('shared', default=True, description='Build shared libraries')
    variant('test', default=True, description='Enable tests for all components')
    variant('benchmark', default=True, description='Enable benchmarks for all components')

    # Component enable/disable variants
    variant('runtime', default=True, description='Enable Clio runtime component')
    variant('cte', default=True, description='Enable context-transfer-engine component')
    variant('cae', default=True, description='Enable context-assimilation-engine component')
    variant('cee', default=True, description='Enable context-exploration-engine component')

    # Feature variants
    variant('posix', default=True, description='Enable POSIX adapter')
    variant('mpiio', default=True, description='Enable MPI I/O adapter')
    variant('stdio', default=True, description='Enable STDIO adapter')
    variant('hdf5', default=True, description='Enable HDF5 import in the CAE')
    variant('vol', default=True,
            description='Build the HDF5 VOL connector (needs HDF5 1.14+)')
    variant('vfd', default=True,
            description='Build the HDF5 virtual file driver (needs HDF5 1.14+)')
    variant('ares', default=False, description='Enable full libfabric install')
    variant('mochi', default=False, description='Build with mochi-thallium support')
    variant('encrypt', default=False, description='Include encryption libraries')
    variant('compress', default=False, description='Include compression libraries')
    variant('python', default=False, description='Install python bindings')
    variant('elf', default=True, description='Build elf toolkit')
    variant('zmq', default=True, description='Build ZeroMQ support')
    # +cuda / cuda_arch come from CudaPackage, +rocm / amdgpu_target from
    # ROCmPackage.
    variant('gpu_vector', default=False,
            description='Build gpu_vector (memory expansion in CUDA kernels): '
                        'uses clang as the CUDA compiler, since nvcc cannot '
                        'compile device coroutines. Needs +cuda.')
    variant('sycl', default=False,
            description='Enable Intel GPU support through SYCL (builds '
                        'gpu_vector for SYCL). Needs the oneAPI compiler.')
    variant('gds', default=False,
            description='Build the NVIDIA GPUDirect Storage (cuFile) adapter')
    variant('clio_ml', default=False,
            description='Build clio-ml (Spark-MLlib-style algorithms over the '
                        'paged vectors); its backend follows +cuda / +sycl')
    variant('viz', default=True,
            description='Build the runtime web dashboard (needs Poco)')
    variant('adios2', default=False, description='Build with ADIOS2 support')
    variant('fuse', default=False, description='Enable FUSE3 adapter (CTE)')
    variant('s3_cae', default=False,
            description='Enable Amazon S3 (s3://) import in CAE (aws-sdk-cpp)')
    variant('gcs', default=False,
            description='Enable Google Cloud Storage (gs://) import in CAE '
                        '(google-cloud-cpp storage)')
    variant('s3_bdev', default=False,
            description='Enable the S3 (kS3) bdev storage tier (Poco + SigV4). '
                        'Distinct from +s3_cae, which is the CAE s3:// importer.')
    variant('daos', default=False,
            description='Enable the native DAOS (kDaos) bdev storage tier: '
                        'a DFS file driven through libdfs/libdaos instead '
                        'of a dfuse mount.')
    variant('gcs_bdev', default=False,
            description='Enable the Google Cloud Storage (kGcs) bdev storage '
                        'tier (Poco). Distinct from +gcs, the CAE gs:// importer.')
    variant('globus', default=False,
            description='Enable Globus transfer support in the CAE (Poco)')
    variant('io_uring', default=True,
            description='Enable the io_uring async I/O backend (Linux)')
    variant('openmp', default=False, description='Enable OpenMP')
    variant('boost_coro', default=False,
            description='Use Boost.Context stackful coroutine backend (issue #620)')

    # Language virtuals. Spack 1.x assigns a package a compiler only if it
    # declares these; without them iowarp gets no compiler wrapper and CMake
    # falls back to /usr/bin/c++, which on SLES 15 is GCC 7.5 and cannot do
    # the C++20 this project requires.
    depends_on('c', type='build')
    depends_on('cxx', type='build')

    # C++20 with coroutines: GCC 10+ / Clang 10+.
    conflicts('%gcc@:9', msg='iowarp requires C++20; use gcc@10: or newer')
    conflicts('%clang@:9', msg='iowarp requires C++20; use clang@10: or newer')

    # Core dependencies (always required)
    depends_on('cmake@3.25:')
    depends_on('catch2@3.0.1')
    depends_on('yaml-cpp')
    depends_on('doxygen')
    depends_on('cereal')
    depends_on('msgpack-c')
    depends_on('libaio')
    depends_on('libzmq', when='+zmq')
    depends_on('nlohmann-json')  # CAE summarizer label_client JSON parsing
    depends_on('curl')           # CAE summarizer label_client HTTP (libcurl)

    # Python dependencies
    depends_on('python')
    depends_on('py-pip')
    depends_on('py-setuptools')
    depends_on('py-pyyaml', when='+python')
    depends_on('py-msgpack', when='+python')

    # Conditional core dependencies
    depends_on('libelf', when='+elf')
    depends_on('mpi', when='+mpiio')
    depends_on('hdf5', when='+hdf5')
    # The VOL connector and the VFD use the HDF5 1.14 plugin APIs.
    depends_on('hdf5@1.14:', when='+vol')
    depends_on('hdf5@1.14:', when='+vfd')
    # ADIOS2 2.12 replaced the plugin-engine interface (adios2/plugin/,
    # built on the C++ bindings rather than core::Engine); the engine is
    # written against the 2.11 one.
    depends_on('adios2@:2.11', when='+adios2')
    depends_on('libfuse@3:', when='+fuse')
    depends_on('liburing', when='+io_uring')
    # The dashboard, the GCS tier and the Globus importer all speak HTTP(S)
    # through Poco (find_package(Poco COMPONENTS Net NetSSL Crypto JSON)).
    depends_on('poco', when='+viz')
    depends_on('poco', when='+gcs_bdev')
    depends_on('poco', when='+globus')
    depends_on('boost+context', when='+boost_coro')

    # CAE cloud import backends (opt-in). The S3 assimilator now needs BOTH:
    #   * Poco -- the in-process s3:// READ path signs SigV4 and streams over
    #     Poco::Net (shared s3_rest.h from the bdev), because loading the AWS SDK
    #     into the CLIO_INIT'ing runtime corrupts startup; and
    #   * aws-sdk-cpp -- still built for the standalone cae_s3_tool helper, used
    #     by the benchmark raw-PUT/raw-GET floors and test seeding, never linked
    #     into the runtime (find_package(AWSSDK COMPONENTS s3)).
    # google-cloud-cpp (iowarp-overlay, storage-only) is for the GCS assimilator.
    depends_on('aws-sdk-cpp', when='+s3_cae')
    depends_on('poco', when='+s3_cae')
    depends_on('google-cloud-cpp', when='+gcs')

    # The S3 bdev tier is a Poco::Net HTTPS client that signs with SigV4; it
    # deliberately does NOT link aws-sdk-cpp, because loading aws-cpp-sdk-core
    # into a process that calls CLIO_INIT corrupts runtime startup, and a
    # per-block bdev cannot fork a helper the way the CAE assimilator does. So
    # +s3_bdev and +s3_cae are genuinely different dependency sets, not aliases.
    #
    # No Spack builtin repo ships `poco`, so this tree vendors one under
    # installers/spack/packages/poco. On a system that already provides Poco
    # (e.g. libpoco-dev), declare it a Spack external instead of building it.
    depends_on('poco', when='+s3_bdev')

    # The DAOS bdev tier links the DAOS client library (libdfs, libdaos,
    # libgurt; headers daos.h / daos_fs.h). CMake finds it through
    # CMAKE_PREFIX_PATH (cmake/FindDAOS.cmake), so any `daos` install works.
    # On a machine with a site DAOS (e.g. Aurora, daos-devel under /usr) do
    # not build it: declare the system install as a Spack external --
    #   packages:
    #     daos:
    #       buildable: false
    #       externals:
    #       - spec: daos@2.6.5
    #         prefix: /usr
    # The client must match the servers, which only the site install does.
    depends_on('daos', when='+daos')

    conflicts('+s3_cae', when='~cae', msg='S3 import lives under CAE; enable +cae')
    conflicts('+gcs', when='~cae', msg='GCS import lives under CAE; enable +cae')

    conflicts('+fuse', when='~cte', msg='fuse adapter lives under CTE; enable +cte')
    conflicts('+daos', when='~runtime',
              msg='the DAOS bdev tier lives in the runtime; enable +runtime')

    # The POSIX/STDIO/MPI-IO adapters intercept calls through the ELF toolkit;
    # without it CMake silently skips them (#1194).
    for _adapter in ('posix', 'stdio', 'mpiio'):
        conflicts('+' + _adapter, when='~elf',
                  msg='the {0} adapter needs +elf'.format(_adapter))

    # Networking libraries
    # +ares: build libfabric with the full Ares-rail fabric set. The
    # spec is a single node in the concretized graph, so this constraint
    # propagates to mochi-thallium's mercury dep automatically when
    # both +ares and +mochi are set.
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

    # GPU compression libraries (conditional on +compress +cuda). These operate
    # on device pointers and back the ctp::Cusz / ctp::Ndzip / ctp::Cuszp wrappers.
    #   - cuSZ has an upstream Spack package.
    #   - ndzip and cuSZp have NO upstream Spack package; they are source-built in
    #     the NVIDIA devcontainer (.devcontainer/nvidia-gpu/Dockerfile) instead.
    #     Add a depends_on here if/when a Spack package becomes available.
    depends_on('cusz', when='+compress +cuda')

    # Encryption libraries (conditional on +encrypt)
    depends_on('openssl', when='+encrypt')

    # GPU support. CudaPackage adds depends_on('cuda') under +cuda;
    # ROCmPackage adds hip and friends under +rocm.
    #
    # gpu_vector's device verbs are C++20 coroutines, which only clang can
    # compile for CUDA, so +gpu_vector brings a clang that the build uses as
    # CMAKE_CUDA_COMPILER (CLIO_GPU_CLANG). The tree is tested with clang 22;
    # older than 19 lacks the device-coroutine support it needs.
    depends_on('llvm@19: +clang', type='build', when='+gpu_vector')
    # CudaPackage refuses newer clangs as nvcc's host compiler (e.g. clang 20+
    # with CUDA <= 12.9). With +gpu_vector clang compiles the CUDA sources
    # itself and nvcc is not involved, so that check does not apply.
    depends_on('cuda+allow-unsupported-compilers', when='+gpu_vector')
    conflicts('+gpu_vector', when='~cuda',
              msg='+gpu_vector builds the CUDA vector; add +cuda (a SYCL '
                  'build gets gpu_vector from +sycl alone)')
    conflicts('+gds', when='~cuda', msg='the GDS adapter needs +cuda')
    conflicts('+gds', when='~elf', msg='the GDS adapter needs +elf')
    conflicts('+sycl', when='+cuda',
              msg='pick one GPU language: +cuda or +sycl')
    # SYCL needs the oneAPI DPC++ compiler (icpx -fsycl) as the C++ compiler.
    requires('%oneapi', when='+sycl',
             msg='+sycl builds with the oneAPI compiler: add %oneapi')

    # clio-ml lives in its own repository and is built as external/clio-ml.
    resource(name='clio-ml', git='https://github.com/iowarp/clio-ml.git',
             branch='main', destination='external', placement='clio-ml',
             when='+clio_ml')
    conflicts('+clio_ml', when='~cte', msg='clio-ml is built on the CTE vectors')
    conflicts('+vol', when='~cte', msg='the VOL connector lives under CTE')
    conflicts('+vfd', when='~cte', msg='the VFD lives under CTE')
    conflicts('+gcs_bdev', when='~runtime',
              msg='the GCS bdev tier lives in the runtime; enable +runtime')
    conflicts('+globus', when='~cae', msg='Globus import lives under CAE')

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
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_RUNTIME', 'runtime'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_CTE', 'cte'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_CAE', 'cae'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_CEE', 'cee'))
        args.append(self.define_from_variant(
            'CLIO_CORE_ENABLE_BOOST_COROUTINES', 'boost_coro'))

        # Feature options. Every name here is an option() in the root
        # CMakeLists.txt; the CTP_*/CTE_*/CAE_* spellings this used to pass
        # are derived there, not read, so those variants built nothing (#1194).
        # define_from_variant also passes OFF, which matters for the adapters
        # whose CMake default is ON (POSIX).
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_ELF', 'elf'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_ZMQ', 'zmq'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_THALLIUM', 'mochi'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_MPI', 'mpiio'))
        if '+mpiio' in self.spec:
            # The MPI-IO adapter reads implementation-specific MPI_Status
            # fields and probes mpicc to tell OpenMPI from MPICH; that probe
            # does not recognize every MPI Spack can choose (Intel MPI, for
            # one, is MPICH-ABI). Say which family was built against.
            if self.spec['mpi'].name == 'openmpi':
                args.append(self.define('CLIO_CTE_OPENMPI', True))
            else:
                args.append(self.define('CLIO_CTE_MPICH', True))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_HDF5', 'hdf5'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_ENCRYPT', 'encrypt'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_PYTHON', 'python'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_IO_URING', 'io_uring'))
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_OPENMP', 'openmp'))
        args.extend(self._gpu_args())
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_CLIO_ML', 'clio_ml'))
        if '+clio_ml' in self.spec:
            args.append(self.define_from_variant('CLIO_ML_PYTHON', 'python'))
        if '+adios2' in self.spec:
            args.append(self.define('CLIO_CTE_ENABLE_ADIOS2_ADAPTER', 'ON'))
            args.append(self.define('CLIO_CORE_ENABLE_GRAY_SCOTT', 'ON'))

        # Tests and benchmarks: one switch each covers every component.
        args.append(self.define_from_variant('CLIO_CORE_ENABLE_TESTS', 'test'))
        args.append(self.define_from_variant(
            'CLIO_CORE_ENABLE_BENCHMARKS', 'benchmark'))

        # Context-transfer-engine (CTE) adapters. The I/O adapters also need
        # +elf (enforced by the conflicts above).
        cte = '+cte' in self.spec
        for variant, option in (
                ('posix', 'CLIO_CTE_ENABLE_POSIX_ADAPTER'),
                ('stdio', 'CLIO_CTE_ENABLE_STDIO_ADAPTER'),
                ('mpiio', 'CLIO_CTE_ENABLE_MPIIO_ADAPTER'),
                ('vfd', 'CLIO_CTE_ENABLE_VFD'),
                ('gds', 'CLIO_CTE_ENABLE_NVIDIA_GDS_ADAPTER'),
                ('compress', 'CLIO_CTE_ENABLE_COMPRESS'),
                ('fuse', 'CLIO_CTE_ENABLE_FUSE_ADAPTER')):
            on = cte and ('+' + variant) in self.spec
            args.append(self.define(option, on))
        args.append(self.define('CLIO_CTE_ENABLE_HDF5_VOL',
                                cte and '+vol' in self.spec))

        # Context-assimilation-engine (CAE) object-store importers.
        if '+cae' in self.spec:
            if '+s3_cae' in self.spec:
                args.append(self.define('CAE_ENABLE_S3', 'ON'))
            if '+gcs' in self.spec:
                args.append(self.define('CAE_ENABLE_GCS', 'ON'))
            args.append(self.define_from_variant('CAE_ENABLE_GLOBUS', 'globus'))

        # The S3 bdev tier lives in the runtime's bdev module, not under CAE,
        # so it is gated outside the '+cae' block above.
        if '+s3_bdev' in self.spec:
            args.append(self.define('CLIO_ENABLE_AMAZON_DRIVE', 'ON'))

        # The native DAOS bdev tier, also in the runtime's bdev module.
        # define_from_variant passes OFF too, so a ~daos build never picks up
        # a stale ON from a reused build directory.
        args.append(self.define_from_variant('CLIO_ENABLE_DAOS', 'daos'))
        args.append(self.define_from_variant('CLIO_ENABLE_GOOGLE_CLOUD', 'gcs_bdev'))

        return args

    def _gpu_args(self):
        """CMake defines for the GPU backends (+cuda, +rocm, +sycl)."""
        spec = self.spec
        args = [
            self.define_from_variant('CLIO_CORE_ENABLE_CUDA', 'cuda'),
            self.define_from_variant('CLIO_CORE_ENABLE_ROCM', 'rocm'),
            self.define_from_variant('CLIO_CORE_ENABLE_SYCL', 'sycl'),
            self.define_from_variant('CLIO_GPU_CLANG', 'gpu_vector'),
        ]
        if '+cuda' in spec:
            archs = spec.variants['cuda_arch'].value
            if archs and archs[0] != 'none':
                args.append(self.define('CMAKE_CUDA_ARCHITECTURES',
                                        ';'.join(archs)))
            cuda_prefix = spec['cuda'].prefix
            args.append(self.define('CUDAToolkit_ROOT', cuda_prefix))
            if '+gpu_vector' in spec:
                # clang compiles the CUDA sources; point it at Spack's CUDA.
                clangxx = self._clangxx()
                args.append(self.define('CMAKE_CUDA_COMPILER', clangxx))
                args.append(self.define(
                    'CMAKE_CUDA_FLAGS',
                    '--cuda-path={0} -Wno-unknown-cuda-version'.format(
                        cuda_prefix)))
        if '+sycl' in spec:
            inc = self._sycl_include_dir()
            if inc:
                args.append(self.define('CLIO_SYCL_INCLUDE_DIR', inc))
        if '+rocm' in spec:
            args.append(self.define('ROCM_PATH', spec['hip'].prefix))
            targets = spec.variants['amdgpu_target'].value
            if targets and targets[0] != 'none':
                args.append(self.define('CMAKE_HIP_ARCHITECTURES',
                                        ';'.join(targets)))
        return args

    def _sycl_include_dir(self):
        """Directory holding sycl/sycl.hpp for the oneAPI compiler in use.

        CMake looks for it next to CMAKE_CXX_COMPILER, which under Spack is
        Spack's compiler wrapper, so name the real compiler's include dir.

        @return the include directory, or None if it cannot be found
        """
        try:
            cxx = self['cxx'].cxx
        except Exception:
            cxx = getattr(self.compiler, 'cxx', None)
        if not cxx:
            return None
        root_dir = os.path.dirname(os.path.dirname(os.path.realpath(cxx)))
        inc = join_path(root_dir, 'include')
        if os.path.exists(join_path(inc, 'sycl', 'sycl.hpp')):
            return inc
        return None

    def _clangxx(self):
        """The clang++ of the llvm dependency, for CMAKE_CUDA_COMPILER.

        An llvm registered as an external from a distribution package may
        have a prefix (/usr) whose bin/ holds only versioned names such as
        clang++-22; fall back to the newest of those.

        @return the path of the clang++ to use
        """
        bindir = self.spec['llvm'].prefix.bin
        # A distribution prefix (/usr) can hold several clangs, with the plain
        # name pointing at the system default; take the one matching llvm's
        # version first.
        major = str(self.spec['llvm'].version.up_to(1))
        matching = join_path(bindir, 'clang++-' + major)
        if os.path.exists(matching):
            return matching
        plain = join_path(bindir, 'clang++')
        if os.path.exists(plain):
            return plain
        versioned = glob.glob(join_path(bindir, 'clang++-[0-9]*'))
        versioned.sort(key=lambda p: int(p.rsplit("-", 1)[1])
                       if p.rsplit("-", 1)[1].isdigit() else -1)
        return versioned[-1] if versioned else plain

    def setup_build_environment(self, env):
        """Let the linker find libcuda on machines without a GPU driver.

        The GPU libraries link -lcuda (the driver). A build host without a
        driver (a login node, CI) has only the toolkit's stub, which carries
        the same soname, so the real driver is used at run time.

        @param env the build environment to modify
        """
        if '+cuda' in self.spec:
            for sub in ('lib64', 'lib'):
                stubs = join_path(self.spec['cuda'].prefix, sub, 'stubs')
                if os.path.isdir(stubs):
                    env.append_flags('LDFLAGS', '-L' + stubs)
                    break

    def setup_run_environment(self, env):
        # Set up library and module paths
        env.prepend_path('LD_LIBRARY_PATH', self.prefix.lib)
        env.prepend_path('CMAKE_MODULE_PATH', self.prefix.cmake)
        env.prepend_path('CMAKE_PREFIX_PATH', self.prefix.cmake)

        # Add Python paths if Python bindings are enabled
        if '+python' in self.spec:
            env.prepend_path('PYTHONPATH', self.prefix.lib)
