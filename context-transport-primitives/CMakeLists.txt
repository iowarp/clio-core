cmake_minimum_required(VERSION 3.10)
project(clio_ctp LANGUAGES C CXX)
include(CTest)
# CMAKE_CXX_STANDARD is set by root CMakeLists.txt

# Normally inherited from the root project; define a fallback for standalone
# CTP configures so MSVC-frontend compilers (cl, clang-cl, icx.exe) take the
# MSVC-style flag path below (see issue #555).
if(NOT DEFINED CLIO_CORE_MSVC_LIKE)
    if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"))
        set(CLIO_CORE_MSVC_LIKE ON)
    else()
        set(CLIO_CORE_MSVC_LIKE OFF)
    endif()
endif()

# ------------------------------------------------------------------------------
# Global variables
# ------------------------------------------------------------------------------
set(CLIO_CTP_SHM_VERSION_MAJOR 2)
set(CLIO_CTP_SHM_VERSION_MINOR 0)
set(CLIO_CTP_SHM_VERSION_PATCH 0)
set(CLIO_CTP_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CLIO_CTP_ROOT}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
# All CTP_ENABLE_* options are now set by root CMakeLists.txt from CLIO_CORE_* parameters
# CMAKE_EXPORT_COMPILE_COMMANDS is set by root CMakeLists.txt
# BUILD_SHARED_LIBS is set by root CMakeLists.txt
# CLIO_CORE_ENABLE_TESTS is set by root CMakeLists.txt
# Benchmarks are controlled by CLIO_CORE_ENABLE_BENCHMARKS
# CTP_ENABLE_WINDOWS_THREADS, CTP_ENABLE_PTHREADS, CTP_DEBUG_LOCK, and CTP_NO_COMPILE are set by root CMakeLists.txt

if(WIN32)
    message(STATUS "Detected Windows OS")
    set(CLIO_CTP_ENABLE_WINDOWS_SYSINFO ON)
    set(CLIO_CTP_ENABLE_WINDOWS_THREADS ON)
else()
    message(STATUS "Detected UNIX OS")
    set(CLIO_CTP_ENABLE_PROCFS_SYSINFO ON)
    set(CLIO_CTP_ENABLE_PTHREADS ON)
endif()

if(CLIO_CTP_NO_COMPILE)
    return()
endif()

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
# DOTENV handling is done by root CMakeLists.txt

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(CLIO_CTP_EXPORTED_TARGETS "ClioCtp")
# Output directories are set by root CMakeLists.txt

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
# Installation directories use relative paths for proper --prefix override support
# Removed CTP_INSTALL_*_DIR macros - use lib, include, bin, share directly

# ------------------------------------------------------------------------------
# CMake Modules
# ------------------------------------------------------------------------------
# CMake modules are now loaded from /workspace/cmake via ClioCoreCommon.cmake
# No need for local cmake directory

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# CMAKE_POSITION_INDEPENDENT_CODE is set by root CMakeLists.txt
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "MinGW" OR CMAKE_CXX_COMPILER_ID MATCHES "^Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC") AND NOT CLIO_CORE_MSVC_LIKE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wfatal-errors")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
        add_compile_definitions(CTP_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wfatal-errors")
    endif()

    set(CLIO_CTP_COMPILER_GNU "ON")

    if(WIN32)
        # Clang/IntelLLVM (icx) targeting Windows: rt/dl are POSIX-only. Use
        # the Winsock import library like the MSVC branch below.
        set(REAL_TIME_FLAGS "ws2_32")
    elseif(APPLE)
        set(REAL_TIME_FLAGS "-ldl")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
        set(REAL_TIME_FLAGS "")
    else()
        set(REAL_TIME_FLAGS "-lrt -ldl")
    endif()
elseif(CLIO_CORE_MSVC_LIKE)
    # MSVC, clang-cl, or icx.exe on Windows: MSVC-style command line.
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /Od")
        add_compile_definitions(CTP_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    endif()

    set(CLIO_CTP_COMPILER_MSVC "ON")
    set(REAL_TIME_FLAGS "ws2_32")
else()
    # Treat unrecognized compilers (e.g. NVHPC on first cmake run) as GNU-compatible
    message(WARNING "Unrecognized compiler '${CMAKE_CXX_COMPILER_ID}', assuming GNU-compatible flags")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
    set(CLIO_CTP_COMPILER_GNU "ON")
    set(REAL_TIME_FLAGS "-lrt -ldl")
endif()

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
# External libraries are now found by ClioCoreCommon.cmake in /workspace/cmake
# No local includes needed

# CMAKE_EXPORT_COMPILE_COMMANDS and RPATH settings are set by root CMakeLists.txt

# ENABLE GPU SUPPORT
if(CLIO_CORE_ENABLE_CUDA)
    wrp_core_enable_cuda(20)
endif()

if(CLIO_CORE_ENABLE_ROCM)
    wrp_core_enable_rocm("HIP" 17)
endif()

if(CLIO_CORE_ENABLE_SYCL)
    wrp_core_enable_sycl(17)
endif()

# ------------------------------------------------------------------------------
# Modular dependency targets
# ------------------------------------------------------------------------------

# ctp::configure - Configuration parsing (yaml-cpp)
# Uses system-provided yaml-cpp (assumes yaml-cpp is installed)
add_library(configure INTERFACE)
target_link_libraries(configure INTERFACE ${YAML_CPP_LIBS})
add_library(ctp::configure ALIAS configure)

# ctp::serialize - Serialization (cereal + msgpack)
add_library(serialize INTERFACE)
target_link_libraries(serialize INTERFACE ${SERIALIZATION_LIBS})
target_compile_definitions(serialize INTERFACE
    CTP_ENABLE_CEREAL=$<BOOL:${CLIO_CORE_ENABLE_CEREAL}>
)
add_library(ctp::serialize ALIAS serialize)

# ctp::interceptor - ELF interception support
add_library(interceptor INTERFACE)
if(CLIO_CORE_ENABLE_ELF)
    target_link_libraries(interceptor INTERFACE ${ELF_LIBS})
    target_link_directories(interceptor INTERFACE ${ELF_LIB_DIRS})
    target_include_directories(interceptor INTERFACE ${ELF_INCLUDES})
endif()
target_compile_definitions(interceptor INTERFACE
    CTP_ENABLE_ELF=$<BOOL:${CLIO_CORE_ENABLE_ELF}>
)
add_library(ctp::interceptor ALIAS interceptor)

# ctp::lightbeam - ZeroMQ and Thallium transport
add_library(lightbeam INTERFACE)
if(CLIO_CORE_ENABLE_ZMQ)
    target_link_libraries(lightbeam INTERFACE ${ZMQ_LIBS})
    target_link_directories(lightbeam INTERFACE ${ZMQ_LIB_DIRS})
    target_include_directories(lightbeam INTERFACE ${ZMQ_INCLUDES})
endif()
if(CLIO_CORE_ENABLE_LIBFABRIC)
    target_link_libraries(lightbeam INTERFACE ${LIBFABRIC_LIBS})
    target_link_directories(lightbeam INTERFACE ${LIBFABRIC_LIB_DIRS})
    target_include_directories(lightbeam INTERFACE ${LIBFABRIC_INCLUDES})
endif()
if(CLIO_CORE_ENABLE_THALLIUM)
    target_link_libraries(lightbeam INTERFACE thallium)
endif()
if(CLIO_CORE_ENABLE_NIXL)
    find_library(NIXL_LIB NAMES nixl PATHS /usr/local/lib /usr/local/lib/x86_64-linux-gnu
                 /usr/lib /usr/lib/x86_64-linux-gnu)
    find_library(NIXL_BUILD_LIB NAMES nixl_build PATHS /usr/local/lib /usr/local/lib/x86_64-linux-gnu
                 /usr/lib /usr/lib/x86_64-linux-gnu)
    find_path(NIXL_INCLUDE NAMES nixl.h PATHS /usr/local/include /usr/include)
    if(NIXL_LIB AND NIXL_INCLUDE)
        message(STATUS "Found NIXL: lib=${NIXL_LIB} include=${NIXL_INCLUDE}")
        target_link_libraries(lightbeam INTERFACE ${NIXL_LIB})
        if(NIXL_BUILD_LIB)
            target_link_libraries(lightbeam INTERFACE ${NIXL_BUILD_LIB})
        endif()
        target_include_directories(lightbeam INTERFACE ${NIXL_INCLUDE})
    else()
        message(WARNING "NIXL requested (CLIO_CORE_ENABLE_NIXL=ON) but not found; disabling")
        set(CLIO_CORE_ENABLE_NIXL OFF)
        set(CLIO_CTP_ENABLE_NIXL OFF CACHE BOOL "" FORCE)
    endif()
endif()
if(WIN32)
    target_link_libraries(lightbeam INTERFACE ws2_32)
endif()
target_compile_definitions(lightbeam INTERFACE
    CTP_ENABLE_LIGHTBEAM=1
    CTP_ENABLE_ZMQ=$<BOOL:${CLIO_CORE_ENABLE_ZMQ}>
    CTP_ENABLE_LIBFABRIC=$<BOOL:${CLIO_CORE_ENABLE_LIBFABRIC}>
    CTP_ENABLE_THALLIUM=$<BOOL:${CLIO_CORE_ENABLE_THALLIUM}>
    CTP_ENABLE_NIXL=$<BOOL:${CLIO_CORE_ENABLE_NIXL}>
    CTP_ENABLE_NVSHMEM=$<BOOL:${CLIO_CORE_ENABLE_NVSHMEM}>
)
add_library(ctp::lightbeam ALIAS lightbeam)

# ctp::thread_all - Threading support (includes Thallium if enabled)
add_library(thread_all INTERFACE)
if(CLIO_CORE_ENABLE_OPENMP)
    target_link_libraries(thread_all INTERFACE ${OpenMP_LIBS})
endif()
if(CLIO_CORE_ENABLE_THALLIUM)
    # Thallium threading dependencies would go here
endif()
if(CLIO_CTP_ENABLE_PTHREADS)
    target_link_libraries(thread_all INTERFACE pthread)
endif()
target_compile_definitions(thread_all INTERFACE
    CTP_ENABLE_OPENMP=$<BOOL:${CLIO_CORE_ENABLE_OPENMP}>
    CTP_ENABLE_PTHREADS=$<BOOL:${CLIO_CTP_ENABLE_PTHREADS}>
    CTP_ENABLE_WINDOWS_THREADS=$<BOOL:${CLIO_CTP_ENABLE_WINDOWS_THREADS}>
    CTP_DEFAULT_THREAD_MODEL=ctp::thread::$<IF:$<BOOL:${CLIO_CTP_ENABLE_PTHREADS}>,Pthread,$<IF:$<BOOL:${CLIO_CTP_ENABLE_WINDOWS_THREADS}>,StdThread,StdThread>>
    CTP_DEFAULT_THREAD_MODEL_GPU=ctp::thread::$<IF:$<BOOL:${CLIO_CORE_ENABLE_CUDA}>,Cuda,$<IF:$<BOOL:${CLIO_CORE_ENABLE_ROCM}>,Rocm,StdThread>>
)
add_library(ctp::thread_all ALIAS thread_all)

# ctp::mpi - MPI support
add_library(mpi INTERFACE)
if(CLIO_CORE_ENABLE_MPI)
    target_link_libraries(mpi INTERFACE ${MPI_LIBS})
endif()
target_compile_definitions(mpi INTERFACE
    CTP_ENABLE_MPI=$<BOOL:${CLIO_CORE_ENABLE_MPI}>
)
add_library(ctp::mpi ALIAS mpi)

# ------------------------------------------------------------------------------
# CMake function for target compile definitions
# (must be defined before add_subdirectory(src) which uses it)
# ------------------------------------------------------------------------------
function(hshm_target_compile_definitions target)
    get_target_property(target_type ${target} TYPE)

    # Core definitions for all CTP targets
    # Module-specific definitions (MPI, ZMQ, etc.) are now handled by modular targets
    set(common_definitions
        CTP_COMPILER_MSVC=$<BOOL:${CLIO_CTP_COMPILER_MSVC}>
        CTP_COMPILER_GNU=$<BOOL:${CLIO_CTP_COMPILER_GNU}>
        CTP_ENABLE_WINDOWS_SYSINFO=$<BOOL:${CLIO_CTP_ENABLE_WINDOWS_SYSINFO}>
        CTP_ENABLE_PROCFS_SYSINFO=$<BOOL:${CLIO_CTP_ENABLE_PROCFS_SYSINFO}>
        CTP_ENABLE_DOXYGEN=$<BOOL:${CLIO_CORE_ENABLE_DOXYGEN}>
        CTP_DEBUG_LOCK=$<BOOL:${CLIO_CTP_DEBUG_LOCK}>
        CTP_DEFAULT_ALLOC_T=ctp::ipc::ThreadLocalAllocator
        CTP_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
        CTP_LOG_LEVEL=${CLIO_CTP_LOG_LEVEL}
    )

    # Add CUDA/ROCM definitions for GPU targets only.
    # Host targets (clio_ctp_host, cxx) do NOT propagate CTP_ENABLE_CUDA
    # to avoid conflicts when a consumer links both host and GPU libraries.
    # Undefined CTP_ENABLE_CUDA evaluates to 0 in #if directives, which is
    # the correct behavior for host-only consumers.
    if(NOT (target STREQUAL "clio_ctp_host" OR target STREQUAL "cxx"))
        list(APPEND common_definitions
            CTP_ENABLE_CUDA=$<BOOL:${CLIO_CORE_ENABLE_CUDA}>
            CTP_ENABLE_ROCM=$<BOOL:${CLIO_CORE_ENABLE_ROCM}>
        )
    endif()

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        target_compile_definitions(${target} INTERFACE ${common_definitions})
    else()
        target_compile_definitions(${target} PUBLIC ${common_definitions})
        target_compile_definitions(${target} PRIVATE
            CTP_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
        )
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Code Coverage (must be defined before add_subdirectory(src) which uses it)
# ------------------------------------------------------------------------------
if(NOT IS_HSHM_MAIN)
    if(CLIO_CTP_ENABLE_COVERAGE)
        set(COVERAGE_FLAGS "-fprofile-arcs -ftest-coverage" CACHE STRING
            "Flags to the coverage program to perform coverage inspection")
        mark_as_advanced(COVERAGE_FLAGS)

        macro(set_coverage_flags target)
            set_target_properties(${target}
                PROPERTIES
                COMPILE_FLAGS ${COVERAGE_FLAGS}
                LINK_FLAGS ${COVERAGE_FLAGS}
            )
        endmacro()
    endif()
endif()

# ------------------------------------------------------------------------------
# Build clio_ctp source libraries (must be before ctp::compress so that
# qtable_predictor and dense_nn_predictor targets are available)
# ------------------------------------------------------------------------------
add_subdirectory(src)

# ctp::compress - Compression support
add_library(compress INTERFACE)
if(CLIO_CTE_ENABLE_COMPRESS)
    target_link_libraries(compress INTERFACE ${COMPRESS_LIBS})
    target_link_directories(compress INTERFACE ${COMPRESS_LIB_DIRS})
    target_include_directories(compress INTERFACE ${COMPRESS_INCLUDES})
    # Link qtable_predictor if it was built
    if(TARGET qtable_predictor)
        target_link_libraries(compress INTERFACE qtable_predictor)
    endif()
    # Link dense_nn_predictor if it was built (requires Eigen3)
    if(TARGET dense_nn_predictor)
        target_link_libraries(compress INTERFACE dense_nn_predictor)
    endif()
    # nvcomp.h includes <cuda_runtime.h>; the nvcomp imported target does not
    # propagate the CUDA toolkit include dir, so link CUDA::cudart to provide it
    # to every consumer of ctp::compress (test + CTE compressor module).
    if(CLIO_CTP_ENABLE_NVCOMP AND TARGET CUDA::cudart)
        target_link_libraries(compress INTERFACE CUDA::cudart)
    endif()
endif()
target_compile_definitions(compress INTERFACE
    CTP_ENABLE_COMPRESS=$<BOOL:${CLIO_CTE_ENABLE_COMPRESS}>
    CTP_ENABLE_LIBPRESSIO=$<BOOL:${CLIO_CTP_ENABLE_LIBPRESSIO}>
    CTP_ENABLE_NVCOMP=$<BOOL:${CLIO_CTP_ENABLE_NVCOMP}>
    CTP_ENABLE_ZFP_SYCL=$<BOOL:${CLIO_CTP_ENABLE_ZFP_SYCL}>
)
add_library(ctp::compress ALIAS compress)

# ctp::aio - Async I/O support
add_library(aio INTERFACE)
set(CLIO_CTP_ENABLE_LIBAIO OFF)
if(CLIO_CTP_ENABLE_IO_URING)
    # Verify liburing is actually available before enabling
    find_path(URING_INCLUDE_DIR NAMES liburing.h)
    find_library(URING_LIBRARY NAMES uring)
    if(URING_INCLUDE_DIR AND URING_LIBRARY)
        # Propagate the liburing header dir to consumers. Without this,
        # any target that includes iouring_io.h (-> #include <liburing.h>)
        # only compiles when liburing.h happens to sit on a default system
        # include path. On hosts where liburing lives only in a non-default
        # prefix (e.g. the conda env, with CLIO_CORE_ENABLE_CONDA=OFF) the
        # io_uring unit tests fail with "liburing.h: No such file or
        # directory". Mirrors the libaio branch below.
        target_include_directories(aio INTERFACE ${URING_INCLUDE_DIR})
        if(CLIO_CORE_STATIC_DEPS)
            find_library(URING_STATIC_LIB NAMES liburing.a REQUIRED)
            message(STATUS "Static liburing: ${URING_STATIC_LIB}")
            target_link_libraries(aio INTERFACE ${URING_STATIC_LIB})
        else()
            target_link_libraries(aio INTERFACE ${URING_LIBRARY})
        endif()
        message(STATUS "io_uring enabled: ${URING_INCLUDE_DIR} / ${URING_LIBRARY}")
    else()
        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            # On Linux a missing liburing is actionable (install liburing-dev),
            # so surface it as a warning.
            message(WARNING "io_uring requested but liburing not found, disabling io_uring")
        else()
            # io_uring is a Linux-only kernel interface and can never be present
            # on macOS/Windows/BSD, so this is expected, not a misconfiguration.
            message(STATUS "io_uring unavailable on ${CMAKE_SYSTEM_NAME}, disabling io_uring")
        endif()
        set(CLIO_CTP_ENABLE_IO_URING OFF)
    endif()
endif()
if(NOT WIN32)
    if(NOT CLIO_CTP_ENABLE_IO_URING)
        find_path(LIBAIO_INCLUDE_DIR NAMES libaio.h)
        find_library(LIBAIO_LIBRARY NAMES aio libaio)
        if(LIBAIO_LIBRARY AND LIBAIO_INCLUDE_DIR)
            target_include_directories(aio INTERFACE ${LIBAIO_INCLUDE_DIR})
            target_link_libraries(aio INTERFACE ${LIBAIO_LIBRARY})
            set(CLIO_CTP_ENABLE_LIBAIO ON)
        endif()
    endif()
    # librt holds POSIX realtime (aio_*, shm_*, clock_gettime) on Linux/BSD,
    # but on macOS these live in libSystem and `-lrt` resolves to nothing.
    if(NOT APPLE)
        target_link_libraries(aio INTERFACE rt)
    endif()
endif()
target_compile_definitions(aio INTERFACE
    CTP_ENABLE_IO_URING=$<BOOL:${CLIO_CTP_ENABLE_IO_URING}>
    CTP_ENABLE_LIBAIO=$<BOOL:${CLIO_CTP_ENABLE_LIBAIO}>
)
add_library(ctp::aio ALIAS aio)

# ctp::encrypt - Encryption support
add_library(encrypt INTERFACE)
if(CLIO_CORE_ENABLE_ENCRYPT)
    target_link_libraries(encrypt INTERFACE ${ENCRYPT_LIBS})
    target_link_directories(encrypt INTERFACE ${ENCRYPT_LIB_DIRS})
    target_include_directories(encrypt INTERFACE ${ENCRYPT_INCLUDES})
endif()
target_compile_definitions(encrypt INTERFACE
    CTP_ENABLE_ENCRYPT=$<BOOL:${CLIO_CORE_ENABLE_ENCRYPT}>
)
add_library(ctp::encrypt ALIAS encrypt)

# ctp::nixl - NIXL (NVIDIA Inference Xfer Library) transport
add_library(nixl_transport INTERFACE)
if(CLIO_CORE_ENABLE_NIXL AND NIXL_LIB AND NIXL_INCLUDE)
    target_link_libraries(nixl_transport INTERFACE ${NIXL_LIB})
    if(NIXL_BUILD_LIB)
        target_link_libraries(nixl_transport INTERFACE ${NIXL_BUILD_LIB})
    endif()
    target_include_directories(nixl_transport INTERFACE ${NIXL_INCLUDE})
endif()
target_compile_definitions(nixl_transport INTERFACE
    CTP_ENABLE_NIXL=$<BOOL:${CLIO_CORE_ENABLE_NIXL}>
)
add_library(ctp::nixl ALIAS nixl_transport)

# ctp::nvshmem - NVSHMEM GPU-to-GPU communication
add_library(nvshmem_transport INTERFACE)
target_compile_definitions(nvshmem_transport INTERFACE
    CTP_ENABLE_NVSHMEM=$<BOOL:${CLIO_CORE_ENABLE_NVSHMEM}>
)
add_library(ctp::nvshmem ALIAS nvshmem_transport)

# Legacy host_deps for backwards compatibility (will be removed)
add_library(host_deps INTERFACE)
target_link_libraries(host_deps INTERFACE
    configure
    serialize
    interceptor
    lightbeam
    thread_all
    mpi
    compress
    aio
    encrypt
    ${REAL_TIME_FLAGS}
)

# Install interface libraries (compiled libraries are installed in src/CMakeLists.txt)
install(TARGETS
    configure
    serialize
    interceptor
    lightbeam
    thread_all
    mpi
    compress
    aio
    encrypt
    nixl_transport
    nvshmem_transport
    host_deps
    EXPORT
    ${CLIO_CTP_EXPORTED_TARGETS}
)

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
if(CLIO_CORE_ENABLE_DOXYGEN)
    add_doxygen_doc(
        BUILD_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/_build
        DOXY_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in
        TARGET_NAME
        dox
        COMMENT
        "HTML documentation"
    )
endif()

if(NOT IS_HSHM_MAIN)
    add_custom_target(lint COMMAND bash ${CLIO_CTP_ROOT}/scripts/lint.sh ${CLIO_CTP_ROOT})
    add_custom_target(preamble COMMAND python3 ${CLIO_CTP_ROOT}/scripts/preamble.py ${CLIO_CTP_ROOT})
endif()

# ------------------------------------------------------------------------------
# Build tests + benchmarks
# ------------------------------------------------------------------------------
set(TEST_MAIN ${CLIO_CTP_ROOT}/test/unit)
# enable_testing() is handled by root CMakeLists.txt

if(CLIO_CORE_ENABLE_TESTS)
    message("Building CTP unit tests")
    add_subdirectory(test)
endif()

if(CLIO_CORE_ENABLE_BENCHMARKS)
    message("Building CTP benchmarks")
    add_subdirectory(benchmark)
endif()

# add_subdirectory(example)

# ------------------------------------------------------------------------------
# Install hshm
# ------------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)

# Component config files are no longer installed separately
# All configuration is handled by unified IOWarp Core (IowarpCoreConfig.cmake)

install(EXPORT ${CLIO_CTP_EXPORTED_TARGETS}
    FILE ${CLIO_CTP_EXPORTED_TARGETS}CoreConfig.cmake
    NAMESPACE ctp::
    DESTINATION lib/cmake/ClioCtp
)

if(CLIO_CORE_ENABLE_JARVIS)
    jarvis_repo_add("${CTP_ROOT}/test/jarvis_hshm")
endif()
