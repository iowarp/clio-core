cmake_minimum_required(VERSION 3.10)
project(hermes_shm LANGUAGES C CXX)
include(CTest)
# CMAKE_CXX_STANDARD is set by root CMakeLists.txt

# ------------------------------------------------------------------------------
# Global variables
# ------------------------------------------------------------------------------
set(HSHM_SHM_VERSION_MAJOR 2)
set(HSHM_SHM_VERSION_MINOR 0)
set(HSHM_SHM_VERSION_PATCH 0)
set(HSHM_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HSHM_ROOT}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_compile_definitions(_CRT_SECURE_NO_DEPRECATE)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------
# All HSHM_ENABLE_* options are now set by root CMakeLists.txt from WRP_CORE_* parameters
# CMAKE_EXPORT_COMPILE_COMMANDS is set by root CMakeLists.txt
# BUILD_SHARED_LIBS is set by root CMakeLists.txt
# HSHM_ENABLE_TESTS and HSHM_ENABLE_BENCHMARKS are set by root CMakeLists.txt
# HSHM_ENABLE_WINDOWS_THREADS, HSHM_ENABLE_PTHREADS, HSHM_DEBUG_LOCK, and HSHM_NO_COMPILE are set by root CMakeLists.txt

if(WIN32)
    message(STATUS "Detected Windows OS")
    set(HSHM_ENABLE_WINDOWS_SYSINFO ON)
    set(HSHM_ENABLE_WINDOWS_THREADS ON)
else()
    message(STATUS "Detected UNIX OS")
    set(HSHM_ENABLE_PROCFS_SYSINFO ON)
    set(HSHM_ENABLE_PTHREADS ON)
endif()

if(HSHM_NO_COMPILE)
    return()
endif()

# ------------------------------------------------------------------------------
# DOTENV
# ------------------------------------------------------------------------------
# DOTENV handling is done by root CMakeLists.txt

# ------------------------------------------------------------------------------
# Setup CMake Environment
# ------------------------------------------------------------------------------
set(HSHM_EXPORTED_TARGETS "HermesShm")
# Output directories are set by root CMakeLists.txt

# ------------------------------------------------------------------------------
# Setup install and output Directories
# ------------------------------------------------------------------------------
# Installation directories use relative paths for proper --prefix override support
# Removed HSHM_INSTALL_*_DIR macros - use lib, include, bin, share directly

# ------------------------------------------------------------------------------
# CMake Modules
# ------------------------------------------------------------------------------
# CMake modules are now loaded from /workspace/cmake via IowarpCoreCommon.cmake
# No need for local cmake directory

# ------------------------------------------------------------------------------
# Optimization
# ------------------------------------------------------------------------------

# CMAKE_POSITION_INDEPENDENT_CODE is set by root CMakeLists.txt
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" OR CMAKE_CXX_COMPILER_ID STREQUAL "MinGW" OR CMAKE_CXX_COMPILER_ID MATCHES "^Intel")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wfatal-errors")
        set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -g -O0")
        add_compile_definitions(HSHM_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -Wfatal-errors")
    endif()

    set(HSHM_COMPILER_GNU "ON")

    if(APPLE)
        set(REAL_TIME_FLAGS "-ldl")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "OpenBSD")
        set(REAL_TIME_FLAGS "")
    else()
        set(REAL_TIME_FLAGS "-lrt -ldl")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("IN DEBUG MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi /Od")
        add_compile_definitions(HSHM_DEBUG)
    else()
        message("IN RELEASE MODE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    endif()

    set(HSHM_COMPILER_MSVC "ON")
else()
    message(FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif()

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
# External libraries are now found by IowarpCoreCommon.cmake in /workspace/cmake
# No local includes needed

# CMAKE_EXPORT_COMPILE_COMMANDS and RPATH settings are set by root CMakeLists.txt

# ENABLE GPU SUPPORT
if(HSHM_ENABLE_CUDA)
    wrp_core_enable_cuda(17)
endif()

if(HSHM_ENABLE_ROCM)
    wrp_core_enable_rocm("HIP" 17)
endif()

# ------------------------------------------------------------------------------
# Modular dependency targets
# ------------------------------------------------------------------------------

# hshm::configure - Configuration parsing (yaml-cpp)
# Uses system-provided yaml-cpp (assumes yaml-cpp is installed)
add_library(configure INTERFACE)
target_link_libraries(configure INTERFACE ${YAML_CPP_LIBS})
add_library(hshm::configure ALIAS configure)

# hshm::serialize - Serialization (cereal)
add_library(serialize INTERFACE)
if(HSHM_ENABLE_CEREAL)
    target_link_libraries(serialize INTERFACE ${SERIALIZATION_LIBS})
endif()
target_compile_definitions(serialize INTERFACE
    HSHM_ENABLE_CEREAL=$<BOOL:${HSHM_ENABLE_CEREAL}>
)
add_library(hshm::serialize ALIAS serialize)

# hshm::interceptor - ELF interception support
add_library(interceptor INTERFACE)
if(HSHM_ENABLE_ELF)
    target_link_libraries(interceptor INTERFACE ${ELF_LIBS})
    target_link_directories(interceptor INTERFACE ${ELF_LIB_DIRS})
    target_include_directories(interceptor INTERFACE ${ELF_INCLUDES})
endif()
target_compile_definitions(interceptor INTERFACE
    HSHM_ENABLE_ELF=$<BOOL:${HSHM_ENABLE_ELF}>
)
add_library(hshm::interceptor ALIAS interceptor)

# hshm::lightbeam - ZeroMQ and Thallium transport
add_library(lightbeam INTERFACE)
if(HSHM_ENABLE_ZMQ)
    target_link_libraries(lightbeam INTERFACE ${ZMQ_LIBS})
    target_link_directories(lightbeam INTERFACE ${ZMQ_LIB_DIRS})
    target_include_directories(lightbeam INTERFACE ${ZMQ_INCLUDES})
endif()
if(HSHM_ENABLE_LIBFABRIC)
    target_link_libraries(lightbeam INTERFACE ${LIBFABRIC_LIBS})
    target_link_directories(lightbeam INTERFACE ${LIBFABRIC_LIB_DIRS})
    target_include_directories(lightbeam INTERFACE ${LIBFABRIC_INCLUDES})
endif()
if(HSHM_ENABLE_THALLIUM)
    # Thallium dependencies would go here
endif()
target_compile_definitions(lightbeam INTERFACE
    HSHM_ENABLE_ZMQ=$<BOOL:${HSHM_ENABLE_ZMQ}>
    HSHM_ENABLE_LIBFABRIC=$<BOOL:${HSHM_ENABLE_LIBFABRIC}>
    HSHM_ENABLE_THALLIUM=$<BOOL:${HSHM_ENABLE_THALLIUM}>
)
add_library(hshm::lightbeam ALIAS lightbeam)

# hshm::thread_all - Threading support (includes Thallium if enabled)
add_library(thread_all INTERFACE)
if(HSHM_ENABLE_OPENMP)
    target_link_libraries(thread_all INTERFACE ${OpenMP_LIBS})
endif()
if(HSHM_ENABLE_THALLIUM)
    # Thallium threading dependencies would go here
endif()
if(HSHM_ENABLE_PTHREADS)
    target_link_libraries(thread_all INTERFACE pthread)
endif()
target_compile_definitions(thread_all INTERFACE
    HSHM_ENABLE_OPENMP=$<BOOL:${HSHM_ENABLE_OPENMP}>
    HSHM_ENABLE_PTHREADS=$<BOOL:${HSHM_ENABLE_PTHREADS}>
    HSHM_ENABLE_WINDOWS_THREADS=$<BOOL:${HSHM_ENABLE_WINDOWS_THREADS}>
    HSHM_DEFAULT_THREAD_MODEL=hshm::thread::$<IF:$<BOOL:${HSHM_ENABLE_PTHREADS}>,Pthread,$<IF:$<BOOL:${HSHM_ENABLE_WINDOWS_THREADS}>,StdThread,StdThread>>
    HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::$<IF:$<BOOL:${HSHM_ENABLE_CUDA}>,Cuda,$<IF:$<BOOL:${HSHM_ENABLE_ROCM}>,Rocm,StdThread>>
)
add_library(hshm::thread_all ALIAS thread_all)

# hshm::mpi - MPI support
add_library(mpi INTERFACE)
if(HSHM_ENABLE_MPI)
    target_link_libraries(mpi INTERFACE ${MPI_LIBS})
endif()
target_compile_definitions(mpi INTERFACE
    HSHM_ENABLE_MPI=$<BOOL:${HSHM_ENABLE_MPI}>
)
add_library(hshm::mpi ALIAS mpi)

# hshm::compress - Compression support
add_library(compress INTERFACE)
if(HSHM_ENABLE_COMPRESS)
    target_link_libraries(compress INTERFACE ${COMPRESS_LIBS})
    target_link_directories(compress INTERFACE ${COMPRESS_LIB_DIRS})
    target_include_directories(compress INTERFACE ${COMPRESS_INCLUDES})
endif()
target_compile_definitions(compress INTERFACE
    HSHM_ENABLE_COMPRESS=$<BOOL:${HSHM_ENABLE_COMPRESS}>
    HSHM_HAS_LIBPRESSIO=$<BOOL:${HSHM_HAS_LIBPRESSIO}>
)
add_library(hshm::compress ALIAS compress)

# hshm::encrypt - Encryption support
add_library(encrypt INTERFACE)
if(HSHM_ENABLE_ENCRYPT)
    target_link_libraries(encrypt INTERFACE ${ENCRYPT_LIBS})
    target_link_directories(encrypt INTERFACE ${ENCRYPT_LIB_DIRS})
    target_include_directories(encrypt INTERFACE ${ENCRYPT_INCLUDES})
endif()
target_compile_definitions(encrypt INTERFACE
    HSHM_ENABLE_ENCRYPT=$<BOOL:${HSHM_ENABLE_ENCRYPT}>
)
add_library(hshm::encrypt ALIAS encrypt)

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
    encrypt
    ${REAL_TIME_FLAGS}
)

# -----------------------------------------------------------------------------
# Documentation
# -----------------------------------------------------------------------------
if(HSHM_ENABLE_DOXYGEN)
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

# ------------------------------------------------------------------------------
# Code Coverage
# ------------------------------------------------------------------------------
if(NOT IS_HSHM_MAIN)
    if(HSHM_ENABLE_COVERAGE)
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
# CMake function for target compile definitions
# ------------------------------------------------------------------------------
function(hshm_target_compile_definitions target)
    get_target_property(target_type ${target} TYPE)

    # Core definitions for all HSHM targets
    # Module-specific definitions (MPI, ZMQ, etc.) are now handled by modular targets
    set(common_definitions
        HSHM_COMPILER_MSVC=$<BOOL:${HSHM_COMPILER_MSVC}>
        HSHM_COMPILER_GNU=$<BOOL:${HSHM_COMPILER_GNU}>
        HSHM_ENABLE_WINDOWS_SYSINFO=$<BOOL:${HSHM_ENABLE_WINDOWS_SYSINFO}>
        HSHM_ENABLE_PROCFS_SYSINFO=$<BOOL:${HSHM_ENABLE_PROCFS_SYSINFO}>
        HSHM_ENABLE_COVERAGE=$<BOOL:${HSHM_ENABLE_COVERAGE}>
        HSHM_ENABLE_DOXYGEN=$<BOOL:${HSHM_ENABLE_DOXYGEN}>
        HSHM_DEBUG_LOCK=$<BOOL:${HSHM_DEBUG_LOCK}>
        HSHM_DEFAULT_ALLOC_T=hipc::ThreadLocalAllocator
        HSHM_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
        HSHM_LOG_LEVEL=${HSHM_LOG_LEVEL}
    )

    # Add CUDA/ROCM definitions for all targets
    # Host targets get 0, GPU targets get 1 (set explicitly in their target definitions)
    if(target STREQUAL "hermes_shm_host" OR target STREQUAL "cxx")
        # Host-only targets: explicitly disable GPU support
        list(APPEND common_definitions
            HSHM_ENABLE_CUDA=0
            HSHM_ENABLE_ROCM=0
        )
    else()
        # GPU and other targets: use CMake variable values
        list(APPEND common_definitions
            HSHM_ENABLE_CUDA=$<BOOL:${HSHM_ENABLE_CUDA}>
            HSHM_ENABLE_ROCM=$<BOOL:${HSHM_ENABLE_ROCM}>
        )
    endif()

    if(target_type STREQUAL "INTERFACE_LIBRARY")
        target_compile_definitions(${target} INTERFACE ${common_definitions})
    else()
        target_compile_definitions(${target} PUBLIC ${common_definitions})
        target_compile_definitions(${target} PRIVATE
            HSHM_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
        )
    endif()
endfunction()

# ------------------------------------------------------------------------------
# Build hermes_shm
# ------------------------------------------------------------------------------
add_subdirectory(src)

if(NOT IS_HSHM_MAIN)
    add_custom_target(lint COMMAND bash ${HSHM_ROOT}/scripts/lint.sh ${HSHM_ROOT})
    add_custom_target(preamble COMMAND python3 ${HSHM_ROOT}/scripts/preamble.py ${HSHM_ROOT})
endif()

# ------------------------------------------------------------------------------
# Build tests + benchmarks
# ------------------------------------------------------------------------------
set(TEST_MAIN ${HSHM_ROOT}/test/unit)
# enable_testing() is handled by root CMakeLists.txt

if(HSHM_ENABLE_TESTS)
    message("Building HSHM unit tests")
    add_subdirectory(test)
endif()

if(HSHM_ENABLE_BENCHMARKS)
    message("Building HSHM benchmarks")
    add_subdirectory(benchmark)
endif()

# add_subdirectory(example)

# ------------------------------------------------------------------------------
# Install hshm
# ------------------------------------------------------------------------------
install(DIRECTORY include/ DESTINATION include)

# Component config files are no longer installed separately
# All configuration is handled by unified IOWarp Core (IowarpCoreConfig.cmake)

install(EXPORT ${HSHM_EXPORTED_TARGETS}
    FILE ${HSHM_EXPORTED_TARGETS}CoreConfig.cmake
    NAMESPACE hshm::
    DESTINATION lib/cmake/HermesShm
)

if(WRP_CORE_ENABLE_JARVIS)
    jarvis_repo_add("${HSHM_ROOT}/test/jarvis_hshm")
endif()
