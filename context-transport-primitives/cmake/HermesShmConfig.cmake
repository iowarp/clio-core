# Find HermesShm header and library.
#

# This module defines the following uncached variables:
# HermesShm_FOUND, if false, do not try to use hermes_shm.
# HermesShm_LIBRARY_DIRS, the directory where the hermes_shm library is found.

# -----------------------------------------------------------------------------
# Define constants
# -----------------------------------------------------------------------------
set(HSHM_SHM_VERSION_MAJOR @HSHM_SHM_VERSION_MAJOR@)
set(HSHM_SHM_VERSION_MINOR @HSHM_SHM_VERSION_MINOR@)
set(HSHM_SHM_VERSION_PATCH @HSHM_SHM_VERSION_PATCH@)

set(HSHM_ENABLE_CMAKE_DOTENV @HSHM_ENABLE_CMAKE_DOTENV@)
set(HSHM_ENABLE_MPI @HSHM_ENABLE_MPI@)
set(HSHM_ENABLE_THALLIUM @HSHM_ENABLE_THALLIUM@)
set(HSHM_ENABLE_OPENMP @HSHM_ENABLE_OPENMP@)
set(HSHM_ENABLE_CEREAL @HSHM_ENABLE_CEREAL@)
set(HSHM_ENABLE_COVERAGE @HSHM_ENABLE_COVERAGE@)
set(HSHM_ENABLE_DOXYGEN @HSHM_ENABLE_DOXYGEN@)
set(HSHM_ENABLE_WINDOWS_THREADS @HSHM_ENABLE_WINDOWS_THREADS@)
set(HSHM_ENABLE_PTHREADS @HSHM_ENABLE_PTHREADS@)
set(HSHM_DEBUG_LOCK @HSHM_DEBUG_LOCK@)
set(HSHM_ENABLE_COMPRESS @HSHM_ENABLE_COMPRESS@)
set(HSHM_ENABLE_ENCRYPT @HSHM_ENABLE_ENCRYPT@)
set(HSHM_ENABLE_ELF @HSHM_ENABLE_ELF@)
set(HSHM_ENABLE_CUDA @HSHM_ENABLE_CUDA@)
set(HSHM_ENABLE_ROCM @HSHM_ENABLE_ROCM@)
set(HIP_PLATFORM @HIP_PLATFORM@)
set(HSHM_NO_COMPILE @HSHM_NO_COMPILE@)

set(HSHM_PREFIX @CMAKE_INSTALL_PREFIX@)
set(HSHM_LIB_DIR @HSHM_INSTALL_LIB_DIR@)
set(HSHM_INCLUDE_DIR @HSHM_INSTALL_INCLUDE_DIR@)
set(HSHM_BIN_DIR @HSHM_INSTALL_BIN_DIR@)

set(HermesShm_PREFIX ${HSHM_PREFIX})
set(HermesShm_LIB_DIR ${HSHM_LIB_DIR})
set(HermesShm_INCLUDE_DIR ${HSHM_INCLUDE_DIR})
set(HermesShm_BIN_DIR ${HSHM_BIN_DIR})

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES @CMAKE_CUDA_ARCHITECTURES@)
endif()

set(REAL_TIME_FLAGS @REAL_TIME_FLAGS@)

# Find the HermesShm Package
include(@CMAKE_INSTALL_PREFIX@/lib/cmake/HermesShm/HermesShmCoreConfig.cmake)
include(@CMAKE_INSTALL_PREFIX@/lib/cmake/HermesShm/HermesShmCommonConfig.cmake)
include_directories(${HSHM_INCLUDE_DIR})
link_directories(${HSHM_LIB_DIR})

# Add my library to RPATH
list(APPEND CMAKE_INSTALL_RPATH "@HSHM_INSTALL_LIB_DIR@")

# ROCm: Target link directories / includes
if(HSHM_ENABLE_ROCM)
    execute_process(COMMAND hipconfig --rocmpath
        OUTPUT_VARIABLE rocm_path)
    message(STATUS "ROCm SDK path: ${rocm_path}")

    # TODO(llogan): This is a hack to make vscode detect HIP headers and not show errors
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem ${rocm_path}/include -D__HIP_PLATFORM_AMD__")
    set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -isystem @HSHM_INSTALL_INCLUDE_DIR@")
endif()
