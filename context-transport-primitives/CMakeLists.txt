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
# CMAKE_EXPORT_COMPILE_COMMANDS is set by root CMakeLists.txt
# BUILD_SHARED_LIBS is set by root CMakeLists.txt
# HSHM_ENABLE_TESTS and HSHM_ENABLE_BENCHMARKS are set by root CMakeLists.txt

option(HSHM_ENABLE_MPI "Enable MPI support" OFF)
option(HSHM_ENABLE_ZMQ "Enable ZeroMQ transport" ON)
option(HSHM_ENABLE_LIBFABRIC "Enable Libfabric transport" OFF)
option(HSHM_ENABLE_THALLIUM "Build tests which depend on thallium" OFF)
option(HSHM_ENABLE_OPENMP "Enable the use of OpenMP" OFF)
option(HSHM_ENABLE_CEREAL "Enable serialization using cereal" ON)
# Coverage is controlled by WRP_CORE_ENABLE_COVERAGE in root CMakeLists.txt
# Set local variable for backward compatibility with existing checks
if(DEFINED WRP_CORE_ENABLE_COVERAGE)
  set(HSHM_ENABLE_COVERAGE ${WRP_CORE_ENABLE_COVERAGE})
else()
  set(HSHM_ENABLE_COVERAGE OFF)
endif()
option(HSHM_ENABLE_DOXYGEN "Check how well the code is documented" OFF)

option(HSHM_ENABLE_WINDOWS_THREADS "Support spawning windows threads" OFF)
option(HSHM_ENABLE_PTHREADS "Support spawning pthreads" OFF)
option(HSHM_DEBUG_LOCK "Used for debugging locks" OFF)
option(HSHM_ENABLE_COMPRESS "Enable compression" OFF)
option(HSHM_ENABLE_ENCRYPT "Enable encryption" OFF)
option(HSHM_ENABLE_ELF "Enable elf" OFF)
option(HSHM_ENABLE_CUDA "Enable CUDA support" OFF)
option(HSHM_ENABLE_ROCM "Enable ROCm support" OFF)
option(HSHM_NO_COMPILE "Disable compiling / installing this library" OFF)

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
if(NOT HSHM_INSTALL_BIN_DIR)
    set(HSHM_INSTALL_BIN_DIR ${CMAKE_INSTALL_PREFIX}/bin)
endif()

if(NOT HSHM_INSTALL_LIB_DIR)
    set(HSHM_INSTALL_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
endif()

if(NOT HSHM_INSTALL_INCLUDE_DIR)
    set(HSHM_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
endif()

if(NOT HSHM_INSTALL_DATA_DIR)
    set(HSHM_INSTALL_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share)
endif()

# ------------------------------------------------------------------------------
# CMake Modules
# ------------------------------------------------------------------------------
if(NOT IS_HSHM_MAIN)
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
    message(${CMAKE_MODULE_PATH})
endif()

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
include(cmake/HermesShmCommonConfig.cmake)

# CMAKE_EXPORT_COMPILE_COMMANDS and RPATH settings are set by root CMakeLists.txt
list(APPEND CMAKE_INSTALL_RPATH "${HSHM_INSTALL_LIB_DIR}")

# ENABLE GPU SUPPORT
if(HSHM_ENABLE_CUDA)
    hshm_enable_cuda(17)
endif()

if(HSHM_ENABLE_ROCM)
    hshm_enable_rocm("HIP" 17)
endif()

# HOST DEPENDENCIES
add_library(host_deps INTERFACE)

target_link_libraries(host_deps INTERFACE
    ${YAML_CPP_LIBS}
    ${REAL_TIME_FLAGS}
    ${SERIALIZATION_LIBS}
    ${COMPRESS_LIBS}
    ${ENCRYPT_LIBS}
    ${Boost_LIBRARIES}
    ${ELF_LIBS}
    ${ZMQ_LIBS}
    ${LIBFABRIC_LIBS}
    ${MPI_LIBS}
    ${OpenMP_LIBS}
)
target_link_directories(host_deps INTERFACE
    ${COMPRESS_LIB_DIRS} ${ENCRYPT_LIB_DIRS} ${Boost_LIBRARY_DIRS}
    ${ELF_LIB_DIRS} ${YAML_CPP_LIBRARY_DIR} ${ZMQ_LIB_DIRS} ${LIBFABRIC_LIB_DIRS})
target_include_directories(host_deps INTERFACE
    ${COMPRESS_INCLUDES} ${ENCRYPT_INCLUDES} ${Boost_INCLUDE_DIRS}
    ${ELF_INCLUDES} ${YAML_CPP_INCLUDE_DIR} ${ZMQ_INCLUDES} ${LIBFABRIC_INCLUDES})

if(HSHM_ENABLE_PTHREADS)
    target_link_libraries(host_deps INTERFACE pthread)
endif()

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
    
    # Common definitions for all target types
    set(common_definitions
        HSHM_COMPILER_MSVC=$<BOOL:${HSHM_COMPILER_MSVC}>
        HSHM_COMPILER_GNU=$<BOOL:${HSHM_COMPILER_GNU}>
        HSHM_ENABLE_MPI=$<BOOL:${HSHM_ENABLE_MPI}>
        HSHM_ENABLE_ZMQ=$<BOOL:${HSHM_ENABLE_ZMQ}>
        HSHM_ENABLE_LIBFABRIC=$<BOOL:${HSHM_ENABLE_LIBFABRIC}>
        HSHM_ENABLE_THALLIUM=$<BOOL:${HSHM_ENABLE_THALLIUM}>
        HSHM_ENABLE_OPENMP=$<BOOL:${HSHM_ENABLE_OPENMP}>
        HSHM_ENABLE_CEREAL=$<BOOL:${HSHM_ENABLE_CEREAL}>
        HSHM_ENABLE_WINDOWS_SYSINFO=$<BOOL:${HSHM_ENABLE_WINDOWS_SYSINFO}>
        HSHM_ENABLE_PROCFS_SYSINFO=$<BOOL:${HSHM_ENABLE_PROCFS_SYSINFO}>
        HSHM_ENABLE_WINDOWS_THREADS=$<BOOL:${HSHM_ENABLE_WINDOWS_THREADS}>
        HSHM_ENABLE_PTHREADS=$<BOOL:${HSHM_ENABLE_PTHREADS}>
        HSHM_ENABLE_COMPRESS=$<BOOL:${HSHM_ENABLE_COMPRESS}>
        HSHM_ENABLE_ENCRYPT=$<BOOL:${HSHM_ENABLE_ENCRYPT}>
        HSHM_ENABLE_ELF=$<BOOL:${HSHM_ENABLE_ELF}>
        HSHM_ENABLE_COVERAGE=$<BOOL:${HSHM_ENABLE_COVERAGE}>
        HSHM_ENABLE_DOXYGEN=$<BOOL:${HSHM_ENABLE_DOXYGEN}>
        HSHM_DEBUG_LOCK=$<BOOL:${HSHM_DEBUG_LOCK}>
        HSHM_DEFAULT_THREAD_MODEL=hshm::thread::$<IF:$<BOOL:${HSHM_ENABLE_PTHREADS}>,Pthread,$<IF:$<BOOL:${HSHM_ENABLE_WINDOWS_THREADS}>,StdThread,StdThread>>
        HSHM_DEFAULT_THREAD_MODEL_GPU=hshm::thread::$<IF:$<BOOL:${HSHM_ENABLE_CUDA}>,Cuda,$<IF:$<BOOL:${HSHM_ENABLE_ROCM}>,Rocm,StdThread>>
        HSHM_DEFAULT_ALLOC_T=hipc::ThreadLocalAllocator
        HSHM_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
        HSHM_ENABLE_CUDA=$<BOOL:${HSHM_ENABLE_CUDA}>
        HSHM_ENABLE_ROCM=$<BOOL:${HSHM_ENABLE_ROCM}>
    )
    
    if(target_type STREQUAL "INTERFACE_LIBRARY")
        target_compile_definitions(${target} INTERFACE ${common_definitions})
    else()
        target_compile_definitions(${target} PUBLIC ${common_definitions})
        target_compile_definitions(${target} PRIVATE
            HSHM_ENABLE_DLL_EXPORT=$<BOOL:${BUILD_SHARED_LIBS}>
            HSHM_ENABLE_CUDA=$<BOOL:${HSHM_ENABLE_CUDA}>
            HSHM_ENABLE_ROCM=$<BOOL:${HSHM_ENABLE_ROCM}>
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
install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})


configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HermesShmConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/HermesShmCommonConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake @ONLY
)

install(
    FILES
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmConfig.cmake
    ${PROJECT_BINARY_DIR}/CMakeFiles/HermesShmCommonConfig.cmake
    DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/HermesShm
)

install(EXPORT ${HSHM_EXPORTED_TARGETS}
    FILE ${HSHM_EXPORTED_TARGETS}CoreConfig.cmake
    NAMESPACE hshm::
    DESTINATION lib/cmake/HermesShm
)

jarvis_repo_add("${HSHM_ROOT}/test/jarvis_hshm" "")
