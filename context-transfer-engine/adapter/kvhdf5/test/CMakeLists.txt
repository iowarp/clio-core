# Catch module (catch_discover_tests). Catch2 was located by the root
# CMakeLists under CLIO_CORE_ENABLE_TESTS, either as a bundled add_subdirectory
# (sets Catch2_SOURCE_DIR, module in extras/) or via find_package (sets
# Catch2_DIR to the package cmake dir that holds Catch.cmake). Neither path puts
# that dir on CMAKE_MODULE_PATH, so add it before including the module.
if(NOT COMMAND catch_discover_tests)
    if(DEFINED Catch2_SOURCE_DIR AND EXISTS "${Catch2_SOURCE_DIR}/extras/Catch.cmake")
        list(APPEND CMAKE_MODULE_PATH "${Catch2_SOURCE_DIR}/extras")
    elseif(DEFINED Catch2_DIR)
        list(APPEND CMAKE_MODULE_PATH "${Catch2_DIR}")
    endif()
    include(Catch)
endif()

# CMAKE_CUDA_ARCHITECTURES is set once by the root CMakeLists; the GPU test
# targets consume it below.

# Shared recipe for the GPU test binaries: the clio contract canary (test/clio)
# and the kvhdf5 end-to-end tests (test/e2e). Both link the same clio runtime
# libs, compile with the same CUDA defs, and need the same runtime env injected
# into their ctest registrations. They differ in only three ways: target name,
# sources, and whether they see kvhdf5's own headers — the canary must NOT (pass
# WITH_KVHDF5_HEADERS only for e2e, which links the clio::cte::kvhdf5 INTERFACE
# target), so it physically cannot include kvhdf5 code.
#
#   kvhdf5_add_gpu_test_target(<target> [WITH_KVHDF5_HEADERS] SRCS <files...>)
#
# Called from the test/clio and test/e2e subdirs, so ../support resolves against
# the CALLER's dir — both sit at the same depth under test/.
function(kvhdf5_add_gpu_test_target target)
    cmake_parse_arguments(ARG "WITH_KVHDF5_HEADERS" "" "SRCS" ${ARGN})

    set(_include_dirs
        ${CMAKE_CURRENT_SOURCE_DIR}/../support
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    )

    set(_link_libs
        clio_run_cxx
        clio_run_cxx_gpu
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        ctp::cuda_cxx
        Catch2::Catch2WithMain
        ${CMAKE_THREAD_LIBS_INIT}
    )
    # kvhdf5's own headers ride in via the adapter INTERFACE target (which also
    # re-states the clio_run link, harmlessly deduped). The canary omits this,
    # so it cannot see kvhdf5 code.
    if(ARG_WITH_KVHDF5_HEADERS)
        list(APPEND _link_libs clio::cte::kvhdf5)
    endif()

    # add_cuda_executable comes from clio's ClioCoreCommon.cmake. DO_COPY=FALSE:
    # compile the .cu in place. Sets CUDA_SEPARABLE_COMPILATION + PIC.
    add_cuda_executable(${target} FALSE
        ${ARG_SRCS}
        INCLUDE_DIRS
            ${_include_dirs}
        LINK_LIBS
            ${_link_libs}
        DEFS
            CTP_ENABLE_CUDA=1
            HSHM_ENABLE_CUDA=1
    )

    # kvhdf5's device-facing headers use C++20 (std::span, the BlobBackend
    # concept), matching the codebase-wide CMAKE_CXX_STANDARD 20. Pin CUDA to 20
    # too so the .cu translation units compile them, regardless of any global
    # CMAKE_CUDA_STANDARD (e.g. a build that forces 17).
    set_target_properties(${target} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # catch_discover_tests runs the binary bare (no shell, empty env). The clio
    # runtime needs CLIO_BIND_ADDR set and finds its chimods by scanning
    # CHI_REPO_PATH / LD_LIBRARY_PATH, so inject those into the registered tests.
    # (build/bin is where clio drops every .so + chimod.)
    catch_discover_tests(${target}
        PROPERTIES
            ENVIRONMENT "CLIO_BIND_ADDR=127.0.0.1;CHI_REPO_PATH=${CMAKE_BINARY_DIR}/bin;LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

# CPU-only unit tests (chunk/metadata logic; no server process)
add_subdirectory(unit)

# clio dependency-contract canary (raw clio, no kvhdf5 headers)
add_subdirectory(clio)

# kvhdf5 GPU end-to-end tests (our producer surface on the clio runtime)
add_subdirectory(e2e)
