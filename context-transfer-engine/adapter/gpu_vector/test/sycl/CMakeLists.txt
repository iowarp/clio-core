#------------------------------------------------------------------------------
# gpu_vector SYCL tests.
#
# Separate from the CUDA tests because a SYCL kernel is a lambda submitted to
# a queue, not a __global__ launched with <<<>>> -- the kernel bodies cannot
# be shared even though everything they call is. The device machinery
# (DeviceVector, yield_coro, the lane stack) and the host relaunch driver are
# the SAME headers the CUDA tests use; see sycl_cuda_compat.h for how.
#
# The SPIR-V target CANNOT build these: DPC++ crashes in codegen on any
# coroutine (a coroutine-frame PHI type assertion), so the yieldable
# machinery is reachable only through a non-SPIR-V backend. Configure with
#   -DSYCL_TARGET=nvptx64-nvidia-cuda -DSYCL_CUDA_ARCH=sm_XX
# and this directory refuses to build a target that would only crash.
#------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_SYCL)
    return()
endif()

if(SYCL_TARGET MATCHES "spir64")
    message(STATUS
        "gpu_vector SYCL tests SKIPPED: SYCL_TARGET=${SYCL_TARGET}. Device "
        "coroutines crash DPC++ codegen on SPIR-V; build with "
        "-DSYCL_TARGET=nvptx64-nvidia-cuda.")
    return()
endif()

# The kernels are their own library: ONLY this source is compiled with
# -fsycl. The test that drives it is ordinary C++, because DPC++'s device
# pass type-checks the whole translation unit and gpu_vector.h's host Vector
# class does not parse there. See gv_sycl_kernels.h.
# SHARED, not STATIC. A shared library is linked with -fsycl and so carries
# a complete device image of its own -- the arrangement libclio_run_cxx_gpu
# already uses. Out of a static archive the device code did not reach the
# executable's image, and the first kernel launch aborted with
# `It != m_DeviceKernelInfoMap.end()` from ProgramManager.
add_library(gv_sycl_kernels SHARED gv_sycl_kernels.cc)
wrp_core_apply_sycl_flags(gv_sycl_kernels)
set_target_properties(gv_sycl_kernels PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

target_include_directories(gv_sycl_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
)
target_link_libraries(gv_sycl_kernels PUBLIC
    clio_run_cxx
    clio_run_cxx_gpu
    clio_cte_core_client
)

add_executable(test_gpu_vector_smoke_sycl test_gpu_vector_smoke_sycl.cc)
# The host side links the device code but is NOT compiled with -fsycl; it
# must still LINK with it so the SYCL runtime and the embedded device image
# come along.
target_link_options(test_gpu_vector_smoke_sycl PRIVATE
    -fsycl -fsycl-targets=${SYCL_TARGET})
target_compile_definitions(test_gpu_vector_smoke_sycl PRIVATE CTP_ENABLE_SYCL=1)

target_include_directories(test_gpu_vector_smoke_sycl PRIVATE
    # <sycl/sycl.hpp> by hand: this TU is NOT compiled with -fsycl, and the
    # driver only adds the SYCL header path under that flag. macros.h still
    # includes the header (it is plain C++) because CTP_ENABLE_SYCL is on.
    ${CLIO_SYCL_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/context-runtime/include
    ${CMAKE_SOURCE_DIR}/context-runtime/test
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
    ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
)

target_link_libraries(test_gpu_vector_smoke_sycl PRIVATE
    gv_sycl_kernels
    clio_run_cxx
    clio_run_cxx_gpu
    clio_admin_client
    clio_bdev_client
    clio_cte_core_client
    clio_cte_core_runtime
    clio::cte::gpu_vector
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(test_gpu_vector_smoke_sycl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_test(NAME cte_gpu_vector_smoke_sycl COMMAND test_gpu_vector_smoke_sycl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
set_tests_properties(cte_gpu_vector_smoke_sycl PROPERTIES
    TIMEOUT 900
    LABELS "cte;gpu;gpu_vector;sycl"
    # Same reason as the CUDA tests: nothing else reclaims the runtime's TCP
    # port or its SHM between test BINARIES.
    FIXTURES_REQUIRED clio_test_cleanup_fixture
    RUN_SERIAL TRUE
)
