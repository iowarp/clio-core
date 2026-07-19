# kvhdf5 GPU end-to-end tests.
#
# Unlike the clio contract canary (../clio), these exercise kvhdf5's OWN
# device-facing producer surface (GpuCteDataset, GpuDatasetHandle, chunking,
# tag_path) top-to-bottom against a live clio CTE runtime. A failure here can be
# kvhdf5 OR clio — the ../clio canary is what disambiguates. WITH_KVHDF5_HEADERS
# links the clio::cte::kvhdf5 INTERFACE target (its include path); shared CTE
# bring-up is ../support/cte_env.h.

if(NOT CLIO_CORE_ENABLE_CUDA)
    message(STATUS "kvhdf5_e2e_tests: skipped (CUDA not enabled)")
    return()
endif()

kvhdf5_add_gpu_test_target(kvhdf5_e2e_tests WITH_KVHDF5_HEADERS
    SRCS
        dataset_gpu_putget_test.cu
        gray_scott_gpu_test.cu
        multichunk_gpu_putget_test.cu
        async_overlap_test.cu
        dataset_lifecycle_test.cu
        gray_scott_async_e2e_test.cu
        dataset_refire_probe_test.cu
        gray_scott_threeway_bench.cu
)

# Convenience target for the three-way Gray-Scott I/O benchmark. The gsbench
# arms are hidden ([.]) Catch2 cases inside kvhdf5_e2e_tests; each must run in
# its OWN process (clio's per-process GPU-backend ceiling), so a wrapper script
# orchestrates them. It is NOT wired into ctest on purpose: the script does a
# process-global `pkill` + shm wipe between arms, which would disrupt a normal
# `ctest` run.
#
#   cmake --build <dir> --target threeway_bench
#
# GSBENCH_BUILD_DIR is injected here so the script finds the binary regardless
# of the build dir; the script's own default assumes the iowarp-core build/.
# All other knobs remain env-overridable (see the script header and
# gray_scott_threeway_bench.cu).
add_custom_target(threeway_bench
    COMMAND ${CMAKE_COMMAND} -E env GSBENCH_BUILD_DIR=${CMAKE_BINARY_DIR}
            bash ${CMAKE_CURRENT_SOURCE_DIR}/run_threeway_bench.sh
    DEPENDS kvhdf5_e2e_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running three-way Gray-Scott I/O benchmark (raw / sync / async)"
)
