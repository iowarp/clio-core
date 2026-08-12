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
        backend_ceiling_test.cu
        occupancy_bench.cu
        pooled_double_buffer_test.cu
        snapshot_reuse_test.cu
        snapshot_read_test.cu
        oob_guard_test.cu
        weak_scaling_bench.cu
)

# The device-side chunk-coordinate guard traps by default in a debug build (its
# tripwire for development). A trap destroys the CUDA context, which is
# unrecoverable in this shared-server, single-process test binary — and it would
# make the guard's graceful-no-op path impossible to assert in-process. Force the
# trap OFF for the whole test binary so oob_guard_test can verify the reported
# no-op directly; production / eval builds keep the debug trap. Defined on the
# TARGET (not per-source) so every TU sees one value — the guard's __device__
# members are inline, so a per-TU mismatch would be an ODR violation.
target_compile_definitions(kvhdf5_e2e_tests PRIVATE KVHDF5_GPU_OOR_TRAP=0)

# The pooled double-buffer test is the reference user of the fused
# WritePipelined producer, so it is also where that kernel's register/spill cost
# is measured. Same rationale as occupancy_bench below: ptxas has to SAY what it
# allocated. (Pre-device-link, as noted there.)
set_source_files_properties(pooled_double_buffer_test.cu PROPERTIES
    COMPILE_OPTIONS "-Xptxas=-v")

# The occupancy bench measures per-kernel register / shared-memory / spill usage,
# so it needs ptxas to SAY what it allocated. Scoped to this one source file to
# keep the rest of the build quiet. Note the target is built with
# CUDA_SEPARABLE_COMPILATION, so these compile-time numbers are pre-device-link;
# `cuobjdump -res-usage bin/kvhdf5_e2e_tests` gives the post-link ground truth
# and is what the reported table uses.
set_source_files_properties(occupancy_bench.cu PROPERTIES
    COMPILE_OPTIONS "-Xptxas=-v")

# The gsbench `hdf5` arm (the conventional GPU->D2H->HDF5 baseline) needs libhdf5.
# The root CMakeLists only looks for HDF5 under CLIO_CTE_ENABLE_HDF5_VOL, which is
# off in most builds, so find it locally instead of forcing that option on (which
# would also pull in the VOL connector). The arm is guarded by GSBENCH_HAVE_HDF5 in
# gray_scott_threeway_bench.cu, so the target still builds — minus that one arm —
# when HDF5 is absent.
if(NOT HDF5_FOUND)
    find_package(HDF5 QUIET COMPONENTS C)
    if(NOT HDF5_FOUND)
        find_package(HDF5 QUIET COMPONENTS C CONFIG)
    endif()
endif()
if(HDF5_FOUND)
    if(NOT HDF5_LIBRARIES)
        # CONFIG mode exports namespaced targets but may not set HDF5_LIBRARIES.
        foreach(_h5 HDF5::HDF5 hdf5::hdf5-shared hdf5-shared hdf5::hdf5-static hdf5-static)
            if(TARGET ${_h5})
                set(HDF5_LIBRARIES ${_h5})
                break()
            endif()
        endforeach()
    endif()
    if(NOT HDF5_INCLUDE_DIRS AND HDF5_INCLUDE_DIR)
        set(HDF5_INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
    endif()
    message(STATUS "gsbench hdf5 arm: ENABLED (HDF5 ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS})")
    target_include_directories(kvhdf5_e2e_tests PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kvhdf5_e2e_tests ${HDF5_LIBRARIES})
    target_compile_definitions(kvhdf5_e2e_tests PRIVATE GSBENCH_HAVE_HDF5=1)
else()
    message(STATUS "gsbench hdf5 arm: DISABLED (HDF5 not found)")
endif()

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

# The C++ self-supervising driver (gsbench_run) replaces the two shell orchestrators above
# (this file's threeway_bench target and scaling_campaign/run_campaign.sh) with one typed
# binary over a shared arm/study registry. It fork/execs kvhdf5_e2e_tests per arm, so it needs
# that target to exist; gated the same way this whole directory is (CUDA-only) since it is
# useless without the CUDA bench binary it drives. See gsbench_driver/SPEC.md.
add_subdirectory(gsbench_driver)
