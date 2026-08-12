# gsbench_run: the C++ self-supervising Gray-Scott benchmark driver. Pure host code (no CUDA,
# no CLIO libs) -- it only fork/execs kvhdf5_e2e_tests and parses text. See SPEC.md.
add_executable(gsbench_run
    main.cpp
    arms.cpp
    studies.cpp
    runner.cpp
    results.cpp
    cli.cpp
)

target_compile_features(gsbench_run PRIVATE cxx_std_17)

# The driver only fork/execs kvhdf5_e2e_tests; it doesn't link against it, but it is useless
# without it, so make sure the child binary exists whenever gsbench_run is built.
add_dependencies(gsbench_run kvhdf5_e2e_tests)

# Convenience target mirroring the existing `threeway_bench` custom target: builds and runs the
# driver's default single-config sweep. NOT wired into ctest (same reason as threeway_bench: it
# pkills leftover processes and wipes /dev/shm state, which would disrupt a normal ctest run).
add_custom_target(gsbench_run_default
    COMMAND $<TARGET_FILE:gsbench_run> --build-dir ${CMAKE_BINARY_DIR}
    DEPENDS gsbench_run kvhdf5_e2e_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    USES_TERMINAL
    COMMENT "Running gsbench_run's default single-config arm sweep"
)
