# cuFile interceptor: file_offset / devPtr_offset honored, the registered
# descriptor copied, the caller's fd left open (#1191). Host-only test code
# (cudaMalloc + cudaMemcpy, no kernels).
add_executable(test_cufile_offsets test_cufile_offsets.cc)
target_link_libraries(test_cufile_offsets PRIVATE clio_cte_nvidia_gds)
add_test(NAME test_cufile_offsets COMMAND test_cufile_offsets)
set_tests_properties(test_cufile_offsets PROPERTIES
    TIMEOUT 60
    LABELS "gpu;adapter")
