#-----------------------------------------------------------------------------
# gpu_vector performance benchmark.
#
# Drives a Vector<clio::run::u32> through write/flush/read and prints
# bandwidth + latency. Knobs: --blocks, --pages-per-block, --page-size,
# --ratio, --total-bytes, --cache-period-ms, --iters, --no-read,
# --bdev-capacity-mib, --gpu-id.
#-----------------------------------------------------------------------------

if(CLIO_CORE_ENABLE_CUDA)
    # Baseline cudaMemcpy bench (pure CUDA runtime; no clio_run deps).
    add_cuda_executable(clio_memcpy_d2h_bench FALSE
        clio_memcpy_d2h_bench.cc
        LINK_LIBS
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_memcpy_d2h_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_memcpy_d2h_bench RUNTIME DESTINATION bin)

    # Raw GPU-kernel write/read bandwidth to pinned host / HBM. No runtime,
    # no IPC — answers "what's the upper bound for kernel writes to DRAM?"
    add_cuda_executable(clio_kernel_pinned_bench FALSE
        clio_kernel_pinned_bench.cc
    )
    set_target_properties(clio_kernel_pinned_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_link_libraries(clio_kernel_pinned_bench PUBLIC
        ctp::cuda_cxx
    )
    install(TARGETS clio_kernel_pinned_bench RUNTIME DESTINATION bin)

    add_cuda_executable(clio_cte_gpu_vector_bench FALSE
        clio_cte_gpu_vector_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
        LINK_LIBS
            clio_run_cxx
            clio_run_cxx_gpu
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            clio::cte::gpu_vector
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_cte_gpu_vector_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_cte_gpu_vector_bench RUNTIME DESTINATION bin)

    # Traditional out-of-memory baseline: synchronous D2H + synchronous PutBlob.
    add_cuda_executable(clio_traditional_oom_bench FALSE
        clio_traditional_oom_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        LINK_LIBS
            clio_run_cxx
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_traditional_oom_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_traditional_oom_bench RUNTIME DESTINATION bin)

    # CUDA UVM baseline: cudaMallocManaged + synchronous PutBlob.
    add_cuda_executable(clio_uvm_bench FALSE
        clio_uvm_bench.cc
        INCLUDE_DIRS
            ${CMAKE_SOURCE_DIR}/context-runtime/include
            ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
            ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        LINK_LIBS
            clio_run_cxx
            clio_admin_client
            clio_bdev_client
            clio_cte_core_client
            clio_cte_core_runtime
            ctp::cuda_cxx
            ${CMAKE_THREAD_LIBS_INIT}
    )
    set_target_properties(clio_uvm_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    install(TARGETS clio_uvm_bench RUNTIME DESTINATION bin)
endif()

if(CLIO_CORE_ENABLE_ROCM)
    add_rocm_gpu_executable(clio_memcpy_d2h_bench FALSE
        clio_memcpy_d2h_bench.cc
    )
    target_compile_options(clio_memcpy_d2h_bench PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
    )
    set_target_properties(clio_memcpy_d2h_bench PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_link_libraries(clio_memcpy_d2h_bench PUBLIC
        ctp::rocm_cxx
        ${CMAKE_THREAD_LIBS_INIT}
    )
    install(TARGETS clio_memcpy_d2h_bench RUNTIME DESTINATION bin)

    add_rocm_gpu_executable(clio_cte_gpu_vector_bench FALSE
        clio_cte_gpu_vector_bench.cc
    )
    target_compile_options(clio_cte_gpu_vector_bench PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
    )
    set_target_properties(clio_cte_gpu_vector_bench PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_include_directories(clio_cte_gpu_vector_bench PUBLIC
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/adapter/gpu_vector/include
    )
    target_link_libraries(clio_cte_gpu_vector_bench PUBLIC
        clio_run_cxx
        clio_run_cxx_gpu
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        clio::cte::gpu_vector
        ctp::rocm_cxx
        ${CMAKE_THREAD_LIBS_INIT}
    )
    install(TARGETS clio_cte_gpu_vector_bench RUNTIME DESTINATION bin)

    # Traditional out-of-memory baseline (ROCm).
    add_rocm_gpu_executable(clio_traditional_oom_bench FALSE
        clio_traditional_oom_bench.cc
    )
    target_compile_options(clio_traditional_oom_bench PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
    )
    set_target_properties(clio_traditional_oom_bench PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_include_directories(clio_traditional_oom_bench PUBLIC
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    )
    target_link_libraries(clio_traditional_oom_bench PUBLIC
        clio_run_cxx
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        ctp::rocm_cxx
        ${CMAKE_THREAD_LIBS_INIT}
    )
    install(TARGETS clio_traditional_oom_bench RUNTIME DESTINATION bin)

    # CUDA UVM baseline (ROCm).
    add_rocm_gpu_executable(clio_uvm_bench FALSE
        clio_uvm_bench.cc
    )
    target_compile_options(clio_uvm_bench PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--generate-code=arch=compute_89,code=sm_89>"
    )
    set_target_properties(clio_uvm_bench PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    target_include_directories(clio_uvm_bench PUBLIC
        ${CMAKE_SOURCE_DIR}/context-runtime/include
        ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
        ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
    )
    target_link_libraries(clio_uvm_bench PUBLIC
        clio_run_cxx
        clio_admin_client
        clio_bdev_client
        clio_cte_core_client
        clio_cte_core_runtime
        ctp::rocm_cxx
        ${CMAKE_THREAD_LIBS_INIT}
    )
    install(TARGETS clio_uvm_bench RUNTIME DESTINATION bin)
endif()
