# NeuroPress differential parity harnesses (issue #693).
#
# These are CROSS-PROJECT checks: they compile NeuroPress's own nn_gpu.cu /
# stats_kernel.cu / entropy_kernel.cu straight from its source tree and run
# them alongside Clio's port on identical inputs. Reading two implementations
# and concluding they agree is not the same as running both and diffing the
# numbers -- the SGD anti-flip divergence fixed in 7b043827 passed every
# source review and was only caught here.
#
# Skipped (not failed) when the NeuroPress tree or CUDA is unavailable, since
# neither is a build dependency of Clio. Point CLIO_NEUROPRESS_SRC_DIR at a
# checkout to enable:  cmake -DCLIO_NEUROPRESS_SRC_DIR=/path/to/NeuroPress
project(clio_ctp)

set(CLIO_NEUROPRESS_SRC_DIR "$ENV{HOME}/NeuroPress" CACHE PATH
    "NeuroPress source tree, for the cross-project parity tests")

# nn_gpu.cu pulls in compression_factory.hpp, which includes <nvcomp.hpp>.
find_path(CLIO_NVCOMP_INCLUDE_DIR nvcomp.hpp
    PATH_SUFFIXES nvcomp
    HINTS /usr/include/libnvcomp5-dev-cuda-12 /usr/local/include)

set(_np_required
    ${CLIO_NEUROPRESS_SRC_DIR}/src/nn/nn_gpu.cu
    ${CLIO_NEUROPRESS_SRC_DIR}/src/stats/stats_kernel.cu
    ${CLIO_NEUROPRESS_SRC_DIR}/src/stats/entropy_kernel.cu)

set(_np_available TRUE)
foreach(_f IN LISTS _np_required)
    if(NOT EXISTS ${_f})
        set(_np_available FALSE)
    endif()
endforeach()

if(NOT CLIO_CORE_ENABLE_CUDA)
    message(STATUS "NeuroPress parity: skipped (CUDA not enabled)")
    return()
endif()
if(NOT _np_available)
    message(STATUS
        "NeuroPress parity: skipped (no NeuroPress tree at "
        "${CLIO_NEUROPRESS_SRC_DIR}; set CLIO_NEUROPRESS_SRC_DIR to enable)")
    return()
endif()
if(NOT CLIO_NVCOMP_INCLUDE_DIR)
    message(STATUS "NeuroPress parity: skipped (nvcomp headers not found)")
    return()
endif()

message(STATUS "NeuroPress parity: enabled (${CLIO_NEUROPRESS_SRC_DIR})")

# One target per harness. Both link the same upstream objects; listing the
# .cu sources directly (rather than a shared object library) keeps each
# executable independently buildable and avoids a second device-link step,
# which this repo has repeatedly found fragile under -rdc.
function(clio_add_neuropress_parity_test target source test_name)
    add_executable(${target}
        ${source}
        ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_globals_stub.cu
        ${_np_required})

    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON)

    # NOT hardcoded to one GPU: the harness runs wherever the rest of the
    # build runs. atomicAdd(double*) needs sm_60+, so refuse anything older
    # rather than emit a kernel that silently misbehaves.
    if(CMAKE_CUDA_ARCHITECTURES AND
       NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
        foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
            string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
            if(_arch_num AND _arch_num LESS 60)
                message(FATAL_ERROR
                    "NeuroPress parity needs sm_60+ for atomicAdd(double*), "
                    "got ${_arch}")
            endif()
        endforeach()
    endif()

    target_include_directories(${target} PRIVATE
        ${CLIO_NEUROPRESS_SRC_DIR}/src
        ${CLIO_NEUROPRESS_SRC_DIR}/src/nn
        ${CLIO_NEUROPRESS_SRC_DIR}/src/stats
        ${CLIO_NEUROPRESS_SRC_DIR}/include
        ${CLIO_NVCOMP_INCLUDE_DIR}
        ${CLIO_NVCOMP_INCLUDE_DIR}/..)

    target_compile_definitions(${target} PRIVATE
        CTP_ENABLE_NEUROPRESS_GPU=1
        CTP_ENABLE_CUDA=1
        CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CLIO_CTP_ROOT}/src/compress/model/weights")

    target_link_libraries(${target} clio_ctp_compress_model)
    if(TARGET ctp::cuda_cxx)
        target_link_libraries(${target} ctp::cuda_cxx)
    endif()
    if(CLIO_CORE_CUDA_NEEDS_GCOV)
        set_property(TARGET ${target} APPEND PROPERTY LINK_LIBRARIES gcov)
    endif()

    add_test(NAME ${test_name} COMMAND ${CMAKE_BINARY_DIR}/bin/${target})
    # Both harnesses exit 77 when no CUDA device is present at RUN time (a
    # CUDA-capable build on a GPU-less machine), so that reports as skipped
    # rather than a spurious failure.
    set_tests_properties(${test_name} PROPERTIES
        SKIP_RETURN_CODE 77
        LABELS "693;parity;gpu")
endfunction()

clio_add_neuropress_parity_test(neuropress_parity_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_parity.cu
    ctp_neuropress_parity)

clio_add_neuropress_parity_test(neuropress_dataset_parity_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_dataset_parity.cu
    ctp_neuropress_dataset_parity)

# Per-flow SGD state: drives upstream's runNNSGDCtx on two CompContexts
# against two Clio worker threads. Needs src/api on the include path for the
# real CompContext layout, and a thread library for the Clio-side flows.
clio_add_neuropress_parity_test(neuropress_flow_parity_exec
    ${CMAKE_CURRENT_SOURCE_DIR}/neuropress_flow_parity.cu
    ctp_neuropress_flow_parity)
find_package(Threads REQUIRED)
target_link_libraries(neuropress_flow_parity_exec Threads::Threads)

# The dataset harness sweeps 1 GiB in 4 MiB chunks on both sides.
set_tests_properties(ctp_neuropress_dataset_parity PROPERTIES TIMEOUT 900)
