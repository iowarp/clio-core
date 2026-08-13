# ------------------------------------------------------------------------------
# NeuroPress <-> Clio callback-trace equivalence harness (issue #693).
#
# CROSS-PROJECT, like test/unit/compress/model/parity: it links native
# NeuroPress's own shared library alongside Clio's port and drives both over the
# same GPU-resident chunk, comparing them callback by callback. Reading two
# implementations and concluding they agree is not the same as running both and
# diffing the numbers.
#
# NeuroPress is NOT a build dependency of Clio, so this target is SKIPPED --
# not failed -- when the library or CUPTI is unavailable. To enable it:
#
#   cmake -DCLIO_NEUROPRESS_SRC_DIR=/path/to/NeuroPress \
#         -DCLIO_NEUROPRESS_BUILD_DIR=/path/to/NeuroPress-build ...
#
# where the build directory is one produced by:
#
#   cmake -S <NeuroPress> -B <build> -DNVCOMP_PREFIX=<prefix with
#         include/nvcomp.hpp and lib/libnvcomp.so>
#   cmake --build <build> --target gpucompress -j
#
# Three translation units on purpose, matching the split the sibling
# neuropress_gpu_direct example uses and for the same reasons:
#   device_probe.cu   kernels only, no Clio and no NeuroPress includes
#   native_path.cc    NeuroPress internals only (they pull in nvcomp)
#   clio_path.cc      Clio headers only
# The driver includes neither project's internals, only the neutral headers.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress callback-trace equivalence: skipped (CUDA off)")
  return()
endif()

set(CLIO_NEUROPRESS_SRC_DIR "$ENV{HOME}/NeuroPress" CACHE PATH
    "NeuroPress source tree, for the cross-project equivalence harness")
set(CLIO_NEUROPRESS_BUILD_DIR "$ENV{HOME}/np-build" CACHE PATH
    "NeuroPress build tree holding libgpucompress")

find_library(CLIO_NEUROPRESS_LIB
  NAMES gpucompress
  HINTS ${CLIO_NEUROPRESS_BUILD_DIR} ${CLIO_NEUROPRESS_BUILD_DIR}/lib
  NO_DEFAULT_PATH)

# nvcomp headers, needed because NeuroPress's api/internal.hpp includes its own
# compression factory which includes <nvcomp.hpp>.
find_path(CLIO_NVCOMP_INCLUDE_DIR nvcomp.hpp
  PATH_SUFFIXES nvcomp
  HINTS /usr/include/libnvcomp5-dev-cuda-12 /usr/local/include)

# CUPTI is the instrumentation. It ships with the CUDA toolkit but is not part
# of the default library path, so it is located explicitly.
find_path(CLIO_CUPTI_INCLUDE_DIR cupti.h
  HINTS ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/targets/x86_64-linux/include
        /usr/local/cuda/extras/CUPTI/include)
find_library(CLIO_CUPTI_LIB
  NAMES cupti
  HINTS /usr/local/cuda/targets/x86_64-linux/lib
        /usr/local/cuda/extras/CUPTI/lib64
        /usr/local/cuda/lib64)

if(NOT EXISTS ${CLIO_NEUROPRESS_SRC_DIR}/include/gpucompress.h)
  message(STATUS
    "NeuroPress callback-trace equivalence: skipped (no NeuroPress tree at "
    "${CLIO_NEUROPRESS_SRC_DIR}; set CLIO_NEUROPRESS_SRC_DIR)")
  return()
endif()
if(NOT CLIO_NEUROPRESS_LIB)
  message(STATUS
    "NeuroPress callback-trace equivalence: skipped (no libgpucompress in "
    "${CLIO_NEUROPRESS_BUILD_DIR}; set CLIO_NEUROPRESS_BUILD_DIR)")
  return()
endif()
if(NOT CLIO_NVCOMP_INCLUDE_DIR)
  message(STATUS
    "NeuroPress callback-trace equivalence: skipped (nvcomp headers not found)")
  return()
endif()
if(NOT CLIO_CUPTI_INCLUDE_DIR OR NOT CLIO_CUPTI_LIB)
  # CUPTI is what makes the transfer and kernel records real rather than
  # assumed, so building without it would produce a report that looks complete
  # and silently is not.
  message(STATUS
    "NeuroPress callback-trace equivalence: skipped (CUPTI not found; it is "
    "what makes the D->H / kernel records observed rather than inferred)")
  return()
endif()

message(STATUS
  "NeuroPress callback-trace equivalence: enabled "
  "(${CLIO_NEUROPRESS_LIB}, cupti=${CLIO_CUPTI_LIB})")

add_cuda_library(npeq_device_probe SHARED FALSE device_probe.cu)
target_include_directories(npeq_device_probe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(neuropress_gpu_chunk_equivalence
  neuropress_gpu_chunk_equivalence.cc
  callback_trace.cc
  comparison.cc
  clio_path.cc
  native_path.cc
)

target_include_directories(neuropress_gpu_chunk_equivalence PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CLIO_NEUROPRESS_SRC_DIR}/include
  ${CLIO_NEUROPRESS_SRC_DIR}/src
  ${CLIO_NVCOMP_INCLUDE_DIR}
  ${CLIO_NVCOMP_INCLUDE_DIR}/..
  ${CLIO_CUPTI_INCLUDE_DIR}
)

target_compile_definitions(neuropress_gpu_chunk_equivalence PRIVATE
  CLIO_CTP_NEUROPRESS_WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/context-transport-primitives/src/compress/model/weights/"
  CTP_ENABLE_NEUROPRESS_GPU=1
  CTP_ENABLE_CUDA=1
)

target_link_libraries(neuropress_gpu_chunk_equivalence
  npeq_device_probe
  ${CLIO_NEUROPRESS_LIB}
  ${CLIO_CUPTI_LIB}
  clio_cte_compressor_runtime
  clio_cte_compressor_client
  clio_cte_core_runtime
  clio_cte_core_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::cxx
  ctp::cuda_cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

# The NeuroPress library is not on any default search path, so record where it
# was found. Without this the executable builds and then fails to start.
get_filename_component(_npeq_np_libdir ${CLIO_NEUROPRESS_LIB} DIRECTORY)
set_target_properties(neuropress_gpu_chunk_equivalence PROPERTIES
  BUILD_RPATH "${_npeq_np_libdir}"
  INSTALL_RPATH "${_npeq_np_libdir}")

install(TARGETS neuropress_gpu_chunk_equivalence npeq_device_probe
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
)

# Registered as a test as well as an example: it is a differential check with a
# pass/fail verdict, and it exits 77 without a GPU so that reports as skipped.
add_test(NAME cte_neuropress_callback_trace_equivalence
  COMMAND neuropress_gpu_chunk_equivalence --out ${CMAKE_BINARY_DIR}/npeq)
set_tests_properties(cte_neuropress_callback_trace_equivalence PROPERTIES
  SKIP_RETURN_CODE 77
  TIMEOUT 1200
  LABELS "693;parity;gpu")
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/npeq)
