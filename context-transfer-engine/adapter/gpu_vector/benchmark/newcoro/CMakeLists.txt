# clio-coroc editions of the paged benchmarks, built in-tree.
#
# Each clio_<name>_paged_newcoro.cc is one self-contained TU (driver, kernels
# and launches) written on the CO_AWAIT API. It is transpiled by clio-coroc
# (context-runtime/coroc/coroc_run.cmake) and the generated TU is compiled as
# CUDA. This is build_newcoro.sh as CMake targets, so the editions build
# wherever the CUDA tree builds -- CI, Windows -- not only in the devcontainer.
#
# Needs a CUDA build and a transpiler: CLIO_COROC (a prebuilt clio-coroc), the
# in-tree clio-coroc target (CLIO_RUN_ENABLE_COROC=ON), or CLIO_ML_COROC
# (clio-ml's name for the same thing).

get_filename_component(_nc_core "${CMAKE_SOURCE_DIR}" ABSOLUTE)
set(_nc_run "${_nc_core}/context-runtime/coroc/coroc_run.cmake")
set(_nc_gvi "${_nc_core}/context-transfer-engine/adapter/gpu_vector/include")
set(_nc_bench "${_nc_core}/context-transfer-engine/adapter/gpu_vector/benchmark")

set(CLIO_COROC "" CACHE FILEPATH
    "Prebuilt clio-coroc for the newcoro benchmarks (empty: use the in-tree target)")
if(CLIO_COROC)
  set(_nc_coroc "${CLIO_COROC}")
  set(_nc_coroc_dep "${CLIO_COROC}")
elseif(TARGET clio-coroc)
  set(_nc_coroc "$<TARGET_FILE:clio-coroc>")
  set(_nc_coroc_dep clio-coroc)
elseif(CLIO_ML_COROC)
  set(_nc_coroc "${CLIO_ML_COROC}")
  set(_nc_coroc_dep "${CLIO_ML_COROC}")
else()
  message(STATUS "gpu_vector newcoro benches: no clio-coroc "
                 "(set CLIO_COROC or CLIO_RUN_ENABLE_COROC=ON); skipped")
  return()
endif()

# The C++20 lowering is selected tree-wide by CLIO_YIELD_CORO; these TUs use
# the transpiled lowering instead, exactly as build_newcoro.sh strips it.
get_directory_property(_nc_defs COMPILE_DEFINITIONS)
list(REMOVE_ITEM _nc_defs CLIO_YIELD_CORO)
set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "${_nc_defs}")

set(_nc_cuda_root "${CUDAToolkit_LIBRARY_ROOT}")
if(NOT _nc_cuda_root)
  get_filename_component(_nc_cuda_root "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  get_filename_component(_nc_cuda_root "${_nc_cuda_root}" DIRECTORY)
endif()

# clio_gv_newcoro_bench(<name>)
#
# Adds clio_<name>_paged_newcoro from <name>/clio_<name>_paged_newcoro.cc:
# transpiles the TU (the tool also rewrites device_vector.h's CLIO_COROC
# region into an overlay), compiles the generated copy as CUDA with
# CLIO_COROC, and links the same libraries as the paged editions.
function(clio_gv_newcoro_bench name)
  set(_tgt clio_${name}_paged_newcoro)
  set(_tu "${_nc_bench}/${name}/${_tgt}.cc")
  set(_gen "${CMAKE_BINARY_DIR}/clio_gv_newcoro/${_tgt}")
  file(RELATIVE_PATH _rel "${_nc_core}" "${_tu}")
  set(_out "${_gen}/${_rel}")

  add_executable(${_tgt})
  target_include_directories(${_tgt} BEFORE PRIVATE
      "${_gen}/context-transfer-engine/adapter/gpu_vector/include")
  target_include_directories(${_tgt} PRIVATE
      "${_nc_bench}/${name}"
      "${_nc_bench}"
      "${_nc_gvi}"
      "${_nc_gvi}/clio_cte/gpu_vector"
      ${CMAKE_SOURCE_DIR}/context-runtime/include
      ${CMAKE_SOURCE_DIR}/context-runtime/modules/admin/include
      ${CMAKE_SOURCE_DIR}/context-runtime/modules/bdev/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/checkpoint/include
      ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include)
  target_link_libraries(${_tgt} PRIVATE
      clio_run_cxx clio_run_cxx_gpu clio_admin_client clio_bdev_client
      clio_cte_core_client clio_cte_core_runtime
      clio_cte_compressor_client clio_cte_compressor_runtime
      clio::cte::gpu_vector ctp::cuda_cxx ${CMAKE_THREAD_LIBS_INIT})
  target_compile_definitions(${_tgt} PRIVATE CLIO_CTE_HAVE_COMPRESSOR=1)
  set_target_properties(${_tgt} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ${CLIO_CUDA_SEPARABLE}
      POSITION_INDEPENDENT_CODE ON
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  set(_inc "$<TARGET_PROPERTY:${_tgt},INCLUDE_DIRECTORIES>")
  set(_def "$<TARGET_PROPERTY:${_tgt},COMPILE_DEFINITIONS>")
  string(JOIN "@@" _expect_s
      "context-transfer-engine/adapter/gpu_vector/include/clio_cte/gpu_vector/device_vector.h")
  string(JOIN "@@" _lang_s -x cuda "--cuda-path=${_nc_cuda_root}"
      --cuda-gpu-arch=sm_80 --cuda-device-only -Wno-unknown-cuda-version
      -std=c++20 -Wno-everything -ferror-limit=0 -DCLIO_COROC)
  file(GLOB _hdrs "${_nc_bench}/${name}/*.h" "${_nc_bench}/*.h")
  add_custom_command(
      OUTPUT "${_out}"
      COMMAND ${CMAKE_COMMAND}
              "-DCOROC=${_nc_coroc}" "-DTU=${_tu}" "-DROOT=${_nc_core}" "-DGEN=${_gen}"
              "-DEXPECT=${_expect_s}"
              "-DLOG=${CMAKE_BINARY_DIR}/clio_gv_newcoro/${_tgt}.log"
              "-DARGS=${_lang_s}$<$<BOOL:${_inc}>:@@-I$<JOIN:${_inc},@@-I>>$<$<BOOL:${_def}>:@@-D$<JOIN:${_def},@@-D>>"
              -P "${_nc_run}"
      DEPENDS "${_nc_run}" "${_tu}" ${_nc_coroc_dep} ${_hdrs}
              "${_nc_gvi}/clio_cte/gpu_vector/device_vector.h"
      COMMENT "gpu_vector: clio-coroc ${_tgt}"
      VERBATIM)
  target_sources(${_tgt} PRIVATE "${_out}")
  set_source_files_properties("${_out}" PROPERTIES LANGUAGE CUDA)
  set_property(SOURCE "${_out}" APPEND PROPERTY COMPILE_DEFINITIONS CLIO_COROC)
endfunction()

foreach(_nc_name kmeans lbann grayscott gmx)
  clio_gv_newcoro_bench(${_nc_name})
endforeach()
