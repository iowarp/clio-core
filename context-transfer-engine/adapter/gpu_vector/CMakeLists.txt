#------------------------------------------------------------------------------
# gpu_vector adapter: header-only device-paged vector over the CTE.
#------------------------------------------------------------------------------

add_library(clio_cte_gpu_vector INTERFACE)
add_library(clio::cte::gpu_vector ALIAS clio_cte_gpu_vector)

target_include_directories(clio_cte_gpu_vector INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Vector::Copy creates the checkpoint fault-handler pool.
target_link_libraries(clio_cte_gpu_vector INTERFACE
    clio_cte_core_client
    clio_cte_checkpoint_client
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)

# Each workload's CMakeLists guards its own targets, so a SYCL build only
# gets the ones that have been ported. The rest are CUDA sources -- __global__
# functions launched with <<<>>> -- and sycl_cuda_compat.h making them PARSE
# under SYCL is worse than useless: they would link against libsycl and then
# launch nothing, because a __global__ function is not a SYCL kernel.
# Behind the benchmarks master switch like every other component's
# benchmarks: a ~benchmark Spack build (or any -DCLIO_CORE_ENABLE_BENCHMARKS=OFF
# configure) still compiled all 32 of these, and on the spack (cuda) CI job
# clang 22 crashed compiling clio_lammps_md_paged_bench for sm_89.
if(CLIO_CORE_ENABLE_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

if(CLIO_CORE_ENABLE_TESTS)
    if(CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM)
        add_subdirectory(test)
    endif()
    # SYCL kernels driving the same DeviceVector. Its own directory because
    # SYCL kernel bodies are lambdas rather than __global__ functions, so the
    # test SOURCES cannot be shared even though every header they use is.
    add_subdirectory(test/sycl)
endif()

#------------------------------------------------------------------------------
# gpu_vector_all: every target defined under this directory, for CI.
#
# CI used to compile a hand-written --target list, so a benchmark or test that
# was not on the list was never compiled, and four paged drivers stopped
# compiling unnoticed. Building this target instead gates everything here,
# including targets added later, on every backend whose configure creates them.
#------------------------------------------------------------------------------

# Targets that are known not to compile, each with the reason. Listed here
# (and printed at configure time) rather than silently skipped, so the list
# shrinks instead of growing. Remove an entry once its target is fixed or
# deleted.
#   clio_{kmeans,lbann,grayscott,gmx}_paged_bench: their launch headers gained
#   parameters in 19b465a48 (Aurora E1/E5) that these CUDA drivers and
#   cuda/*_launch_cuda.cc never picked up. The maintained editions are the
#   clio_*_paged_newcoro targets (benchmark/newcoro).
set(GV_ALL_KNOWN_BROKEN
    clio_kmeans_paged_bench
    clio_lbann_paged_bench
    clio_grayscott_paged_bench
    clio_gmx_paged_bench)

# gv_collect_targets(<dir> <out_var>)
#
# Appends to <out_var> every executable and library target defined in <dir>
# or any of its subdirectories (BUILDSYSTEM_TARGETS is per-directory, so the
# walk recurses through SUBDIRECTORIES).
function(gv_collect_targets dir out_var)
  get_property(_tgts DIRECTORY "${dir}" PROPERTY BUILDSYSTEM_TARGETS)
  set(_found ${${out_var}})
  foreach(_t IN LISTS _tgts)
    get_target_property(_type ${_t} TYPE)
    if(_type MATCHES "EXECUTABLE|STATIC_LIBRARY|SHARED_LIBRARY|MODULE_LIBRARY")
      list(APPEND _found ${_t})
    endif()
  endforeach()
  get_property(_subdirs DIRECTORY "${dir}" PROPERTY SUBDIRECTORIES)
  foreach(_d IN LISTS _subdirs)
    gv_collect_targets("${_d}" _found)
  endforeach()
  set(${out_var} ${_found} PARENT_SCOPE)
endfunction()

set(_gv_all_targets)
gv_collect_targets("${CMAKE_CURRENT_SOURCE_DIR}" _gv_all_targets)
set(_gv_all_skipped)
foreach(_t IN LISTS GV_ALL_KNOWN_BROKEN)
  if(_t IN_LIST _gv_all_targets)
    list(REMOVE_ITEM _gv_all_targets ${_t})
    list(APPEND _gv_all_skipped ${_t})
  endif()
endforeach()
add_custom_target(gpu_vector_all)
if(_gv_all_targets)
  add_dependencies(gpu_vector_all ${_gv_all_targets})
endif()
list(LENGTH _gv_all_targets _gv_all_n)
message(STATUS "gpu_vector_all: ${_gv_all_n} target(s)")
if(_gv_all_skipped)
  message(STATUS "gpu_vector_all: KNOWN BROKEN, not built: ${_gv_all_skipped}")
endif()
