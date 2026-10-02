cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_clio_module_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_viz.cc
    # Data placement engine: one source file per placement policy
    src/dpe/dpe.cc
    src/dpe/random_dpe.cc
    src/dpe/round_robin_dpe.cc
    src/dpe/max_bw_dpe.cc
    # Data organizer (issue #738): one source file per organization policy
    src/data_organizer/data_organizer.cc
    src/data_organizer/frecency_organizer.cc
    src/autogen/core_lib_exec.cc
    # GPU runtime concept removed — kernels submit tasks to the CPU runtime
    # via gpu2cpu_queue and the CPU executes them through the standard
    # clio::run::Container dispatch path.
)

# Surface the GPU backend flags into the runtime library so its ctp
# headers resolve the same way they do inside clio_run_cxx. This is an
# ABI requirement, not a feature switch: clio_ctp/types/numbers.h changes
# the width of min_u32 & friends under CTP_ENABLE_{CUDA,ROCM}, so a TU
# that disagrees about these flags disagrees about the layout of every
# shared struct that uses them.
if(CLIO_CORE_ENABLE_SYCL)
  # Host TU: CTP_ENABLE_SYCL=1 without -fsycl. See context-runtime/src for the
  # rationale; CLIO_SYCL_INCLUDE_DIR is discovered in wrp_core_enable_sycl().
  target_compile_definitions(clio_cte_core_runtime PRIVATE
    CTP_ENABLE_SYCL=1 SYCL_DISABLE_FSYCL_SYCLHPP_WARNING=1)
  if(CLIO_SYCL_INCLUDE_DIR)
    target_include_directories(clio_cte_core_runtime PRIVATE ${CLIO_SYCL_INCLUDE_DIR})
  endif()
endif()
if(CLIO_CORE_ENABLE_CUDA)
  target_compile_definitions(clio_cte_core_runtime PRIVATE CTP_ENABLE_CUDA=1)
endif()
if(CLIO_CORE_ENABLE_ROCM)
  target_compile_definitions(clio_cte_core_runtime PRIVATE CTP_ENABLE_ROCM=1)
endif()

# Create client library using modern ChiMod build functions
add_clio_module_client(
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
    src/tag.cc
)

# Tell clio_cte/api.h to emit dllexport (instead of the default dllimport
# that dependent test binaries / adapter DLLs see) for data globals like
# g_cte_client.  Mirrors CLIO_RUN_BUILDING_DLL on the clio_run_cxx target.
if(TARGET clio_cte_core_client)
  target_compile_definitions(clio_cte_core_client PRIVATE CLIO_CTE_CORE_BUILDING_DLL=1)
endif()

# Note: Cross-namespace dependencies (clio_run admin and bdev) are automatically
# linked by add_chimod_* functions in ClioCoreCommon.cmake for non-clio_run namespaces

# Web dashboard page for this module (issue #990): mounted at /viz/clio_cte_core
# by the runtime as soon as a CTE pool comes up.
clio_add_viz_assets(MOD_NAME clio_cte_core)

add_subdirectory(util)
