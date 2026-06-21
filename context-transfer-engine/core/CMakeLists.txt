cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_clio_module_runtime(
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
    # GPU runtime concept removed — kernels submit tasks to the CPU runtime
    # via gpu2cpu_queue and the CPU executes them through the standard
    # chi::Container dispatch path.
)

# Surface the GPU backend flags into the runtime library so the
# GpuMetadataCache helpers (which reach ctp::GpuApi::MallocManaged etc.)
# see the matching #if CTP_ENABLE_{CUDA,ROCM,SYCL} branches. Without
# these the GPU code paths in core_runtime.cc are no-ops.
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

# Note: Cross-namespace dependencies (chimaera admin and bdev) are automatically
# linked by add_chimod_* functions in ChimaeraCommon.cmake for non-chimaera namespaces

add_subdirectory(util)
