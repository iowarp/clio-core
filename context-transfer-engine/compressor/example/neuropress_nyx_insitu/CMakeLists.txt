# ------------------------------------------------------------------------------
# Nyx in situ: AMReX device memory handed to NeuroPress inside the running
# simulation (issue #693).
#
# Nyx is an AMReX executable with no library interface, so this example
# inverts the LAMMPS-as-a-library shape: instead of Clio linking the
# simulation in, the simulation calls out. What is built here is ONE shared
# library, libclio_nyx_insitu.so, exporting the five plain-C entry points in
# clio_nyx_insitu.h. A ~90 line patch to Nyx's own in-situ callback
# (Nyx::updateInSitu, Source/IO/Nyx_output.cpp) dlopens it and hands over
# fab.dataPtr(comp) -- device memory -- per component.
#
# Consequences of that split, both deliberate:
#   * NOTHING here needs AMReX. The library takes a void* and integers, so it
#     builds anywhere the compressor does and cannot go out of step with a
#     particular AMReX build.
#   * Nyx's build is untouched. No CMake change, no Clio in its link line;
#     the patch resolves symbols at runtime, the way HDF5 dlopens a VOL.
#
# The cold read-back needs no target of its own: the CSV this library writes
# is exactly ../neuropress_field_replay's, so `neuropress_field_replay
# --readback` reads the blobs from a separate process with no Nyx anywhere.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress Nyx in-situ example: skipped (CUDA off)")
  return()
endif()

message(STATUS "NeuroPress Nyx in-situ example: enabled")

add_library(clio_nyx_insitu SHARED clio_nyx_insitu.cc)

target_include_directories(clio_nyx_insitu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(clio_nyx_insitu
  clio_cte_core_client
  clio_cte_core_runtime
  clio_cte_compressor_client
  clio::run::admin_runtime
  clio::run::admin_client
  clio::run::bdev_runtime
  clio::run::bdev_client
  ctp::cxx
  ${CMAKE_THREAD_LIBS_INIT}
)

# cudaMemcpy3DAsync and cudaPointerGetAttributes are called directly: the
# valid-box extraction is a strided device-to-device copy, and the residency
# check has to be the real CUDA query rather than a build-time assumption.
find_package(CUDAToolkit QUIET)
if(TARGET CUDA::cudart)
  target_link_libraries(clio_nyx_insitu CUDA::cudart)
endif()

install(TARGETS clio_nyx_insitu LIBRARY DESTINATION lib)
