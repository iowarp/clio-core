# ------------------------------------------------------------------------------
# Gray-Scott -> HDF5 -> Clio -> NeuroPress (issue #693).
#
# The only example here whose application code contains NO Clio: it includes
# hdf5.h and the ported simulation, links no Clio library, and still gets
# per-chunk NeuroPress selection. Clio arrives through the environment --
# HDF5 dlopens the VOL connector, which hosts the runtime in-process when
# CLIO_WITH_RUNTIME=1. See README.md for the full recipe and why that variable
# is load-bearing.
#
# Needs HDF5 for the writer and the ported 3D simulation from
# ../../generator/grayscott. Deliberately does NOT link clio_cte_* : adding
# them would defeat the point the example exists to make.
# ------------------------------------------------------------------------------

if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress Gray-Scott HDF5 example: skipped (CUDA off)")
  return()
endif()

find_package(HDF5 QUIET COMPONENTS C)
if(NOT HDF5_FOUND)
  message(STATUS "NeuroPress Gray-Scott HDF5 example: skipped (HDF5 not found)")
  return()
endif()

message(STATUS "NeuroPress Gray-Scott HDF5 example: enabled")

set(_gs_dir ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/generator/grayscott)

add_executable(neuropress_grayscott_h5
  neuropress_grayscott_h5.cu
  ${_gs_dir}/grayscott_sim.cu
)

set_target_properties(neuropress_grayscott_h5 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(neuropress_grayscott_h5 PRIVATE
  ${_gs_dir}
  ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(neuropress_grayscott_h5
  ${HDF5_C_LIBRARIES}
)

install(TARGETS neuropress_grayscott_h5 RUNTIME DESTINATION bin)
