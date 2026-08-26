# ------------------------------------------------------------------------
# VPIC in-situ adapter: libclio_vpic_insitu.so
#
# A VPIC-Kokkos deck calls the six C entry points in clio_vpic_insitu.h to
# hand its device-resident field views to Clio without writing a file. The
# deck is compiled by vpic-kokkos's own generated `bin/vpic` script (which
# drives nvcc_wrapper), so everything of Clio's stays behind this shared
# library and the deck sees a C header only -- the same split the Nyx
# adapter uses, and for the same reason.
# ------------------------------------------------------------------------
if(NOT CLIO_CORE_ENABLE_CUDA)
  message(STATUS "NeuroPress VPIC in-situ example: skipped (CUDA off)")
  return()
endif()

message(STATUS "NeuroPress VPIC in-situ example: enabled")

add_library(clio_vpic_insitu SHARED clio_vpic_insitu.cc)

target_include_directories(clio_vpic_insitu PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_link_libraries(clio_vpic_insitu
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

find_package(CUDAToolkit QUIET)
if(TARGET CUDA::cudart)
  target_link_libraries(clio_vpic_insitu CUDA::cudart)
endif()

install(TARGETS clio_vpic_insitu LIBRARY DESTINATION lib)
