# ------------------------------------------------------------------------------
# IOWarp HDF5 VOL Connector
# Requires: CLIO_CTE_ENABLE_HDF5_VOL=ON and HDF5 >= 1.14
#
# NOTE: the VOL connector API this source targets (H5VL_VERSION 3, the
# multi-dataset read/write callback signatures in H5VLconnector.h) is already
# present and ABI-stable in HDF5 1.14.x. The previous ">= 2.0.0" floor was
# over-conservative and excluded every HDF5 that ships with the MiV stack
# (1.14.6 conda). Lowered to 1.14.0 so the connector builds against the same
# HDF5 that neuroh5/MiV-Simulator link against.
# ------------------------------------------------------------------------------

if(NOT HDF5_FOUND)
  message(WARNING "HDF5 not found. HDF5 VOL adapter will not be built.")
  return()
endif()

if(HDF5_VERSION VERSION_LESS "1.14.0")
  message(WARNING "HDF5 ${HDF5_VERSION} found but >= 1.14.0 required for VOL connector. "
                  "HDF5 VOL adapter will not be built.")
  return()
endif()

message(STATUS "Building HDF5 VOL adapter (iowarp_hdf5_vol)")

add_library(iowarp_hdf5_vol SHARED
    iowarp_vol.cc
)

target_include_directories(iowarp_hdf5_vol PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${HDF5_C_INCLUDE_DIRS}
)

target_link_libraries(iowarp_hdf5_vol
    clio_cte_core_client
    clio_cte_core_runtime
    chimaera::admin_runtime
    chimaera::admin_client
    chimaera::bdev_runtime
    chimaera::bdev_client
    ctp::cxx
    ${HDF5_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

set_target_properties(iowarp_hdf5_vol PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS iowarp_hdf5_vol
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

message(STATUS "HDF5 VOL adapter configured successfully")
