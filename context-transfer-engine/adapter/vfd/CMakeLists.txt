# HDF5 Virtual File Driver (VFD) backed by the context-filesystem chimod.
#
# The driver uses the HDF5 1.14+ multi-version H5FD_class_t API (read_vector /
# read_selection / H5FD_CLASS_VERSION), so it requires HDF5 >= 1.14. Point the
# build at a suitable install with -DHDF5_ROOT=<prefix> (or CMAKE_PREFIX_PATH).
project(HDF5_CLIO_VFD)

find_package(HDF5 REQUIRED COMPONENTS C)
if(HDF5_VERSION AND HDF5_VERSION VERSION_LESS "1.14")
  message(FATAL_ERROR
    "clio_vfd requires HDF5 >= 1.14 (the 1.14+ VFD class API); "
    "found ${HDF5_VERSION} at ${HDF5_INCLUDE_DIRS}.")
endif()
message(STATUS "VFD: building against HDF5 ${HDF5_VERSION} (${HDF5_INCLUDE_DIRS})")

# Dynamically loaded VFDs must be shared libraries.
add_library(clio_vfd SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/H5FDclio.cc)
set_target_properties(clio_vfd PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(clio_vfd SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})
target_link_libraries(clio_vfd
  clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
  clio_cte_core_client
  ctp::cxx
  ctp::interceptor
  ${HDF5_LIBRARIES})

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(
  TARGETS clio_vfd
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(
  FILES ${CMAKE_CURRENT_SOURCE_DIR}/H5FDclio.h
  DESTINATION include/adapter/vfd
  COMPONENT headers
)
