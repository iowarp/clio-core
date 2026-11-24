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
set(CLIO_VFD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/H5FDclio.cc)
if(WIN32)
  # Win32 half of the platform layer. A separate translation unit so that
  # <windows.h> -- and the macros it defines over ordinary identifiers like
  # Yield() -- never reaches the driver body or clio's headers.
  list(APPEND CLIO_VFD_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/H5FDclio_compat_win.cc)
endif()

add_library(clio_vfd SHARED ${CLIO_VFD_SOURCES})
set_target_properties(clio_vfd PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

target_include_directories(clio_vfd SYSTEM PUBLIC ${HDF5_INCLUDE_DIRS})
# The CTE root, so shared adapter headers resolve as "adapter/<name>.h" -- the
# config-string grammar is shared with the VOL connector and must not be
# duplicated: one CLIO config dialect, one parser. This spelling used to arrive
# transitively (clio_cte_cfs_adapter exported ${CLIO_CTE_ROOT} PUBLIC); that
# target is gone, so name it here as the other adapters now do.
target_include_directories(clio_vfd PRIVATE ${CLIO_CTE_ROOT})
target_link_libraries(clio_vfd
  clio_cte_filesystem_client  # descriptor layer + chimod client
  clio_cte_core_client
  ctp::cxx
  ctp::interceptor
  ${HDF5_LIBRARIES})

# Flips CLIO_VFD_API (H5FDclio.h) to dllexport for this target; every consumer
# leaves it undefined and gets dllimport.
target_compile_definitions(clio_vfd PRIVATE CLIO_VFD_BUILD)

if(WIN32 AND NOT HDF5_USE_STATIC_LIBRARIES)
  # HDF5's public GLOBALS -- H5P_CLS_FILE_ACCESS_ID_g, which H5Pset_fapl_clio
  # reaches through the H5P_FILE_ACCESS macro -- are declared H5_DLLVAR, and
  # H5api_adpt.h only expands that to __declspec(dllimport) when the CONSUMER
  # defines H5_BUILT_AS_DYNAMIC_LIB. Without it MSVC emits a plain extern and
  # the link fails with LNK2019 on that symbol; functions are unaffected, so
  # the build gets all the way to linking before anything complains.
  #
  # Guarded on HDF5_USE_STATIC_LIBRARIES because the macro would be wrong for a
  # static HDF5, where the globals are ordinary externs.
  target_compile_definitions(clio_vfd PRIVATE H5_BUILT_AS_DYNAMIC_LIB)
endif()

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
