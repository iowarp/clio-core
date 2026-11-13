# FindDAOS.cmake -- locate the DAOS client development files.
#
# The DAOS client packages (daos-devel) ship no CMake package config and no
# pkg-config file, so this module finds the headers and libraries directly.
#
# Hints (all optional): the DAOS_ROOT CMake or environment variable, plus the
# usual CMAKE_PREFIX_PATH (which is how a Spack `daos` dependency is found).
#
# Result variables:
#   DAOS_FOUND          - headers and all three libraries were found
#   DAOS_INCLUDE_DIRS   - directory containing daos.h / daos_fs.h
#   DAOS_LIBRARIES      - libdfs, libdaos, libgurt
#
# Imported target:
#   DAOS::dfs           - the DFS file API (libdfs) together with libdaos
#                         (pool/container/event-queue API) and libgurt
#                         (d_errstr and the d_iov/d_sg_list helpers)

set(_daos_hints "")
if(DAOS_ROOT)
  list(APPEND _daos_hints "${DAOS_ROOT}")
endif()
if(DEFINED ENV{DAOS_ROOT})
  list(APPEND _daos_hints "$ENV{DAOS_ROOT}")
endif()

find_path(DAOS_INCLUDE_DIR
  NAMES daos_fs.h
  HINTS ${_daos_hints}
  PATH_SUFFIXES include)

find_library(DAOS_DFS_LIBRARY
  NAMES dfs
  HINTS ${_daos_hints}
  PATH_SUFFIXES lib64 lib)

find_library(DAOS_LIBRARY
  NAMES daos
  HINTS ${_daos_hints}
  PATH_SUFFIXES lib64 lib)

find_library(DAOS_GURT_LIBRARY
  NAMES gurt
  HINTS ${_daos_hints}
  PATH_SUFFIXES lib64 lib)

unset(_daos_hints)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DAOS
  REQUIRED_VARS DAOS_DFS_LIBRARY DAOS_LIBRARY DAOS_GURT_LIBRARY DAOS_INCLUDE_DIR)

if(DAOS_FOUND)
  set(DAOS_INCLUDE_DIRS "${DAOS_INCLUDE_DIR}")
  set(DAOS_LIBRARIES
      "${DAOS_DFS_LIBRARY}" "${DAOS_LIBRARY}" "${DAOS_GURT_LIBRARY}")
  if(NOT TARGET DAOS::dfs)
    add_library(DAOS::dfs INTERFACE IMPORTED)
    set_target_properties(DAOS::dfs PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${DAOS_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${DAOS_LIBRARIES}")
  endif()
endif()

mark_as_advanced(
  DAOS_INCLUDE_DIR DAOS_DFS_LIBRARY DAOS_LIBRARY DAOS_GURT_LIBRARY)
