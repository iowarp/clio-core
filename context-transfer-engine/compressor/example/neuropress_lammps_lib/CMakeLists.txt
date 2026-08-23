# ------------------------------------------------------------------------------
# LAMMPS as a library, Clio in the same process (issue #693).
#
# Links a prebuilt liblammps into a small driver that hosts Clio's runtime
# in-process and stores Atom::x/v/f through the compressor between `run`
# segments. No HDF5 in the path, no LAMMPS patch -- the LAMMPS build here is
# the same stock one ../neuropress_lammps_h5 drives as an executable.
#
# Needs a CMake build of LAMMPS. Point these at it (defaults match the
# sibling examples' ~/src/lammps/build):
#   -DLAMMPS_SRC_DIR=<tree>      for src/library.h
#   -DLAMMPS_BUILD_DIR=<build>   for liblammps.{a,so}
# Skips itself, with a message, when either is missing -- a checkout without
# LAMMPS still configures.
# ------------------------------------------------------------------------------

set(LAMMPS_SRC_DIR "$ENV{HOME}/src/lammps" CACHE PATH
    "LAMMPS source tree (provides src/library.h)")
set(LAMMPS_BUILD_DIR "${LAMMPS_SRC_DIR}/build" CACHE PATH
    "LAMMPS CMake build directory (provides liblammps.a or liblammps.so)")

if(NOT EXISTS ${LAMMPS_SRC_DIR}/src/library.h)
  message(STATUS "NeuroPress LAMMPS library example: skipped (no "
                 "${LAMMPS_SRC_DIR}/src/library.h; set LAMMPS_SRC_DIR)")
  return()
endif()

find_library(LAMMPS_LIBRARY NAMES lammps
  PATHS ${LAMMPS_BUILD_DIR} NO_DEFAULT_PATH)
if(NOT LAMMPS_LIBRARY)
  message(STATUS "NeuroPress LAMMPS library example: skipped (no liblammps in "
                 "${LAMMPS_BUILD_DIR}; set LAMMPS_BUILD_DIR)")
  return()
endif()

# A STATIC liblammps does not carry its own dependencies (Kokkos, CUDA,
# HDF5 for the H5MD package, OpenMP, jpeg...). Rather than guess at them,
# take them from the link line LAMMPS' own build used for `lmp`: every
# library token after the objects, in order, with the build-relative Kokkos
# archives made absolute. A shared liblammps needs none of this.
set(_lmp_deps "")
if(LAMMPS_LIBRARY MATCHES "\\.a$")
  set(_link_txt ${LAMMPS_BUILD_DIR}/CMakeFiles/lmp.dir/link.txt)
  if(NOT EXISTS ${_link_txt})
    message(STATUS "NeuroPress LAMMPS library example: skipped (static "
                   "liblammps but no ${_link_txt} to read its deps from)")
    return()
  endif()
  file(READ ${_link_txt} _link)
  string(REGEX REPLACE "[ \t\r\n]+" ";" _toks "${_link}")
  foreach(t IN LISTS _toks)
    if(t MATCHES "\\.o$" OR t MATCHES "^-o$" OR t STREQUAL "lmp")
      continue()
    endif()
    if(t MATCHES "\\.a$" OR t MATCHES "\\.so(\\.[0-9]+)*$" OR t MATCHES "^-l")
      if(NOT t MATCHES "^-l" AND NOT IS_ABSOLUTE "${t}")
        set(t "${LAMMPS_BUILD_DIR}/${t}")
      endif()
      list(APPEND _lmp_deps "${t}")
    endif()
  endforeach()
  # liblammps.a itself is the first token; keep it there and drop our copy.
  list(REMOVE_ITEM _lmp_deps "${LAMMPS_LIBRARY}")
endif()

message(STATUS "NeuroPress LAMMPS library example: enabled (${LAMMPS_LIBRARY})")

add_executable(neuropress_lammps_lib neuropress_lammps_lib.cc)

target_include_directories(neuropress_lammps_lib PRIVATE
  ${LAMMPS_SRC_DIR}/src)

# library.h only: no Kokkos headers, no nvcc, plain C++ against the C API.
target_link_libraries(neuropress_lammps_lib
  ${LAMMPS_LIBRARY}
  ${_lmp_deps}
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

install(TARGETS neuropress_lammps_lib RUNTIME DESTINATION bin)
