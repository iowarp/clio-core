# ------------------------------------------------------------------------------
# LAMMPS as a library, Clio in the same process (issue #693).
#
# Links a prebuilt liblammps into a small driver that hosts Clio's runtime
# in-process and stores Atom::x/v/f through the compressor between `run`
# segments. No HDF5 in the path, no LAMMPS patch -- the LAMMPS build here is
# the same stock one ../neuropress_lammps_h5 drives as an executable.
#
# Needs a CMake build of LAMMPS WITH KOKKOS/CUDA. Point these at it (defaults
# match the sibling examples' ~/src/lammps/build):
#   -DLAMMPS_SRC_DIR=<tree>      for src/library.h and src/KOKKOS/atom_kokkos.h
#   -DLAMMPS_BUILD_DIR=<build>   for liblammps.{a,so} and its Kokkos archives
# Skips itself, with a message, when any of that is missing -- a checkout
# without LAMMPS, or with a non-Kokkos LAMMPS, still configures.
#
# Kokkos is required because `--order device` reads LAMMPS' Kokkos device
# views through lammps_device_view.cc. That one file cannot be compiled by a
# host compiler (Kokkos_Setup_Cuda.hpp errors out without __CUDACC__), so it
# is built by invoking LAMMPS' OWN compiler launcher with LAMMPS' OWN flags,
# read out of its build tree -- the same trick this file already uses to
# recover a static liblammps' link line.
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

# ------------------------------------------------------------------------------
# The Kokkos translation unit.
#
# LAMMPS compiles its own KOKKOS sources as:
#   <kokkos>/bin/kokkos_launch_compiler <kokkos>/bin/nvcc_wrapper c++ c++ \
#       $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -c ...
# Reuse exactly that, read from the build tree, rather than reconstructing a
# Kokkos/CUDA flag set here. Guessing it wrong does not fail the build --
# LMP_KOKKOS_NO_LEGACY, for instance, silently changes tdual_x_array's layout,
# which lammps_device_view.cc static_asserts precisely because the wrong answer
# is undetectable at runtime.
# ------------------------------------------------------------------------------
set(_lmp_kk_launcher ${LAMMPS_SRC_DIR}/lib/kokkos/bin/kokkos_launch_compiler)
set(_lmp_nvcc_wrapper ${LAMMPS_SRC_DIR}/lib/kokkos/bin/nvcc_wrapper)
set(_lmp_flags_make ${LAMMPS_BUILD_DIR}/CMakeFiles/lammps.dir/flags.make)

if(NOT EXISTS ${LAMMPS_SRC_DIR}/src/KOKKOS/atom_kokkos.h
   OR NOT EXISTS ${_lmp_kk_launcher}
   OR NOT EXISTS ${_lmp_nvcc_wrapper}
   OR NOT EXISTS ${_lmp_flags_make})
  message(STATUS "NeuroPress LAMMPS library example: skipped (LAMMPS at "
                 "${LAMMPS_SRC_DIR} has no KOKKOS package, or its build at "
                 "${LAMMPS_BUILD_DIR} predates one; --order device needs "
                 "src/KOKKOS/atom_kokkos.h, lib/kokkos/bin/ and "
                 "CMakeFiles/lammps.dir/flags.make)")
  return()
endif()

# CXX_DEFINES / CXX_INCLUDES / CXX_FLAGS, verbatim, as LAMMPS built its own
# KOKKOS sources with. separate_arguments so a flag containing spaces does not
# become two.
file(STRINGS ${_lmp_flags_make} _lmp_flag_lines)
set(_lmp_kk_defines "")
set(_lmp_kk_includes "")
set(_lmp_kk_cxxflags "")
foreach(_line IN LISTS _lmp_flag_lines)
  if(_line MATCHES "^CXX_DEFINES = (.*)$")
    set(_lmp_kk_defines "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^CXX_INCLUDES = (.*)$")
    set(_lmp_kk_includes "${CMAKE_MATCH_1}")
  elseif(_line MATCHES "^CXX_FLAGS = (.*)$")
    set(_lmp_kk_cxxflags "${CMAKE_MATCH_1}")
  endif()
endforeach()
if(_lmp_kk_includes STREQUAL "")
  message(STATUS "NeuroPress LAMMPS library example: skipped (could not read "
                 "CXX_INCLUDES from ${_lmp_flags_make})")
  return()
endif()
separate_arguments(_lmp_kk_defines UNIX_COMMAND "${_lmp_kk_defines}")
separate_arguments(_lmp_kk_includes UNIX_COMMAND "${_lmp_kk_includes}")
separate_arguments(_lmp_kk_cxxflags UNIX_COMMAND "${_lmp_kk_cxxflags}")

set(_lmp_device_obj ${CMAKE_CURRENT_BINARY_DIR}/lammps_device_view.o)
add_custom_command(
  OUTPUT ${_lmp_device_obj}
  COMMAND ${_lmp_kk_launcher} ${_lmp_nvcc_wrapper} ${CMAKE_CXX_COMPILER}
          ${CMAKE_CXX_COMPILER}
          ${_lmp_kk_defines} ${_lmp_kk_includes} ${_lmp_kk_cxxflags}
          -c ${CMAKE_CURRENT_SOURCE_DIR}/lammps_device_view.cc
          -o ${_lmp_device_obj}
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lammps_device_view.cc ${_lmp_flags_make}
  COMMENT "Building lammps_device_view.o with LAMMPS' Kokkos compiler"
  VERBATIM)
set_source_files_properties(${_lmp_device_obj} PROPERTIES
  EXTERNAL_OBJECT TRUE GENERATED TRUE)

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

add_executable(neuropress_lammps_lib
  neuropress_lammps_lib.cc
  ${_lmp_device_obj})

target_include_directories(neuropress_lammps_lib PRIVATE
  ${LAMMPS_SRC_DIR}/src)

# neuropress_lammps_lib.cc itself still sees library.h only -- no Kokkos
# headers, no nvcc, plain C++ against the C API. Everything that needs Kokkos
# is behind the extern "C" surface in the prebuilt object above.
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
