# CheckPerConfigArchiveLayout.cmake
# Regression check for issue #528: on multi-config generators (Visual Studio,
# Xcode), static archives must live in per-configuration directories
# (bin/<CONFIG>) so that building one configuration cannot clobber the
# archives of another, while runtime artifacts (.exe/.dll) stay in the flat
# bin/ directory that tests and the module loader rely on.
#
# Invoked as a CTest test via:
#   cmake -DARCHIVE_FILE=<path> -DEXE_FILE=<path> -DBIN_DIR=<path>
#         -DCONFIG=<config> -P CheckPerConfigArchiveLayout.cmake
#
# Arguments:
#   ARCHIVE_FILE - $<TARGET_FILE:...> of a static library in the active config
#   EXE_FILE     - $<TARGET_FILE:...> of an executable in the active config
#   BIN_DIR      - ${CMAKE_BINARY_DIR}/bin
#   CONFIG       - $<CONFIG>, the active configuration name

foreach(_var ARCHIVE_FILE EXE_FILE BIN_DIR CONFIG)
  if(NOT DEFINED ${_var})
    message(FATAL_ERROR "Missing required argument -D${_var}=...")
  endif()
endforeach()

file(TO_CMAKE_PATH "${ARCHIVE_FILE}" _archive_file)
file(TO_CMAKE_PATH "${EXE_FILE}" _exe_file)
file(TO_CMAKE_PATH "${BIN_DIR}" _bin_dir)

if(NOT EXISTS "${_archive_file}")
  message(FATAL_ERROR "Archive does not exist: ${_archive_file}")
endif()
if(NOT EXISTS "${_exe_file}")
  message(FATAL_ERROR "Executable does not exist: ${_exe_file}")
endif()

get_filename_component(_archive_dir "${_archive_file}" DIRECTORY)
get_filename_component(_exe_dir "${_exe_file}" DIRECTORY)

if(NOT _archive_dir STREQUAL "${_bin_dir}/${CONFIG}")
  message(FATAL_ERROR
    "Static archive is not in a per-configuration directory.\n"
    "  expected dir: ${_bin_dir}/${CONFIG}\n"
    "  actual dir:   ${_archive_dir}\n"
    "Sharing one archive path across configurations lets one configuration "
    "overwrite another's .lib, which breaks the next link with LNK2038 "
    "runtime-library mismatches (issue #528).")
endif()

if(NOT _exe_dir STREQUAL "${_bin_dir}")
  message(FATAL_ERROR
    "Executable is not in the flat runtime directory.\n"
    "  expected dir: ${_bin_dir}\n"
    "  actual dir:   ${_exe_dir}\n"
    "Tests and the module loader resolve binaries at bin/ regardless of "
    "configuration.")
endif()

message(STATUS
  "Per-config archive layout OK: ${_archive_dir} (archive), ${_exe_dir} (exe)")
