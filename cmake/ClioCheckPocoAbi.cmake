# ClioCheckPocoAbi -- refuse to configure against a Poco whose headers and
# library come from different installations.
#
# WHY. Poco's Net classes (HTTPClientSession, HTTPRequest, HTTPResponse) are
# built on the caller's stack, so their layout comes from the headers while the
# code that fills them in comes from the library. When the two are different
# releases the library writes past the header-sized object. Nothing fails at
# compile or link time; the process dies later with "stack smashing detected".
# Issue #762: a conda env provided libPocoNet 1.14.2 but an incomplete set of
# Poco headers, so `#include <Poco/Net/HTTPClientSession.h>` silently fell
# through to the 1.11.0 headers in /usr/include.
#
# HOW. Two checks, neither of which runs a program (so they also work when
# cross compiling and on Windows, where a probe would not find the Poco DLLs):
#   1. Every Poco header the tree includes must exist under the include
#      directories of the Poco targets that find_package located. A header
#      missing there is what lets the compiler fall through to another copy.
#   2. A compile-time check that POCO_VERSION in those headers equals the
#      version of the Poco package (and so the library) that was found.

# Poco headers included by this tree; each must come from the found package.
set(CLIO_POCO_REQUIRED_HEADERS
  Poco/Version.h
  Poco/Environment.h
  Poco/Net/HTTPClientSession.h
  Poco/Net/HTTPRequest.h
  Poco/Net/HTTPResponse.h)

#------------------------------------------------------------------------------
# clio_check_poco_abi()
#
# Verify that the Poco headers the compiler resolves belong to the same Poco
# installation and release as the library find_package(Poco) located. Call
# after find_package(Poco ...). Does nothing when Poco was not found or carries
# no version. Emits FATAL_ERROR naming the offending header or both versions.
#------------------------------------------------------------------------------
function(clio_check_poco_abi)
  if(NOT Poco_FOUND OR NOT TARGET Poco::Net OR NOT TARGET Poco::Foundation)
    return()
  endif()

  # 1. Headers must be present in the found package's own include dirs.
  set(_inc_dirs "")
  foreach(_tgt Poco::Foundation Poco::Net)
    get_target_property(_dirs ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
    if(_dirs)
      list(APPEND _inc_dirs ${_dirs})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _inc_dirs)
  foreach(_hdr ${CLIO_POCO_REQUIRED_HEADERS})
    unset(_found_hdr)
    find_file(_found_hdr ${_hdr} PATHS ${_inc_dirs} NO_DEFAULT_PATH NO_CACHE)
    if(NOT _found_hdr)
      message(FATAL_ERROR
        "Poco ${Poco_VERSION} was found (Poco_DIR=${Poco_DIR}) but its include "
        "directories (${_inc_dirs}) do not contain ${_hdr}. The compiler would "
        "take that header from some other Poco installation, whose object "
        "layouts need not match this library (stack smashing at run time, "
        "issue #762). Install the complete Poco headers alongside the library "
        "or point CMAKE_PREFIX_PATH at a complete installation.")
    endif()
  endforeach()

  # 2. POCO_VERSION in the headers must equal the found package version.
  if(NOT Poco_VERSION MATCHES "^([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    message(STATUS "Poco header/library version check: skipped "
                   "(package version '${Poco_VERSION}' not parseable)")
    return()
  endif()
  math(EXPR _want "(${CMAKE_MATCH_1} << 16) | (${CMAKE_MATCH_2} << 8) | ${CMAKE_MATCH_3}"
       OUTPUT_FORMAT HEXADECIMAL)
  set(_probe_src "${CMAKE_BINARY_DIR}/CMakeFiles/ClioCheckPocoAbi/poco_version_probe.cc")
  file(WRITE "${_probe_src}" "
#include <Poco/Version.h>
static_assert((POCO_VERSION >> 8) == ${_want},
              \"POCO_VERSION in the headers differs from the Poco package\");
int main() { return 0; }
")
  try_compile(_version_ok
    "${CMAKE_BINARY_DIR}/CMakeFiles/ClioCheckPocoAbi/build" "${_probe_src}"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${_inc_dirs}"
    # static_assert needs C++11, and Poco 1.15 headers need C++17. Without
    # this the probe builds at the compiler default -- C++98 for Apple clang
    # -- and fails as a bogus "version mismatch" (#1261, macOS wheel).
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_VARIABLE _probe_out)
  if(NOT _version_ok)
    message(FATAL_ERROR
      "Poco header/library version mismatch: the Poco package found at "
      "${Poco_DIR} is ${Poco_VERSION}, but the Poco/Version.h the compiler "
      "resolves is a different release. Code that builds Poco::Net objects on "
      "the stack would corrupt it (\"stack smashing detected\", issue #762). "
      "Make the Poco headers and libraries come from one installation.\n"
      "${_probe_out}")
  endif()
  message(STATUS "Poco header/library version check: OK (${Poco_VERSION})")
endfunction()

# clio_link_poco_private(<target>)
#
# Link <target> PRIVATE against Poco::Net and Poco::Foundation, the way every
# Clio library that uses Poco must (PRIVATE keeps Poco's _FILE_OFFSET_BITS
# INTERFACE definitions away from the POSIX interception adapters).
#
# When Poco is a static library on Linux (the pip wheel builds it that way,
# because the wheel does not bundle third-party .so files), each shared library
# that links it embeds its own copy. Poco's headers force default visibility
# on its symbols, so both copies would be exported and the dynamic linker would
# bind one library's Poco globals onto the other's -- their destructors then run
# twice at exit ("free(): double free detected", wheel import aborts). Linking
# with --exclude-libs keeps each embedded copy's symbols local to its library.
#
# Arguments:
#   target - the shared/static library or executable that uses Poco::Net.
function(clio_link_poco_private target)
  target_link_libraries(${target} PRIVATE Poco::Net Poco::Foundation)
  get_target_property(_poco_type Poco::Foundation TYPE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND _poco_type STREQUAL "STATIC_LIBRARY")
    target_link_options(${target} PRIVATE
      "LINKER:--exclude-libs,libPocoNet.a:libPocoFoundation.a:libPocoNetd.a:libPocoFoundationd.a")
  endif()
endfunction()
