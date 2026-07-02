# Find FUSE3.
#   Linux:   libfuse3, discovered via pkg-config.
#   Windows: WinFsp ships a FUSE3-compatible header (inc/fuse3/fuse.h) and an
#            import library (lib/winfsp-x64.lib) but NO pkg-config file, so
#            locate it directly. Default install dir is "%ProgramFiles(x86)%\
#            WinFsp"; override with -DWINFSP_ROOT=... if installed elsewhere.
# Both branches set FUSE3_FOUND, FUSE3_INCLUDEDIR and FUSE3_LIBRARIES so the
# target wiring below is identical on either platform.
if(WIN32)
  if(NOT WINFSP_ROOT)
    set(WINFSP_ROOT "$ENV{ProgramFiles\(x86\)}/WinFsp")
  endif()
  find_path(FUSE3_INCLUDEDIR
    NAMES fuse3/fuse.h
    HINTS "${WINFSP_ROOT}/inc")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(_winfsp_lib_names winfsp-x64)
  else()
    set(_winfsp_lib_names winfsp-x86)
  endif()
  find_library(FUSE3_LIBRARIES
    NAMES ${_winfsp_lib_names} winfsp
    HINTS "${WINFSP_ROOT}/lib")
  if(FUSE3_INCLUDEDIR AND FUSE3_LIBRARIES)
    set(FUSE3_FOUND TRUE)
    set(FUSE3_VERSION "WinFsp")
  endif()
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FUSE3 QUIET fuse3)
endif()

if(NOT FUSE3_FOUND)
  if(WIN32)
    message(WARNING "WinFsp not found. FUSE adapter will not be built.")
    message(STATUS "Install WinFsp (https://winfsp.dev) or pass -DWINFSP_ROOT=<dir>")
  else()
    message(WARNING "FUSE3 not found. FUSE adapter will not be built.")
    message(STATUS "To install FUSE3: sudo apt install libfuse3-dev")
  endif()
  return()
endif()

message(STATUS "Found FUSE3: ${FUSE3_VERSION}")

add_executable(clio_cte_fuse
    fuse_cte.cc)

target_compile_definitions(clio_cte_fuse PRIVATE
    FUSE_USE_VERSION=35
    CLIO_CTE_FUSE_ENABLED)

target_include_directories(clio_cte_fuse PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CLIO_CTE_ROOT}
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/filesystem/include
    ${FUSE3_INCLUDE_DIRS}
    ${FUSE3_INCLUDEDIR})  # parent of include/fuse3 so
                          # `#include <fuse3/fuse.h>` resolves

target_link_directories(clio_cte_fuse PRIVATE ${FUSE3_LIBRARY_DIRS})
target_link_libraries(clio_cte_fuse
    ${FUSE3_LIBRARIES}
    clio_cte_filesystem_client  # the FUSE adapter now delegates to the chimod
    clio_cte_core_client
    ctp::cxx)

install(TARGETS clio_cte_fuse
    RUNTIME DESTINATION bin)
