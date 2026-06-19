# Find FUSE3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FUSE3 fuse3)

if(NOT FUSE3_FOUND)
  message(FATAL_ERROR "FUSE3 not found but CLIO_CTE_ENABLE_FUSE_ADAPTER=ON. "
    "Install it with: sudo apt install libfuse3-dev fuse3")
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
