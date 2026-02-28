# Find ADIOS2 (only if not already found)
if(NOT ADIOS2_FOUND)
  find_package(ADIOS2 QUIET)
  if(NOT ADIOS2_FOUND)
    message(WARNING "ADIOS2 not found. ADIOS2 adapter will not be built.")
    message(STATUS "To install ADIOS2: conda install adios2 or build from source")
    return()
  endif()
endif()

message(STATUS "Found ADIOS2")
if(ADIOS2_CONFIG)
  message(STATUS "ADIOS2 config: ${ADIOS2_CONFIG}")
endif()

# Find MPI (optional for ADIOS2 adapter)
find_package(MPI QUIET COMPONENTS CXX)
if(NOT MPI_CXX_FOUND)
  message(WARNING "MPI not found. ADIOS2 adapter will be built without MPI support.")
endif()

include_directories(${CMAKE_SOURCE_DIR})

# ------------------------------------------------------------------------------
# Set sources
# ------------------------------------------------------------------------------
set(IOWARP_ENGINE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/iowarp_engine.cc)

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------

# New CTE-based Iowarp Engine
add_library(iowarp_engine ${IOWARP_ENGINE_SRCS})
# ADIOS2 2.10.2 headers use Variable<T>() constructor syntax that is
# incompatible with C++20 on GCC 11 (triggers parsing error in Variable.h).
set_target_properties(iowarp_engine PROPERTIES CXX_STANDARD 17)
target_include_directories(iowarp_engine PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${ADIOS2_INCLUDE_DIR}
)

target_link_libraries(iowarp_engine PUBLIC
  wrp_cte::core_client
  adios2::adios2
  curl
)

# Add MPI support if available
if(MPI_CXX_FOUND)
  target_link_libraries(iowarp_engine PUBLIC MPI::MPI_CXX)
  message(STATUS "ADIOS2 adapter: MPI support enabled")
else()
  message(STATUS "ADIOS2 adapter: MPI support disabled")
endif()

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
  TARGETS
  iowarp_engine
  EXPORT
  ${WRP_CTE_EXPORTED_TARGETS}
  LIBRARY DESTINATION ${WRP_CTE_INSTALL_LIB_DIR}
  ARCHIVE DESTINATION ${WRP_CTE_INSTALL_LIB_DIR}
  RUNTIME DESTINATION ${WRP_CTE_INSTALL_BIN_DIR}
)
