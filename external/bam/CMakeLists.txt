cmake_minimum_required(VERSION 3.20)
project(bam LANGUAGES C CXX)

# Strip code coverage flags — gcov's hidden symbol visibility is incompatible
# with CUDA separable compilation shared library linking.
if(CLIO_CORE_ENABLE_COVERAGE)
  string(REPLACE "--coverage" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-fprofile-arcs" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-fprofile-arcs" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "-ftest-coverage" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
  string(REPLACE "-ftest-coverage" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
  string(REPLACE "--coverage" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
  # Remove --coverage from directory properties (set by parent add_*_options)
  get_directory_property(_dir_link_opts LINK_OPTIONS)
  if(_dir_link_opts)
    list(REMOVE_ITEM _dir_link_opts "--coverage")
    set_directory_properties(PROPERTIES LINK_OPTIONS "${_dir_link_opts}")
  endif()
  get_directory_property(_dir_compile_opts COMPILE_OPTIONS)
  if(_dir_compile_opts)
    list(REMOVE_ITEM _dir_compile_opts "--coverage" "-fprofile-arcs" "-ftest-coverage")
    # Also remove generator expressions that wrap these flags
    list(FILTER _dir_compile_opts EXCLUDE REGEX "coverage|fprofile-arcs|ftest-coverage")
    set_directory_properties(PROPERTIES COMPILE_OPTIONS "${_dir_compile_opts}")
  endif()
endif()

#------------------------------------------------------------------------------
# Options
#------------------------------------------------------------------------------
option(BAM_ENABLE_NVME "Enable NVMe backend via gpu-nvme-bridge" OFF)
option(BAM_ENABLE_TESTS "Enable BaM tests" OFF)

#------------------------------------------------------------------------------
# CUDA
#------------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(BAM_HAS_CUDA TRUE)
  message(STATUS "BaM: CUDA found — GPU page cache enabled")
else()
  set(BAM_HAS_CUDA FALSE)
  message(STATUS "BaM: CUDA not found — building host-only stubs")
endif()

#------------------------------------------------------------------------------
# Sources
#------------------------------------------------------------------------------
set(BAM_HOST_SOURCES
  src/controller.cc
)

set(BAM_CUDA_SOURCES
  src/page_cache_host.cc
  src/page_cache.cu
  src/array.cu
)

#------------------------------------------------------------------------------
# Library
#------------------------------------------------------------------------------
if(BAM_HAS_CUDA)
  add_library(bam
    ${BAM_HOST_SOURCES}
    ${BAM_CUDA_SOURCES}
  )
  set_target_properties(bam PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 17
    CXX_STANDARD 17
  )
  target_compile_options(bam PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
else()
  # Host-only build: just the controller (no GPU page cache)
  add_library(bam ${BAM_HOST_SOURCES})
  set_target_properties(bam PROPERTIES CXX_STANDARD 17)
endif()

target_include_directories(bam
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BAM_HAS_CUDA)
  target_include_directories(bam PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()

#------------------------------------------------------------------------------
# NVMe backend (optional)
#------------------------------------------------------------------------------
if(BAM_ENABLE_NVME)
  target_compile_definitions(bam PUBLIC BAM_ENABLE_NVME)

  # gpu-nvme-bridge uapi header (bundled in gnb/)
  target_include_directories(bam PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gnb)
  message(STATUS "BaM: NVMe enabled, using bundled gnb/ headers")
endif()

#------------------------------------------------------------------------------
# Install
#------------------------------------------------------------------------------
install(TARGETS bam
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/bam
  DESTINATION include
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
if(BAM_ENABLE_TESTS AND BAM_HAS_CUDA)
  add_subdirectory(test)
endif()
