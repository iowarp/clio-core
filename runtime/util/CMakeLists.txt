# util/CMakeLists.txt
# Utility executables for Chimaera runtime

cmake_minimum_required(VERSION 3.10)

# Build chimaera_start_runtime
add_executable(chimaera_start_runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_start_runtime.cc
)
# Removed CHIMAERA_RUNTIME compile definition - using runtime IsRuntime() checks instead
target_link_libraries(chimaera_start_runtime
  PUBLIC chimaera_cxx
)
# Include directories are now inherited from chimaera_cxx target

# Build chimaera_stop_runtime
add_executable(chimaera_stop_runtime
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_stop_runtime.cc
)
# chimaera_stop_runtime is client-only code, so do not define CHIMAERA_RUNTIME
target_link_libraries(chimaera_stop_runtime
  PUBLIC chimaera_cxx
)
# Include directories are now inherited from chimaera_cxx target

# Build chi_refresh_repo
add_executable(chi_refresh_repo
  ${CMAKE_CURRENT_SOURCE_DIR}/chi_refresh_repo.cc
)
# chi_refresh_repo is a standalone utility, no need for CHIMAERA_RUNTIME
target_link_libraries(chi_refresh_repo
  PUBLIC hshm::cxx
)

# Build chimaera_compose
add_executable(chimaera_compose
  ${CMAKE_CURRENT_SOURCE_DIR}/chimaera_compose.cc
)
# chimaera_compose is client-only code, so do not define CHIMAERA_RUNTIME
target_link_libraries(chimaera_compose
  PUBLIC chimaera_cxx
)
# Include directories are now inherited from chimaera_cxx target

# Install utilities
install(TARGETS chimaera_start_runtime chimaera_stop_runtime chi_refresh_repo chimaera_compose
  RUNTIME DESTINATION bin
)