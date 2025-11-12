cmake_minimum_required(VERSION 3.10)

# Create runtime library using modern ChiMod build functions
add_chimod_runtime(
  CHIMOD_NAME core
  SOURCES
    src/core_runtime.cc
    src/core_config.cc
    src/core_dpe.cc
    src/autogen/core_lib_exec.cc
)

# Create client library using modern ChiMod build functions
add_chimod_client(
  CHIMOD_NAME core
  SOURCES
    src/core_client.cc
    src/content_transfer_engine.cc
    src/tag.cc
)

# Add bdev ChiMod include directory for CTE core (unified build mode)
# In subdirectory mode, bdev headers are not automatically discovered
target_include_directories(wrp_cte_core_client PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/modules/bdev/include
)

target_include_directories(wrp_cte_core_runtime PRIVATE
  ${CMAKE_SOURCE_DIR}/runtime/modules/bdev/include
)