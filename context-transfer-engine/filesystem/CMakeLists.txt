cmake_minimum_required(VERSION 3.10)
project(clio_cte_filesystem)

#------------------------------------------------------------------------------
# Filesystem chimod — a filesystem interface over the CTE (sibling to the
# compressor chimod). Built via the standard ChiMod build functions.
#------------------------------------------------------------------------------

add_clio_module_runtime(
  SOURCES
    src/filesystem_runtime.cc
    src/autogen/filesystem_lib_exec.cc
)

add_clio_module_client(
  SOURCES
    src/filesystem_client.cc
)

# The filesystem chimod is an interface over the CTE core — link its client
# (and pull in the core headers for TagId/Context/the core client API).
target_link_libraries(clio_cte_filesystem_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_filesystem_client PUBLIC clio_cte_core_client)
target_include_directories(clio_cte_filesystem_runtime PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)
target_include_directories(clio_cte_filesystem_client PRIVATE
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/core/include)

# Export the g_fs_client global on Windows. This must be THIS module's macro:
# defining CLIO_CTE_CORE_BUILDING_DLL here also told the core's headers that
# g_cte_client was being defined locally, which it is not -- an LNK2019 that
# stayed hidden while the descriptor layer (its only user here) was excluded
# from Windows builds.
if(TARGET clio_cte_filesystem_client)
  target_compile_definitions(clio_cte_filesystem_client PRIVATE
    CLIO_CTE_FS_BUILDING_DLL=1)
endif()
