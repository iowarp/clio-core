#------------------------------------------------------------------------------
# Write-through policy cache chimod. All logic lives in cache_base/
# (shared with the other policy cache); this module only fixes the policy.
#------------------------------------------------------------------------------
set(CLIO_CTE_CACHE_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../cache_base)

add_clio_module_runtime(
  SOURCES
    src/core_persist_write_thru_runtime.cc
    ${CLIO_CTE_CACHE_BASE_DIR}/src/cache_base.cc
)

add_clio_module_client(
  SOURCES
    src/core_persist_write_thru_client.cc
)

target_link_libraries(clio_cte_core_persist_write_thru_runtime PUBLIC clio_cte_core_client)
target_link_libraries(clio_cte_core_persist_write_thru_client PUBLIC clio_cte_core_client)
foreach(tgt clio_cte_core_persist_write_thru_runtime clio_cte_core_persist_write_thru_client)
  target_include_directories(${tgt} PUBLIC
      $<BUILD_INTERFACE:${CLIO_CTE_CACHE_BASE_DIR}/include>)
endforeach()
install(DIRECTORY ${CLIO_CTE_CACHE_BASE_DIR}/include/
        DESTINATION include)
