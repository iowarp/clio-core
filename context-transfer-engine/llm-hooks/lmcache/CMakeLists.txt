cmake_minimum_required(VERSION 3.20)

set(LMCACHE_SOURCES
    src/lmcache_store.cc
)

add_library(clio_llm_lmcache SHARED ${LMCACHE_SOURCES})

target_include_directories(clio_llm_lmcache
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(clio_llm_lmcache
    PUBLIC
        clio_cte_core_client
)

target_compile_features(clio_llm_lmcache PUBLIC cxx_std_17)

install(TARGETS clio_llm_lmcache
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

if(CLIO_CORE_ENABLE_PYTHON)
    if(NOT COMMAND nanobind_add_module)
        message(WARNING "llm-hooks/lmcache: Python bindings skipped because nanobind_add_module is unavailable.")
    else()
        nanobind_add_module(
            clio_cte_lmcache_ext
            python/lmcache_bindings.cc
        )
        target_link_libraries(clio_cte_lmcache_ext PRIVATE clio_llm_lmcache)
        target_compile_features(clio_cte_lmcache_ext PRIVATE cxx_std_17)
        install(
            TARGETS clio_cte_lmcache_ext
            LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}"
        )
        message(STATUS "CTE LMCache Python extension will be installed to: ${PYTHON_SITE_PACKAGES}")
    endif()
endif()

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
