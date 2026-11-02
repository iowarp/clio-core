# memory: memory backends, allocators and shared-memory smart pointers
clio_ctp_add_sources(src/malloc_allocator.cc)

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
