# thread: locks and the CPU/GPU thread models
clio_ctp_add_sources(src/thread_model.cc)

if(CLIO_CORE_ENABLE_TESTS)
    add_subdirectory(test)
endif()
