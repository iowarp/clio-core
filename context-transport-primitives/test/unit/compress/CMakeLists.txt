project(hermes_shm)

#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------

# Basic compression unit tests
add_executable(test_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress.cc)

# Note: Unified compression benchmark (benchmark_compression_unified_exec) has been removed
# Note: AI model predictor tests (Q-Table, LinReg, Distribution, Contention)
# have been moved to context-transfer-engine/compressor/test

#------------------------------------------------------------------------------
# Link Dependencies
#------------------------------------------------------------------------------

add_dependencies(test_compress_exec hermes_shm_host)
target_link_libraries(test_compress_exec
        hermes_shm_host hshm::compress Catch2::Catch2)
target_compile_definitions(test_compress_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------

add_test(NAME ctp_compress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_exec "~[error=FatalError]")

#------------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------

install(TARGETS
        test_compress_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#------------------------------------------------------------------------------
