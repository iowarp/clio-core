cmake_minimum_required(VERSION 3.10)

#------------------------------------------------------------------------------
# Multi-Threaded Compression Benchmark System
# Generates 100K+ compression results for ML training
#------------------------------------------------------------------------------

add_executable(wrp_cte_compression_benchmark
    generate_training_dataset.cc
)

target_include_directories(wrp_cte_compression_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
)

target_compile_features(wrp_cte_compression_benchmark PRIVATE cxx_std_17)

# Link compression libraries
target_link_libraries(wrp_cte_compression_benchmark PRIVATE
    hshm::compress
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
)

# Enable compression support
target_compile_definitions(wrp_cte_compression_benchmark PRIVATE
    HSHM_ENABLE_COMPRESS=1
)

# Add LibPressio support if available
find_package(LibPressio QUIET)
if(LibPressio_FOUND)
    target_link_libraries(wrp_cte_compression_benchmark PRIVATE LibPressio::libpressio)
    target_compile_definitions(wrp_cte_compression_benchmark PRIVATE HSHM_ENABLE_LIBPRESSIO=1)
    message(STATUS "LibPressio found - lossy compression enabled")
else()
    message(STATUS "LibPressio not found - only lossless compression available")
endif()

install(TARGETS wrp_cte_compression_benchmark
    RUNTIME DESTINATION bin
)

# Also create alias for backward compatibility
add_executable(wrp_cte_generate_dataset ALIAS wrp_cte_compression_benchmark)

message(STATUS "Multi-threaded compression benchmark configured:")
message(STATUS "  - 30+ compression configurations")
message(STATUS "  - 34 size bins (1KB-4MB)")
message(STATUS "  - 64 feature-targeted data bins")
message(STATUS "  - Multi-threaded work-stealing execution")
message(STATUS "  - Output: compression_results.csv")

#------------------------------------------------------------------------------
# Gray-Scott Real Data Compression Analysis Tool
#------------------------------------------------------------------------------

# Find ADIOS2 (only if not already found)
if(NOT TARGET adios2::cxx11)
    find_package(ADIOS2 REQUIRED)
endif()

add_executable(gray_scott_analysis
    gray_scott_analysis.cc
)

target_include_directories(gray_scott_analysis PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/context-transfer-engine/compressor/include
    ${CMAKE_SOURCE_DIR}/context-transport-primitives/include
)

target_link_libraries(gray_scott_analysis PRIVATE
    adios2::cxx
    hshm::compress
    hshm::cxx
    ${CMAKE_THREAD_LIBS_INIT}
    curl
)

# Enable compression support
target_compile_definitions(gray_scott_analysis PRIVATE
    HSHM_ENABLE_COMPRESS=1
)

# Add LibPressio support if available
find_package(LibPressio QUIET)
if(LibPressio_FOUND)
    target_link_libraries(gray_scott_analysis PRIVATE LibPressio::libpressio)
    target_compile_definitions(gray_scott_analysis PRIVATE HSHM_ENABLE_LIBPRESSIO=1)
    message(STATUS "Gray-Scott: LibPressio found - lossy compression enabled")
else()
    message(STATUS "Gray-Scott: LibPressio not found - only lossless compression available")
endif()

target_compile_features(gray_scott_analysis PRIVATE cxx_std_17)

install(TARGETS gray_scott_analysis
    RUNTIME DESTINATION bin
)

message(STATUS "Gray-Scott analysis tool configured with ADIOS2 support")
