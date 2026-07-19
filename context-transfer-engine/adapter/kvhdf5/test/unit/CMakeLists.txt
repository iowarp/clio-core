# CPU-only unit tests. No CUDA language, no clio runtime, no server process —
# just cuda::std headers (libcu++ from the CUDA toolkit) + Catch2. These drive
# the same chunk/metadata logic the GPU producer path reuses. CUDAToolkit is
# found here (idempotent) only for its include dir, which is where <cuda/std/*>
# lives; the tests compile as plain C++.
find_package(CUDAToolkit REQUIRED)

add_executable(kvhdf5_unit_tests
    chunking_test.cpp
    dataset_io_test.cpp
    tag_path_test.cpp
)

set_target_properties(kvhdf5_unit_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_include_directories(kvhdf5_unit_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(kvhdf5_unit_tests PRIVATE
    Catch2::Catch2WithMain
)

catch_discover_tests(kvhdf5_unit_tests)
