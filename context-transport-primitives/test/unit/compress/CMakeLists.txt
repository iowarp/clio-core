project(hermes_shm)



#------------------------------------------------------------------------------
# Build Tests
#------------------------------------------------------------------------------
add_executable(test_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress.cc)

# LibPressio test commented out - source file doesn't exist
# add_executable(test_compress_libpressio_exec
#         ${TEST_MAIN}/main.cc
#         test_init.cc
#         test_compress_libpressio.cc)

add_executable(benchmark_compress_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress_benchmark.cc)

add_executable(benchmark_compress_parameter_study_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress_parameter_study.cc)

add_executable(benchmark_compress_lossy_parameter_study_exec
        ${TEST_MAIN}/main.cc
        test_init.cc
        test_compress_lossy_parameter_study.cc)

# Compression predictor tests (C++ ML models)
add_executable(test_xgboost_predictor
        test_xgboost_predictor.cc)

add_executable(test_dense_nn_predictor
        test_dense_nn_predictor.cc)

add_dependencies(test_compress_exec hermes_shm_host)
target_link_libraries(test_compress_exec
        hermes_shm_host hshm::compress Catch2::Catch2)
target_compile_definitions(test_compress_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

# add_dependencies(test_compress_libpressio_exec hermes_shm_host)
# target_link_libraries(test_compress_libpressio_exec
#         hermes_shm_host hshm::compress Catch2::Catch2)
# target_compile_definitions(test_compress_libpressio_exec
#         PRIVATE HSHM_ENABLE_COMPRESS=1)

add_dependencies(benchmark_compress_exec hermes_shm_host)
target_link_libraries(benchmark_compress_exec
        hermes_shm_host hshm::compress Catch2::Catch2)
target_compile_definitions(benchmark_compress_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

add_dependencies(benchmark_compress_parameter_study_exec hermes_shm_host)
target_link_libraries(benchmark_compress_parameter_study_exec
        hermes_shm_host hshm::compress Catch2::Catch2)
target_compile_definitions(benchmark_compress_parameter_study_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

add_dependencies(benchmark_compress_lossy_parameter_study_exec hermes_shm_host)
# Link lossy compression libraries directly
find_library(ZFP_LIBRARY NAMES zfp PATHS /usr/local/lib)
find_library(SZ3_LIBRARY NAMES SZ3c PATHS /usr/local/lib)
find_library(SZ_LIBRARY NAMES SZ PATHS /usr/local/lib)
find_library(FPZIP_LIBRARY NAMES fpzip PATHS /usr/local/lib)

set(LOSSY_LIBS "")
if(ZFP_LIBRARY)
    list(APPEND LOSSY_LIBS ${ZFP_LIBRARY})
    message(STATUS "Found ZFP library: ${ZFP_LIBRARY}")
endif()
if(SZ3_LIBRARY)
    list(APPEND LOSSY_LIBS ${SZ3_LIBRARY})
    message(STATUS "Found SZ3 library: ${SZ3_LIBRARY}")
endif()
if(SZ_LIBRARY)
    list(APPEND LOSSY_LIBS ${SZ_LIBRARY})
    message(STATUS "Found SZ library: ${SZ_LIBRARY}")
endif()
if(FPZIP_LIBRARY)
    list(APPEND LOSSY_LIBS ${FPZIP_LIBRARY})
    message(STATUS "Found FPZIP library: ${FPZIP_LIBRARY}")
endif()

target_link_libraries(benchmark_compress_lossy_parameter_study_exec
        hermes_shm_host hshm::compress Catch2::Catch2 ${LOSSY_LIBS})
target_compile_definitions(benchmark_compress_lossy_parameter_study_exec
        PRIVATE HSHM_ENABLE_COMPRESS=1)

# Link XGBoost predictor test
find_package(xgboost QUIET)
if(xgboost_FOUND AND TARGET xgboost_predictor)
    target_link_libraries(test_xgboost_predictor
            xgboost_predictor)
    set_property(TARGET test_xgboost_predictor PROPERTY CXX_STANDARD 17)
    message(STATUS "XGBoost predictor test enabled")
else()
    message(STATUS "XGBoost predictor not available, test will not be built")
    # Create a dummy target that does nothing
    set_target_properties(test_xgboost_predictor PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

# Link Dense NN predictor test (uses MiniDNN + Eigen)
find_package(Eigen3 QUIET)
find_package(nlohmann_json QUIET)
set(MINIDNN_INCLUDE_DIR "${HSHM_ROOT}/external/MiniDNN/include")
if(EXISTS "${MINIDNN_INCLUDE_DIR}/MiniDNN.h")
    set(MINIDNN_FOUND TRUE)
else()
    set(MINIDNN_FOUND FALSE)
endif()

if(Eigen3_FOUND AND MINIDNN_FOUND AND nlohmann_json_FOUND AND TARGET dense_nn_predictor)
    target_link_libraries(test_dense_nn_predictor
            dense_nn_predictor)
    target_include_directories(test_dense_nn_predictor PRIVATE
            ${MINIDNN_INCLUDE_DIR})
    set_property(TARGET test_dense_nn_predictor PROPERTY CXX_STANDARD 17)
    message(STATUS "Dense NN predictor test enabled with MiniDNN + Eigen")
else()
    message(STATUS "Dense NN predictor not available (requires Eigen3 + MiniDNN + nlohmann_json)")
    # Create a dummy target that does nothing
    set_target_properties(test_dense_nn_predictor PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

add_test(NAME ctp_compress COMMAND
        ${CMAKE_BINARY_DIR}/bin/test_compress_exec "~[error=FatalError]")

#-----------------------------------------------------------------------------
# Install Targets
#------------------------------------------------------------------------------
install(TARGETS
        test_compress_exec
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

#-----------------------------------------------------------------------------
# Coverage
#-----------------------------------------------------------------------------
if(HSHM_ENABLE_COVERAGE)
    set_coverage_flags(test_compress_exec)
endif()
