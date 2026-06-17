# Try to find it on the system first
find_package(AWSSDK COMPONENTS s3 QUIET)

if(NOT AWSSDK_FOUND)
    message(STATUS "AWSSDK not found on system. Fetching from source via FetchContent (this might take a while)...")
    include(FetchContent)
    
    # We only need the s3 module
    set(BUILD_ONLY "s3" CACHE STRING "Only build s3 module" FORCE)
    set(ENABLE_TESTING OFF CACHE BOOL "Disable AWS SDK tests" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build AWS SDK statically" FORCE)
    
    FetchContent_Declare(
        AWSSDK
        GIT_REPOSITORY https://github.com/aws/aws-sdk-cpp.git
        GIT_TAG        1.11.332
    )
    FetchContent_MakeAvailable(AWSSDK)
endif()

add_executable(clio_ctp_s3_test test_s3_drive.cc)

if(AWSSDK_FOUND)
    set(AWS_S3_TARGET AWSSDK::s3)
else()
    set(AWS_S3_TARGET aws-cpp-sdk-s3 aws-cpp-sdk-core)
endif()

target_link_libraries(clio_ctp_s3_test
    PRIVATE
    Catch2::Catch2WithMain
    ${AWS_S3_TARGET}
)

if(NOT AWSSDK_FOUND)
    target_include_directories(clio_ctp_s3_test PRIVATE
        ${awssdk_SOURCE_DIR}/aws-cpp-sdk-core/include
        ${awssdk_SOURCE_DIR}/aws-cpp-sdk-s3/include
    )
endif()

add_test(NAME clio_ctp_s3_test COMMAND clio_ctp_s3_test)
