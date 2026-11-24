project(clio_ctp)

# Tests that span several modules or exercise external services. Each module's
# own tests live in <module>/test; main.cc, main_mpi.cc and basic_test.h here
# are the shared harness (TEST_MAIN).

if(CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM OR CLIO_CORE_ENABLE_SYCL)
    add_subdirectory(gpu)
endif()

# Gated on CAE_ENABLE_S3, not CLIO_ENABLE_AMAZON_DRIVE: this test drives the AWS
# SDK directly, and CAE_ENABLE_S3 is the option that implies the SDK is present.
# The google_cloud test below is Poco-based, so its bdev flag is the right gate.
if(CAE_ENABLE_S3)
    add_subdirectory(amazon_s3)
endif()

if(CLIO_ENABLE_GOOGLE_CLOUD)
    add_subdirectory(google_cloud)
endif()
