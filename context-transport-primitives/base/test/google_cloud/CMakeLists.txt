# Google Cloud Storage "drive" example. Unlike the AWS SDK (fetched on demand for
# the S3 example), Poco is already located by the root CMakeLists.txt
# (find_package(Poco COMPONENTS Net NetSSL Crypto JSON ...)), and the root guard
# only adds this subdirectory when Poco was found, so the targets below resolve.

add_executable(clio_ctp_gcs_test test_gcs_drive.cc)

target_link_libraries(clio_ctp_gcs_test
    PRIVATE
    Catch2::Catch2WithMain
    Poco::Net
    Poco::NetSSL
    Poco::Crypto
    Poco::JSON
)

add_test(NAME clio_ctp_gcs_test COMMAND clio_ctp_gcs_test)
