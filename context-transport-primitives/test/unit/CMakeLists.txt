project(clio_ctp)

# Find Catch2 testing framework
find_package(Catch2 3 REQUIRED)

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(allocator)
add_subdirectory(backend)
add_subdirectory(data_structures)
add_subdirectory(introspect)
add_subdirectory(util)
# Following test directories disabled - depend on deleted code:
# add_subdirectory(types)
add_subdirectory(thread)
# add_subdirectory(singleton)
# add_subdirectory(solvers)

if(NOT CLIO_CTP_COMPILER_MSVC)
    add_subdirectory(interceptor)
endif()

if(CLIO_CTE_ENABLE_COMPRESS)
    add_subdirectory(compress)
endif()

if(CLIO_CORE_ENABLE_ENCRYPT)
    add_subdirectory(encrypt)
endif()

add_subdirectory(io)

if(CLIO_CORE_ENABLE_CUDA OR CLIO_CORE_ENABLE_ROCM OR CLIO_CORE_ENABLE_SYCL)
    add_subdirectory(gpu)
endif()

add_subdirectory(lightbeam)

if(CLIO_ENABLE_AMAZON_DRIVE)
    add_subdirectory(amazon_s3)
endif()