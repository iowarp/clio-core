project(hermes_shm)

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

if(NOT HSHM_COMPILER_MSVC)
    add_subdirectory(interceptor)
endif()

if(WRP_CTE_ENABLE_COMPRESS)
    add_subdirectory(compress)
endif()

if(WRP_CORE_ENABLE_ENCRYPT)
    add_subdirectory(encrypt)
endif()

if(WRP_CORE_ENABLE_CUDA OR WRP_CORE_ENABLE_ROCM)
    add_subdirectory(gpu)
endif()

add_subdirectory(lightbeam)