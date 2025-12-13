project(hermes_shm)

set(TEST_MAIN ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(allocator)
add_subdirectory(backend)
add_subdirectory(data_structures)
add_subdirectory(introspect)
# Following test directories disabled - depend on deleted code:
# add_subdirectory(types)
add_subdirectory(thread)
# add_subdirectory(singleton)
# add_subdirectory(solvers)

if(NOT HSHM_COMPILER_MSVC)
    add_subdirectory(interceptor)
endif()

if(HSHM_ENABLE_COMPRESS)
    add_subdirectory(compress)
endif()

if(HSHM_ENABLE_ENCRYPT)
    add_subdirectory(encrypt)
endif()

if(HSHM_ENABLE_CUDA OR HSHM_ENABLE_ROCM)
    add_subdirectory(gpu)
endif()

# Lightbeam tests disabled - depend on deleted code
# if(HSHM_ENABLE_ZMQ)
#     add_subdirectory(lightbeam)
# endif()