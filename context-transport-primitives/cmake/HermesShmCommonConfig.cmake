message("Importing packages for hermes_shm")

# ------------------------------------------------------------------------------
# External libraries
# ------------------------------------------------------------------------------
# Pkg-Config
find_package(PkgConfig REQUIRED)

if(PkgConfig)
    message(STATUS "found pkg config")
endif()

# Doxygen
if(HSHM_ENABLE_DOXYGEN)
    find_package(Perl REQUIRED)
    find_package(Doxygen REQUIRED)
    message(STATUS "found doxygen at ${DOXYGEN_EXECUTABLE}")
endif()

# ZeroMQ directories (only if ZMQ is enabled)
# Note: ZeroMQ variables are set later in the file when HSHM_ENABLE_ZMQ is ON

# Catch2 (only if tests are enabled)
if(HSHM_BUILD_TESTS)
    find_package(Catch2 3.0.1 REQUIRED)
    message(STATUS "found catch2.h at ${Catch2_CXX_INCLUDE_DIRS}")
endif()

# YAML-CPP
# Skip find_package if target already exists (from submodule)
if(NOT TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
    find_package(yaml-cpp REQUIRED)
    message(STATUS "found yaml-cpp at ${yaml-cpp_DIR}")
endif()

if (${YAML_CPP_LIBRARIES})
    set(YAML_CPP_LIBS ${YAML_CPP_LIBRARIES})
elseif(TARGET yaml-cpp::yaml-cpp)
        set(YAML_CPP_LIBS yaml-cpp::yaml-cpp)
else()
  set(YAML_CPP_LIBS yaml-cpp)
endif()


# MPICH
if(HSHM_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS C CXX)
    set(MPI_LIBS MPI::MPI_CXX)
    message(STATUS "found mpi.h at ${MPI_CXX_INCLUDE_DIRS}")
endif()

# ROCm
if(HSHM_ENABLE_ROCM)
    find_package(HIP REQUIRED)
endif()

# OpenMP
if(HSHM_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C CXX)
    set(OpenMP_LIBS OpenMP::OpenMP_CXX)
    message(STATUS "found omp.h at ${OpenMP_CXX_INCLUDE_DIRS}")
endif()

# thallium
if(HSHM_ENABLE_THALLIUM)
    find_package(thallium CONFIG REQUIRED)

    if(thallium_FOUND)
        message(STATUS "found thallium at ${thallium_DIR}")
    endif()

    set(SERIALIZATION_LIBS thallium ${SERIALIZATION_LIBS})
    set(HSHM_ENABLE_CEREAL ON)
endif()

# Cereal
if(HSHM_ENABLE_CEREAL)
    # Skip find_package if target already exists (from submodule)
    if(NOT TARGET cereal AND NOT TARGET cereal::cereal)
        find_package(cereal CONFIG REQUIRED)

        if(cereal_FOUND)
            message(STATUS "found cereal at ${cereal_DIR}")
        endif()
    endif()

    set(SERIALIZATION_LIBS cereal::cereal ${SERIALIZATION_LIBS})
endif()

# Boost
# find_package(Boost REQUIRED)
# message(STATUS "found boost.h at ${Boost_INCLUDE_DIRS}")

# Compression libraries
if(HSHM_ENABLE_COMPRESS)
    message("HSHM_ENABLE_COMPRESS is ON")
    pkg_check_modules(bzip2 REQUIRED bzip2)
    message(STATUS "found bz2.h at ${bzip2_INCLUDE_DIRS}")

    pkg_check_modules(lzo2 REQUIRED lzo2)
    message(STATUS "found lzo2.h at ${lzo2_INCLUDE_DIRS}")
    get_filename_component(lzo2_dir "${lzo2_INCLUDE_DIRS}" DIRECTORY)

    pkg_check_modules(libzstd REQUIRED libzstd)
    message(STATUS "found zstd.h at ${libzstd_INCLUDE_DIRS}")

    pkg_check_modules(liblz4 REQUIRED liblz4)
    message(STATUS "found lz4.h at ${liblz4_INCLUDE_DIRS}")

    pkg_check_modules(zlib REQUIRED zlib)
    message(STATUS "found zlib.h at ${zlib_INCLUDE_DIRS}")

    pkg_check_modules(liblzma REQUIRED liblzma)
    message(STATUS "found liblzma.h at ${liblzma_INCLUDE_DIRS}")

    pkg_check_modules(libbrotlicommon REQUIRED libbrotlicommon libbrotlidec libbrotlienc)
    message(STATUS "found libbrotli.h at ${libbrotlicommon_INCLUDE_DIRS}")

    pkg_check_modules(snappy REQUIRED snappy)
    message(STATUS "found libbrotli.h at ${snappy_INCLUDE_DIRS}")

    pkg_check_modules(blosc2 REQUIRED blosc2)
    message(STATUS "found blosc2.h at ${blosc2_INCLUDE_DIRS}")

    set(COMPRESS_LIBS
        bz2
        ${lzo2_LIBRARIES}
        ${libzstd_LIBRARIES}
        ${liblz4_LIBRARIES}
        ${zlib_LIBRARIES}
        ${liblzma_LIBRARIES}
        ${libbrotlicommon_LIBRARIES}
        ${snappy_LIBRARIES}
        ${blosc2_LIBRARIES}
    )
    set(COMPRESS_INCLUDES
        ${bzip2_INCLUDE_DIRS}
        ${lzo2_INCLUDE_DIRS} ${lzo2_dir}
        ${libzstd_INCLUDE_DIRS}
        ${liblz4_INCLUDE_DIRS}
        ${zlib_INCLUDE_DIRS}
        ${liblzma_INCLUDE_DIRS}
        ${libbrotlicommon_INCLUDE_DIRS}
        ${snappy_INCLUDE_DIRS}
        ${blosc2_INCLUDE_DIRS}
    )
    set(COMPRESS_LIB_DIRS
        ${bzip2_LIBRARY_DIRS}
        ${lzo2_LIBRARY_DIRS}
        ${libzstd_LIBRARY_DIRS}
        ${liblz4_LIBRARY_DIRS}
        ${zlib_LIBRARY_DIRS}
        ${liblzma_LIBRARY_DIRS}
        ${libbrotlicommon_LIBRARY_DIRS}
        ${snappy_LIBRARY_DIRS}
        ${blosc2_LIBRARY_DIRS}
    )
endif()

# Encryption libraries
if(HSHM_ENABLE_ENCRYPT)
    pkg_check_modules(libcrypto REQUIRED libcrypto)
    message(STATUS "found libcrypto.h at ${libcrypto_INCLUDE_DIRS}")

    set(ENCRYPT_LIBS ${libcrypto_LIBRARIES})
    set(ENCRYPT_INCLUDES ${libcrypto_INCLUDE_DIRS})
    set(ENCRYPT_LIB_DIRS ${libcrypto_LIBRARY_DIRS})
endif()

# Add elf
if(HSHM_ENABLE_ELF)
    pkg_check_modules(libelf REQUIRED libelf)
    message(STATUS "found libelf.h at ${libelf_INCLUDE_DIRS}")

    set(ELF_LIBS ${libelf_LIBRARIES})
    set(ELF_INCLUDES ${libelf_INCLUDE_DIRS})
    set(ELF_LIB_DIRS ${libelf_LIBRARY_DIRS})
endif()

# ZeroMQ
if(HSHM_ENABLE_ZMQ)
    pkg_check_modules(ZeroMQ REQUIRED libzmq)
    message(STATUS "found zmq.h at ${ZeroMQ_INCLUDE_DIRS}")
    set(ZMQ_LIBS ${ZeroMQ_LIBRARIES})
    set(ZMQ_INCLUDES ${ZeroMQ_INCLUDE_DIRS})
    set(ZMQ_LIB_DIRS ${ZeroMQ_LIBRARY_DIRS})
endif()

# Libfabric
if(HSHM_ENABLE_LIBFABRIC)
    pkg_check_modules(Libfabric REQUIRED libfabric)
    message(STATUS "found rdma/fabric.h at ${Libfabric_INCLUDE_DIRS}")
    set(LIBFABRIC_LIBS ${Libfabric_LIBRARIES})
    set(LIBFABRIC_INCLUDES ${Libfabric_INCLUDE_DIRS})
    set(LIBFABRIC_LIB_DIRS ${Libfabric_LIBRARY_DIRS})
endif()

# ------------------------------------------------------------------------------
# GPU Support Functions
# ------------------------------------------------------------------------------

# Enable cuda boilerplate
macro(hshm_enable_cuda CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()

    message("USING CUDA ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler -diag-suppress=177,20014,20011,20012")
    enable_language(CUDA)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)
endmacro()

# Enable rocm boilerplate
macro(hshm_enable_rocm GPU_RUNTIME CXX_STANDARD)
    set(GPU_RUNTIME ${GPU_RUNTIME})
    enable_language(${GPU_RUNTIME})
    set(CMAKE_${GPU_RUNTIME}_STANDARD ${CXX_STANDARD})
    set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
    set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler")
    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

    if(GPU_RUNTIME STREQUAL "CUDA")
        include_directories("${ROCM_ROOT}/include")
    endif()

    if(NOT HIP_FOUND)
        find_package(HIP REQUIRED)
    endif()
endmacro()

# Function for setting source files for rocm
function(set_rocm_sources MODE DO_COPY SRC_FILES ROCM_SOURCE_FILES_VAR)
    set(ROCM_SOURCE_FILES ${${ROCM_SOURCE_FILES_VAR}} PARENT_SCOPE)
    set(GPU_RUNTIME ${GPU_RUNTIME} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm_${MODE}/${SOURCE})
            configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        else()
            set(ROCM_SOURCE ${SOURCE})
        endif()

        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${GPU_RUNTIME})
    endforeach()

    set(${ROCM_SOURCE_FILES_VAR} ${ROCM_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for setting source files for cuda
function(set_cuda_sources DO_COPY SRC_FILES CUDA_SOURCE_FILES_VAR)
    set(CUDA_SOURCE_FILES ${${CUDA_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(CUDA_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/cuda/${SOURCE})
            configure_file(${SOURCE} ${CUDA_SOURCE} COPYONLY)
        else()
            set(CUDA_SOURCE ${SOURCE})
        endif()

        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE})
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    endforeach()

    set(${CUDA_SOURCE_FILES_VAR} ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for adding a ROCm library
function(add_rocm_gpu_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(gpu "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm host-only library
function(add_rocm_host_library TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(host "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_host_executable TARGET)
    set(SRC_FILES ${ARGN})
    add_executable(${TARGET} ${SRC_FILES})
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Function for adding a ROCm executable
function(add_rocm_gpu_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(exec "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_executable(${TARGET} ${ROCM_SOURCE_FILES})
    target_link_libraries(${TARGET} PUBLIC amdhip64 amd_comgr)
    target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
    target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
endfunction()

# Function for adding a CUDA library
function(add_cuda_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${CUDA_SOURCE_FILES})

    # target_link_libraries(${TARGET} PUBLIC cudart)
    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    # -fvisibility=hidden -fvisibility-inlines-hidden
    if(SHARED STREQUAL "SHARED")
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Shared
        )
    else()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Static
        )
    endif()
endfunction()

# Function for adding a CUDA executable
function(add_cuda_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_executable(${TARGET} ${CUDA_SOURCE_FILES})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # target_link_libraries(${TARGET} PUBLIC cudart)
    target_compile_options(${TARGET} PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endfunction()

# Function for autoregistering a jarvis repo
macro(jarvis_repo_add REPO_PATH PIPELINE_PATH)
    # Get the file name of the source path
    get_filename_component(REPO_NAME ${REPO_PATH} NAME)

    # Install jarvis repo
    install(DIRECTORY ${REPO_PATH}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis)

    # Add jarvis repo after installation
    # Ensure install commands use env vars from host system, particularly PATH and PYTHONPATH
    install(CODE "execute_process(COMMAND env \"PATH=$ENV{PATH}\" \"PYTHONPATH=$ENV{PYTHONPATH}\" jarvis repo add ${CMAKE_INSTALL_PREFIX}/jarvis/${REPO_NAME})")

    if(REPO_NAME)
        install(DIRECTORY ${PIPELINE_PATH}
            DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis)
    endif()
endmacro()

# DOXYGEN
function(add_doxygen_doc)
    set(options)
    set(oneValueArgs BUILD_DIR DOXY_FILE TARGET_NAME COMMENT)
    set(multiValueArgs)

    cmake_parse_arguments(DOXY_DOC
        "${options}"
        "${oneValueArgs}"
        "${multiValueArgs}"
        ${ARGN}
    )

    configure_file(
        ${DOXY_DOC_DOXY_FILE}
        ${DOXY_DOC_BUILD_DIR}/Doxyfile
        @ONLY
    )

    add_custom_target(${DOXY_DOC_TARGET_NAME}
        COMMAND
        ${DOXYGEN_EXECUTABLE} Doxyfile
        WORKING_DIRECTORY
        ${DOXY_DOC_BUILD_DIR}
        COMMENT
        "Building ${DOXY_DOC_COMMENT} with Doxygen"
        VERBATIM
    )

    message(STATUS "Added ${DOXY_DOC_TARGET_NAME} [Doxygen] target to build documentation")
endfunction()

# FIND PYTHON
macro(find_first_path_python)
    if(DEFINED ENV{PATH})
        string(REPLACE ":" ";" PATH_LIST $ENV{PATH})

        foreach(PATH_ENTRY ${PATH_LIST})
            find_program(PYTHON_SCAN
                NAMES python3 python
                PATHS ${PATH_ENTRY}
                NO_DEFAULT_PATH
            )

            if(PYTHON_SCAN)
                message(STATUS "Found Python in PATH: ${PYTHON_SCAN}")
                set(Python_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                set(Python3_EXECUTABLE ${PYTHON_SCAN} CACHE FILEPATH "Python executable" FORCE)
                break()
            endif()
        endforeach()
    endif()
    
    set(Python_FIND_STRATEGY LOCATION)
    find_package(Python3 COMPONENTS Interpreter Development)

    if(Python3_FOUND)
        message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Python3 not found")
    endif()
endmacro()