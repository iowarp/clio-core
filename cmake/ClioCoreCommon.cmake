# IowarpCoreCommon.cmake - Common CMake functions for IOWarp Core and external repos
#
# This file provides shared utilities for both the IOWarp Core build and external
# repositories that depend on it.

# Guard against multiple inclusions
if(IOWARP_CORE_COMMON_INCLUDED)
  return()
endif()
set(IOWARP_CORE_COMMON_INCLUDED TRUE)

message(STATUS "Loading IowarpCoreCommon.cmake")

#------------------------------------------------------------------------------
# Dependency Target Resolution
#------------------------------------------------------------------------------

# Macro to resolve yaml-cpp target name across different versions
# Older versions use "yaml-cpp", newer versions use "yaml-cpp::yaml-cpp"
macro(resolve_yaml_cpp_target)
  if(NOT DEFINED YAML_CPP_LIBS)
    if(TARGET yaml-cpp::yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp::yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp::yaml-cpp")
    elseif(TARGET yaml-cpp)
      set(YAML_CPP_LIBS yaml-cpp)
      message(STATUS "Using yaml-cpp target: yaml-cpp")
    else()
      message(FATAL_ERROR "yaml-cpp target not found. Expected either 'yaml-cpp::yaml-cpp' or 'yaml-cpp'")
    endif()
  endif()
endmacro()

#------------------------------------------------------------------------------
# GPU Support Functions
#------------------------------------------------------------------------------

# Enable cuda boilerplate
macro(wrp_core_enable_cuda CXX_STANDARD)
    set(CMAKE_CUDA_STANDARD ${CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures to compile for" FORCE)
    endif()

    message(STATUS "USING CUDA ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
    # These GCC/Clang-style warning suppressions are forwarded to the host
    # compiler by nvcc on GCC/Clang hosts, but nvcc rejects them outright with
    # the MSVC host ("nvcc fatal : Unknown option '-Wno-unused-variable'"),
    # which also breaks CMake's CUDA compiler/arch probe on Windows. Only add
    # them when the host compiler isn't MSVC.
    if(NOT MSVC)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unused-variable")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-format -Wno-pedantic -Wno-sign-compare -Wno-unused-but-set-variable")
    endif()
    enable_language(CUDA)

    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

    # When code coverage is enabled, add_link_options(--coverage) causes the
    # CUDA device link stub to reference __gcov_* symbols. Set a flag so
    # add_cuda_library/add_cuda_executable can link libgcov to resolve them.
    if(CLIO_CORE_ENABLE_COVERAGE)
        set(CLIO_CORE_CUDA_NEEDS_GCOV TRUE CACHE INTERNAL "")
    endif()

    # Cache critical CUDA platform variables so they survive any nested
    # project() call (e.g. from external/llama.cpp) that may reset the
    # CMake variable scope.  Without caching, _CMAKE_CUDA_WHOLE_FLAG and
    # CMAKE_CUDA_COMPILE_OBJECT are silently lost and the generate step
    # fails with "Error required internal CMake variable not set."
    foreach(_cuda_var
            CMAKE_INCLUDE_FLAG_CUDA
            _CMAKE_CUDA_WHOLE_FLAG
            _CMAKE_CUDA_RDC_FLAG
            _CMAKE_CUDA_PTX_FLAG
            _CMAKE_CUDA_EXTRA_FLAGS
            _CMAKE_COMPILE_AS_CUDA_FLAG
            CMAKE_CUDA_COMPILE_OBJECT
            CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
            CMAKE_CUDA_LINK_EXECUTABLE
            CMAKE_CUDA_DEVICE_LINK_COMPILE_WHOLE_COMPILATION
            CMAKE_CUDA_COMPILER_HAS_DEVICE_LINK_PHASE
            CMAKE_CUDA_CREATE_SHARED_LIBRARY
            CMAKE_CUDA_CREATE_SHARED_MODULE
            CMAKE_CUDA_DEVICE_LINK_LIBRARY
            CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
            CMAKE_CUDA_DEVICE_LINK_COMPILE
            CMAKE_CUDA_HOST_LINK_LAUNCHER
            CMAKE_SHARED_LIBRARY_CUDA_FLAGS
            CMAKE_SHARED_LIBRARY_CREATE_CUDA_FLAGS)
        if(DEFINED ${_cuda_var})
            set(${_cuda_var} "${${_cuda_var}}" CACHE INTERNAL "" FORCE)
        endif()
    endforeach()
endmacro()

# Enable rocm boilerplate
macro(wrp_core_enable_rocm GPU_RUNTIME CXX_STANDARD)
    # Detect HIP platform once so the add_rocm_* helpers below can route
    # NVIDIA-backed builds (HIP_PLATFORM=nvidia, e.g. dev container with
    # only CUDA hardware) through nvcc instead of clang/HIP-AMD. The env
    # var is what hipconfig and hipcc use; mirror it into a cache var.
    if(NOT DEFINED CLIO_ROCM_HIP_PLATFORM)
        if(DEFINED ENV{HIP_PLATFORM})
            set(CLIO_ROCM_HIP_PLATFORM "$ENV{HIP_PLATFORM}" CACHE STRING
                "HIP platform (amd|nvidia)")
        else()
            set(CLIO_ROCM_HIP_PLATFORM "amd" CACHE STRING
                "HIP platform (amd|nvidia)")
        endif()
    endif()
    message(STATUS "ROCm enabled: HIP_PLATFORM=${CLIO_ROCM_HIP_PLATFORM}")

    set(ROCM_ROOT
        "/opt/rocm"
        CACHE PATH
        "Root directory of the ROCm installation"
    )

    if(CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
        # HIP-NVCC: hipcc internally invokes nvcc. Reuse CMake's CUDA
        # language so the GPU sources are compiled with nvcc + the HIP
        # runtime headers. The add_rocm_* helpers below set
        # `__HIP_PLATFORM_NVIDIA__` and add ${ROCM_ROOT}/include.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
                "CUDA architectures to compile for" FORCE)
        endif()
        enable_language(CUDA)
        # Cache the CUDA standard so it reaches every subdirectory
        # (the bare `set(...)` only lives in the calling scope, which
        # is why test/* targets in deeper add_subdirectory chains
        # weren't seeing the standard or the per-target arch flag).
        set(CMAKE_CUDA_STANDARD ${CXX_STANDARD} CACHE STRING
            "CUDA C++ standard" FORCE)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL
            "Require CUDA standard" FORCE)
        set(GPU_RUNTIME "CUDA")
        # Mirror the cuda-language platform-variable cache from
        # wrp_core_enable_cuda — without this, generate-time fails with
        # "Error required internal CMake variable not set" once a nested
        # project() call resets the CMake variable scope.
        foreach(_cuda_var
                CMAKE_INCLUDE_FLAG_CUDA
                _CMAKE_CUDA_WHOLE_FLAG
                _CMAKE_CUDA_RDC_FLAG
                _CMAKE_CUDA_PTX_FLAG
                _CMAKE_CUDA_EXTRA_FLAGS
                _CMAKE_COMPILE_AS_CUDA_FLAG
                CMAKE_CUDA_COMPILE_OBJECT
                CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
                CMAKE_CUDA_LINK_EXECUTABLE
                CMAKE_CUDA_DEVICE_LINK_COMPILE_WHOLE_COMPILATION
                CMAKE_CUDA_COMPILER_HAS_DEVICE_LINK_PHASE
                CMAKE_CUDA_CREATE_SHARED_LIBRARY
                CMAKE_CUDA_CREATE_SHARED_MODULE
                CMAKE_CUDA_DEVICE_LINK_LIBRARY
                CMAKE_CUDA_DEVICE_LINK_EXECUTABLE
                CMAKE_CUDA_DEVICE_LINK_COMPILE
                CMAKE_CUDA_HOST_LINK_LAUNCHER
                CMAKE_SHARED_LIBRARY_CUDA_FLAGS
                CMAKE_SHARED_LIBRARY_CREATE_CUDA_FLAGS)
            if(DEFINED ${_cuda_var})
                set(${_cuda_var} "${${_cuda_var}}" CACHE INTERNAL "" FORCE)
            endif()
        endforeach()
    else()
        # HIP-AMD: hipcc invokes clang/Clang. Use CMake's HIP language.
        set(GPU_RUNTIME ${GPU_RUNTIME})
        enable_language(${GPU_RUNTIME})
        set(CMAKE_${GPU_RUNTIME}_STANDARD ${CXX_STANDARD})
        set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)
        set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
        if(GPU_RUNTIME STREQUAL "CUDA")
            include_directories("${ROCM_ROOT}/include")
        endif()
        if(NOT HIP_FOUND)
            find_package(HIP REQUIRED)
        endif()
    endif()
endmacro()

# Enable Intel GPU / SYCL (oneAPI icpx -fsycl, or AdaptiveCpp acpp)
#
# Detects the SYCL compiler flavor and sets CLIO_SYCL_COMPILER (DPCPP|ACPP).
# Configures default values for SYCL_TARGET / SYCL_DEVICE / SYCL_CUDA_ARCH
# that callers can override on the CMake command line.
#
# Flags are NOT applied globally to avoid breaking PCH and non-SYCL targets.
# Use add_sycl_library / add_sycl_executable (defined below near
# add_cuda_library / add_cuda_executable), or call wrp_core_apply_sycl_flags()
# directly on a target that contains SYCL device code.
macro(wrp_core_enable_sycl CXX_STANDARD)
    set(CMAKE_CXX_STANDARD ${CXX_STANDARD})
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Compiler flavor detection. Override with -DWRP_SYCL_COMPILER=DPCPP|ACPP.
    if(NOT DEFINED CLIO_SYCL_COMPILER)
        get_filename_component(_wrp_cxx_name "${CMAKE_CXX_COMPILER}" NAME)
        if(_wrp_cxx_name MATCHES "^(acpp|syclcc|hipsycl)" OR
           CMAKE_CXX_COMPILER_ID STREQUAL "AdaptiveCpp")
            set(CLIO_SYCL_COMPILER "ACPP" CACHE STRING
                "SYCL compiler flavor (DPCPP|ACPP)")
        else()
            # Default to DPC++ for icpx, dpcpp, and clang++ with -fsycl.
            set(CLIO_SYCL_COMPILER "DPCPP" CACHE STRING
                "SYCL compiler flavor (DPCPP|ACPP)")
        endif()
    endif()

    # SYCL target triple. Override with -DSYCL_TARGET=...
    #   spir64               JIT, runs on any OpenCL/Level Zero device
    #   spir64_gen           AOT, requires SYCL_DEVICE (e.g. pvc, dg2, gen12lp)
    #   nvptx64-nvidia-cuda  NVIDIA GPUs via DPC++ CUDA backend
    #   amdgcn-amd-amdhsa    AMD GPUs via DPC++ ROCm backend
    if(NOT DEFINED CACHE{SYCL_TARGET})
        set(SYCL_TARGET "spir64" CACHE STRING
            "SYCL target triple (spir64|spir64_gen|nvptx64-nvidia-cuda|amdgcn-amd-amdhsa)")
    endif()
    if(NOT DEFINED CACHE{SYCL_DEVICE})
        set(SYCL_DEVICE "" CACHE STRING
            "SYCL AOT device for spir64_gen target (e.g. pvc, dg2, gen12lp)")
    endif()
    if(NOT DEFINED CACHE{SYCL_CUDA_ARCH})
        set(SYCL_CUDA_ARCH "sm_70" CACHE STRING
            "CUDA architecture when SYCL_TARGET=nvptx64-nvidia-cuda")
    endif()

    # Opt-in flag for SYCL device-side virtual functions.
    # -fsycl-allow-virtual-functions is supported by recent DPC++ nightlies;
    # leave OFF unless you have confirmed your toolchain accepts it.
    option(CLIO_SYCL_ALLOW_VIRTUAL_FUNCTIONS
        "Pass -fsycl-allow-virtual-functions to DPC++ (recent compiler only)"
        OFF)

    message(STATUS "SYCL enabled: compiler=${CLIO_SYCL_COMPILER} target=${SYCL_TARGET} device=${SYCL_DEVICE}")
endmacro()

# Apply SYCL compile and link flags to a specific target.
# Used by add_sycl_library / add_sycl_executable; callers may also invoke
# directly when integrating with externally-defined targets.
#
# Flags applied:
#   DPCPP  -fsycl, -fsycl-targets=<target>, -fsycl-allow-func-ptr
#          -fsycl-unnamed-lambda, plus AOT/CUDA backend args when applicable.
#          -fsycl-allow-virtual-functions if CLIO_SYCL_ALLOW_VIRTUAL_FUNCTIONS=ON.
#   ACPP   --acpp-targets=<target>.
#
# -fsycl-allow-func-ptr is required by chi::gpu::Container's function-pointer
# dispatch table; without it DPC++ rejects taking the address of device
# functions.
function(wrp_core_apply_sycl_flags target)
    if(NOT DEFINED CLIO_SYCL_COMPILER)
        message(FATAL_ERROR
            "CLIO_SYCL_COMPILER is not set; call wrp_core_enable_sycl(<CXX_STANDARD>) "
            "before defining SYCL targets")
    endif()

    target_compile_definitions(${target} PRIVATE CTP_ENABLE_SYCL=1)

    if(CLIO_SYCL_COMPILER STREQUAL "DPCPP")
        # -fsycl-allow-func-ptr is a Clang -cc1 flag in current DPC++
        # nightlies, so pass it through with -Xclang. Ditto for the
        # virtual-functions opt-in.
        target_compile_options(${target} PRIVATE
            -fsycl
            -fsycl-targets=${SYCL_TARGET}
            -fsycl-unnamed-lambda
            -Xclang -fsycl-allow-func-ptr
        )
        target_link_options(${target} PRIVATE
            -fsycl
            -fsycl-targets=${SYCL_TARGET}
        )

        # AOT device flags only meaningful for spir64_gen target
        if(SYCL_TARGET STREQUAL "spir64_gen" AND SYCL_DEVICE)
            target_compile_options(${target} PRIVATE
                "SHELL:-Xsycl-target-backend \"-device ${SYCL_DEVICE}\""
            )
            target_link_options(${target} PRIVATE
                "SHELL:-Xsycl-target-backend \"-device ${SYCL_DEVICE}\""
            )
        endif()

        # NVIDIA backend (DPC++ with CUDA support, e.g. clang-sycl-cuda nightlies)
        if(SYCL_TARGET MATCHES "nvptx64")
            target_compile_options(${target} PRIVATE
                "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${SYCL_CUDA_ARCH}"
            )
            target_link_options(${target} PRIVATE
                "SHELL:-Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=${SYCL_CUDA_ARCH}"
            )
        endif()

        if(CLIO_SYCL_ALLOW_VIRTUAL_FUNCTIONS)
            target_compile_options(${target} PRIVATE
                -Xclang -fsycl-allow-virtual-functions)
            target_link_options(${target} PRIVATE
                -Xclang -fsycl-allow-virtual-functions)
        endif()
    elseif(CLIO_SYCL_COMPILER STREQUAL "ACPP")
        target_compile_options(${target} PRIVATE
            --acpp-targets=${SYCL_TARGET}
        )
        target_link_options(${target} PRIVATE
            --acpp-targets=${SYCL_TARGET}
        )
    else()
        message(FATAL_ERROR "Unknown CLIO_SYCL_COMPILER='${CLIO_SYCL_COMPILER}' (expected DPCPP or ACPP)")
    endif()
endfunction()

# Function for setting source files for rocm
function(set_rocm_sources MODE DO_COPY SRC_FILES ROCM_SOURCE_FILES_VAR)
    set(ROCM_SOURCE_FILES ${${ROCM_SOURCE_FILES_VAR}} PARENT_SCOPE)
    set(GPU_RUNTIME ${GPU_RUNTIME} PARENT_SCOPE)

    # Pick the language CMake should use for these sources. Under
    # HIP-NVCC mode (HIP_PLATFORM=nvidia) we route through nvcc via
    # CMake's CUDA language; otherwise stick with the requested HIP
    # runtime (typically HIP for AMD).
    if(CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
        set(_wrp_rocm_lang CUDA)
    else()
        set(_wrp_rocm_lang ${GPU_RUNTIME})
    endif()

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(ROCM_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/rocm_${MODE}/${SOURCE})
            configure_file(${SOURCE} ${ROCM_SOURCE} COPYONLY)
        else()
            set(ROCM_SOURCE ${SOURCE})
        endif()

        list(APPEND ROCM_SOURCE_FILES ${ROCM_SOURCE})
        set_source_files_properties(${ROCM_SOURCE} PROPERTIES LANGUAGE ${_wrp_rocm_lang})
    endforeach()

    set(${ROCM_SOURCE_FILES_VAR} ${ROCM_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Function for setting source files for cuda
function(set_cuda_sources DO_COPY SRC_FILES CUDA_SOURCE_FILES_VAR)
    set(CUDA_SOURCE_FILES ${${CUDA_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            # Mirror the source under a private cuda/ dir so the same .cc can be
            # compiled as both CXX (CPU lib) and CUDA (GPU lib) without object
            # collisions. GLOB_RECURSE yields ABSOLUTE paths; embedding an
            # absolute Windows path (with its drive colon) under cuda/ makes an
            # invalid destination ("Fail to copy: Invalid argument"). Use a
            # path relative to the project root instead (also tidier on Linux,
            # which otherwise nested the full /abs/path under cuda/).
            # SRC_FILES may be absolute (file(GLOB_RECURSE)) or relative
            # (explicit names); normalize to absolute, then derive a dest path
            # relative to the project root.
            get_filename_component(_cuda_abs "${SOURCE}" ABSOLUTE)
            file(RELATIVE_PATH _cuda_rel "${CMAKE_SOURCE_DIR}" "${_cuda_abs}")
            set(CUDA_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/cuda/${_cuda_rel}")
            configure_file(${_cuda_abs} ${CUDA_SOURCE} COPYONLY)
        else()
            set(CUDA_SOURCE ${SOURCE})
        endif()

        list(APPEND CUDA_SOURCE_FILES ${CUDA_SOURCE})
        set_source_files_properties(${CUDA_SOURCE} PROPERTIES LANGUAGE CUDA)
    endforeach()

    set(${CUDA_SOURCE_FILES_VAR} ${CUDA_SOURCE_FILES} PARENT_SCOPE)
endfunction()

# Apply HIP / ROCm flags appropriate for the current HIP_PLATFORM. On
# AMD this means -fgpu-rdc (Clang relocatable device code) plus the
# AMD HIP runtime libraries; on NVIDIA we route through nvcc (CUDA
# language) and link cudart instead, with the ROCm include directory
# in the path so `hip/hip_runtime.h` resolves to the HIP-NVCC headers.
function(_wrp_apply_rocm_flags TARGET)
    if(CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
        target_compile_definitions(${TARGET} PUBLIC
            __HIP_PLATFORM_NVIDIA__=1
            __HIP_PLATFORM_NVCC__=1)
        target_include_directories(${TARGET} PUBLIC ${ROCM_ROOT}/include)
        find_package(CUDAToolkit QUIET)
        if(TARGET CUDA::cudart)
            target_link_libraries(${TARGET} PUBLIC CUDA::cudart)
        endif()
        # Resolve "native" to the real CUDA arch like add_cuda_library does;
        # CMake otherwise leaves CUDA_ARCHITECTURES literally "native" on
        # subdirectory targets, which produces a cubin that the active
        # device's compute capability rejects ("invalid device function").
        if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native"
           AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
            set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
                CACHE STRING "CUDA architectures to compile for" FORCE)
        endif()
        set_target_properties(${TARGET} PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
            CUDA_RUNTIME_LIBRARY Shared)
    else()
        target_link_libraries(${TARGET} PUBLIC -fgpu-rdc)
        target_compile_options(${TARGET} PUBLIC -fgpu-rdc)
        set_target_properties(${TARGET} PROPERTIES
            POSITION_INDEPENDENT_CODE ON)
    endif()
endfunction()

# Function for adding a ROCm library
function(add_rocm_gpu_library TARGET SHARED DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(gpu "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${SHARED} ${ROCM_SOURCE_FILES})
    _wrp_apply_rocm_flags(${TARGET})
endfunction()

# Function for adding a ROCm host-only library
function(add_rocm_host_library TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(host "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_library(${TARGET} ${ROCM_SOURCE_FILES})
    _wrp_apply_rocm_flags(${TARGET})
endfunction()

# Function for adding a ROCm host-only executable
function(add_rocm_host_executable TARGET)
    set(SRC_FILES ${ARGN})
    add_executable(${TARGET} ${SRC_FILES})
    _wrp_apply_rocm_flags(${TARGET})
endfunction()

# Function for adding a ROCm executable
function(add_rocm_gpu_executable TARGET DO_COPY)
    set(SRC_FILES ${ARGN})
    set(ROCM_SOURCE_FILES "")
    set_rocm_sources(exec "${DO_COPY}" "${SRC_FILES}" ROCM_SOURCE_FILES)
    add_executable(${TARGET} ${ROCM_SOURCE_FILES})
    if(CLIO_ROCM_HIP_PLATFORM STREQUAL "nvidia")
        # On NVIDIA hardware no AMD runtime libs to link.
    else()
        target_link_libraries(${TARGET} PUBLIC amdhip64 amd_comgr)
    endif()
    _wrp_apply_rocm_flags(${TARGET})
endfunction()


# Function for adding a CUDA library (NVCC only)
#
# Uses CMake's native CUDA language support with NVCC.
#
# Usage:
#   add_cuda_library(TARGET SHARED|STATIC DO_COPY source1.cc ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...])
function(add_cuda_library TARGET SHARED DO_COPY)
    cmake_parse_arguments(CUDA "" "" "INCLUDE_DIRS;LINK_LIBS" ${ARGN})
    set(SRC_FILES ${CUDA_UNPARSED_ARGUMENTS})

    # Resolve "native" to the detected GPU architecture before add_library so
    # CMake does not attempt to re-detect the GPU at configure time for targets
    # created in subdirectories processed after the first enable_language(CUDA)
    # call.
    if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native" AND CMAKE_CUDA_ARCHITECTURES_NATIVE)
        set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES_NATIVE}"
            CACHE STRING "CUDA architectures to compile for" FORCE)
    endif()

    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)

    add_library(${TARGET} ${SHARED} ${CUDA_SOURCE_FILES})

    set_target_properties(${TARGET} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    if(SHARED STREQUAL "SHARED")
        # NOTE: do NOT set WINDOWS_EXPORT_ALL_SYMBOLS here. For nvcc
        # separable-compilation libs it does not reliably export symbols, and
        # it conflicts with explicit __declspec(dllexport): CMake's generated
        # .def omits dllexport'd symbols, and the /DEF: link then drops them.
        # Cross-DLL symbols from GPU libs are exported explicitly instead (e.g.
        # CLIO_RUN_GPU_API on the ChiServerBootstrap* entry points, CTP_DLL in
        # clio_ctp_cuda).
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

    if(CUDA_INCLUDE_DIRS)
        foreach(_dir IN LISTS CUDA_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC
                $<BUILD_INTERFACE:${_dir}>)
        endforeach()
    endif()

    # Resolve __gcov_* symbols from CUDA device link stubs when coverage is on
    if(CLIO_CORE_CUDA_NEEDS_GCOV)
        set_property(TARGET ${TARGET} APPEND PROPERTY LINK_LIBRARIES gcov)
    endif()

    if(CUDA_LINK_LIBS)
        target_link_libraries(${TARGET} ${CUDA_LINK_LIBS})
    endif()
endfunction()

# Function for adding a CUDA executable (NVCC only)
#
# Uses CMake's native CUDA language support with NVCC.
#
# Usage:
#   add_cuda_executable(TARGET DO_COPY source1.cc ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...]
#       [DEFS DEF1 DEF2 ...])
function(add_cuda_executable TARGET DO_COPY)
    cmake_parse_arguments(CUDA "" "" "INCLUDE_DIRS;LINK_LIBS;DEFS" ${ARGN})
    set(SRC_FILES ${CUDA_UNPARSED_ARGUMENTS})

    set(CUDA_SOURCE_FILES "")
    set_cuda_sources("${DO_COPY}" "${SRC_FILES}" CUDA_SOURCE_FILES)
    add_executable(${TARGET} ${CUDA_SOURCE_FILES})
    set_target_properties(${TARGET} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    if(${DO_COPY})
        target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(CUDA_INCLUDE_DIRS)
        foreach(_dir IN LISTS CUDA_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC
                $<BUILD_INTERFACE:${_dir}>)
        endforeach()
    endif()

    if(CUDA_DEFS)
        target_compile_definitions(${TARGET} PRIVATE ${CUDA_DEFS})
    endif()

    # Resolve __gcov_* symbols from CUDA device link stubs when coverage is on
    if(CLIO_CORE_CUDA_NEEDS_GCOV)
        set_property(TARGET ${TARGET} APPEND PROPERTY LINK_LIBRARIES gcov)
    endif()

    if(CUDA_LINK_LIBS)
        target_link_libraries(${TARGET} ${CUDA_LINK_LIBS})
    endif()
endfunction()


# Function for setting SYCL source files.
#
# SYCL is a single-source C++ model: device code lives in regular .cc/.cpp
# files compiled by the configured CXX compiler (icpx/clang++ for DPC++,
# acpp for AdaptiveCpp). DO_COPY mirrors set_cuda_sources / set_rocm_sources
# behavior — when TRUE, sources are configure_file()'d into a per-target
# binary directory.
function(set_sycl_sources DO_COPY SRC_FILES SYCL_SOURCE_FILES_VAR)
    set(SYCL_SOURCE_FILES ${${SYCL_SOURCE_FILES_VAR}} PARENT_SCOPE)

    foreach(SOURCE IN LISTS SRC_FILES)
        if(${DO_COPY})
            set(SYCL_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/sycl/${SOURCE})
            configure_file(${SOURCE} ${SYCL_SOURCE} COPYONLY)
        else()
            set(SYCL_SOURCE ${SOURCE})
        endif()

        list(APPEND SYCL_SOURCE_FILES ${SYCL_SOURCE})
    endforeach()

    set(${SYCL_SOURCE_FILES_VAR} ${SYCL_SOURCE_FILES} PARENT_SCOPE)
endfunction()


# Function for adding a SYCL library (DPC++ icpx, or AdaptiveCpp acpp).
#
# Compiles via the configured CXX compiler with SYCL flags applied through
# wrp_core_apply_sycl_flags(). wrp_core_enable_sycl(<CXX_STANDARD>) must be
# called before use so CLIO_SYCL_COMPILER and SYCL_TARGET are set.
#
# Usage:
#   add_sycl_library(TARGET SHARED|STATIC DO_COPY source1.cc ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...])
function(add_sycl_library TARGET SHARED DO_COPY)
    cmake_parse_arguments(SYCL "" "" "INCLUDE_DIRS;LINK_LIBS" ${ARGN})
    set(SRC_FILES ${SYCL_UNPARSED_ARGUMENTS})

    set(SYCL_SOURCE_FILES "")
    set_sycl_sources("${DO_COPY}" "${SRC_FILES}" SYCL_SOURCE_FILES)

    add_library(${TARGET} ${SHARED} ${SYCL_SOURCE_FILES})

    set_target_properties(${TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON)

    wrp_core_apply_sycl_flags(${TARGET})

    if(SYCL_INCLUDE_DIRS)
        foreach(_dir IN LISTS SYCL_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC
                $<BUILD_INTERFACE:${_dir}>)
        endforeach()
    endif()

    if(SYCL_LINK_LIBS)
        target_link_libraries(${TARGET} ${SYCL_LINK_LIBS})
    endif()
endfunction()


# Function for adding a SYCL executable (DPC++ icpx, or AdaptiveCpp acpp).
#
# Usage:
#   add_sycl_executable(TARGET DO_COPY source1.cc ...
#       [INCLUDE_DIRS dir1 dir2 ...]
#       [LINK_LIBS lib1 lib2 ...]
#       [DEFS DEF1 DEF2 ...])
function(add_sycl_executable TARGET DO_COPY)
    cmake_parse_arguments(SYCL "" "" "INCLUDE_DIRS;LINK_LIBS;DEFS" ${ARGN})
    set(SRC_FILES ${SYCL_UNPARSED_ARGUMENTS})

    set(SYCL_SOURCE_FILES "")
    set_sycl_sources("${DO_COPY}" "${SRC_FILES}" SYCL_SOURCE_FILES)

    add_executable(${TARGET} ${SYCL_SOURCE_FILES})

    set_target_properties(${TARGET} PROPERTIES
        POSITION_INDEPENDENT_CODE ON)

    wrp_core_apply_sycl_flags(${TARGET})

    if(${DO_COPY})
        target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()

    if(SYCL_INCLUDE_DIRS)
        foreach(_dir IN LISTS SYCL_INCLUDE_DIRS)
            target_include_directories(${TARGET} PUBLIC
                $<BUILD_INTERFACE:${_dir}>)
        endforeach()
    endif()

    if(SYCL_DEFS)
        target_compile_definitions(${TARGET} PRIVATE ${SYCL_DEFS})
    endif()

    if(SYCL_LINK_LIBS)
        target_link_libraries(${TARGET} ${SYCL_LINK_LIBS})
    endif()
endfunction()


#------------------------------------------------------------------------------
# Jarvis Repo Management
#------------------------------------------------------------------------------

# Function for autoregistering a jarvis repo
macro(jarvis_repo_add REPO_PATH)
    # Get the file name of the source path
    get_filename_component(REPO_NAME ${REPO_PATH} NAME)

    # Install jarvis repo
    install(DIRECTORY ${REPO_PATH}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/jarvis)

    # Add jarvis repo after installation
    # Ensure install commands use env vars from host system, particularly PATH and PYTHONPATH
    install(CODE "execute_process(COMMAND env \"PATH=$ENV{PATH}\" \"PYTHONPATH=$ENV{PYTHONPATH}\" jarvis repo add ${CMAKE_INSTALL_PREFIX}/jarvis/${REPO_NAME})")
endmacro()

#------------------------------------------------------------------------------
# Doxygen Documentation
#------------------------------------------------------------------------------

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

#------------------------------------------------------------------------------
# Python Finding Utilities
#------------------------------------------------------------------------------

# FIND PYTHON
macro(find_first_path_python)
    # If scikit-build-core or the caller has already set Python3_EXECUTABLE
    # (e.g. pointing at the target interpreter with dev headers), respect it
    # and skip the PATH scan so we don't accidentally pick up a build-env
    # interpreter that lacks development headers.
    if(NOT Python3_EXECUTABLE AND DEFINED ENV{PATH})
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
    find_package(Python3 COMPONENTS Interpreter Development.Module)

    if(Python3_FOUND)
        message(STATUS "Found Python3: ${Python3_EXECUTABLE}")
    else()
        message(FATAL_ERROR "Python3 not found")
    endif()
endmacro()

#------------------------------------------------------------------------------
# ChiMod Helper Functions
#------------------------------------------------------------------------------

# Helper function to link runtime to client library (called via DEFER)
# This allows linking to work regardless of which target is defined first
function(_clio_run_link_runtime_to_client RUNTIME_TARGET CLIENT_TARGET)
  if(TARGET ${CLIENT_TARGET})
    target_link_libraries(${RUNTIME_TARGET} PUBLIC ${CLIENT_TARGET})
    message(STATUS "Deferred linking: Runtime ${RUNTIME_TARGET} linked to client ${CLIENT_TARGET}")
  endif()
endfunction()

# Function to read repository namespace from clio_repo.yaml (preferred) or
# chimaera_repo.yaml (legacy). Both formats are accepted; the new clio_repo.yaml
# takes precedence so projects can migrate gradually. Searches up the directory
# tree from the given path.
function(read_repo_namespace output_var start_path)
  set(current_path "${start_path}")
  set(namespace "chimaera")  # Default fallback

  # Search up the directory tree; at each level look for the new name first,
  # then the legacy name. Either is sufficient.
  while(NOT "${current_path}" STREQUAL "/" AND NOT "${current_path}" STREQUAL "")
    set(_repo_candidates
        "${current_path}/clio_repo.yaml"      # preferred (new)
        "${current_path}/chimaera_repo.yaml") # legacy, still supported
    foreach(repo_file ${_repo_candidates})
      if(EXISTS "${repo_file}")
        file(READ "${repo_file}" REPO_YAML_CONTENT)
        # Anchor to line-start so we only match the real top-level YAML
        # entry, not occurrences of the substring "namespace:" inside a
        # comment. file(READ) returns the whole file, and CMake regex
        # has no multiline flag, so we prepend a newline and require
        # one before the keyword.
        string(REGEX MATCH "\n[ \t]*namespace: *([^\n\r]+)"
               NAMESPACE_MATCH "\n${REPO_YAML_CONTENT}")
        if(NAMESPACE_MATCH)
          string(REGEX REPLACE "\n[ \t]*namespace: *" ""
                 namespace "${NAMESPACE_MATCH}")
          string(STRIP "${namespace}" namespace)
          set(${output_var} "${namespace}" PARENT_SCOPE)
          return()
        endif()
      endif()
    endforeach()

    # Move up one directory
    get_filename_component(current_path "${current_path}" DIRECTORY)
  endwhile()

  set(${output_var} "${namespace}" PARENT_SCOPE)
endfunction()

# Function to read module configuration from clio_mod.yaml (preferred) or
# chimaera_mod.yaml (legacy). Both formats are accepted; the new name takes
# precedence to enable gradual migration.
function(clio_run_read_module_config MODULE_DIR)
  set(_mod_candidates
      "${MODULE_DIR}/clio_mod.yaml"
      "${MODULE_DIR}/chimaera_mod.yaml")
  set(CONFIG_FILE "")
  foreach(_cand ${_mod_candidates})
    if(EXISTS "${_cand}")
      set(CONFIG_FILE "${_cand}")
      break()
    endif()
  endforeach()

  if(NOT CONFIG_FILE)
    message(FATAL_ERROR
            "Missing clio_mod.yaml (or legacy chimaera_mod.yaml) in ${MODULE_DIR}")
  endif()

  # Parse YAML file (simple regex parsing for key: value pairs)
  file(READ ${CONFIG_FILE} CONFIG_CONTENT)

  # Extract module_name
  string(REGEX MATCH "module_name:[ ]*([^\n\r]*)" MODULE_MATCH ${CONFIG_CONTENT})
  if(MODULE_MATCH)
    string(REGEX REPLACE "module_name:[ ]*" "" CLIO_RUN_MODULE_NAME "${MODULE_MATCH}")
    string(STRIP "${CLIO_RUN_MODULE_NAME}" CLIO_RUN_MODULE_NAME)
  endif()
  set(CLIO_RUN_MODULE_NAME ${CLIO_RUN_MODULE_NAME} PARENT_SCOPE)

  # Extract namespace
  string(REGEX MATCH "namespace:[ ]*([^\n\r]*)" NAMESPACE_MATCH ${CONFIG_CONTENT})
  if(NAMESPACE_MATCH)
    string(REGEX REPLACE "namespace:[ ]*" "" CLIO_RUN_NAMESPACE "${NAMESPACE_MATCH}")
    string(STRIP "${CLIO_RUN_NAMESPACE}" CLIO_RUN_NAMESPACE)
  endif()
  set(CLIO_RUN_NAMESPACE ${CLIO_RUN_NAMESPACE} PARENT_SCOPE)

  # Derive a filesystem-safe form of the namespace. The CMake target prefix
  # (e.g. `clio::run::admin_client`) uses `::` separators, but the same
  # value gets reused for library output names, install destinations and
  # export file names where `::` is illegal or filesystem-unfriendly. Replace
  # `::` with `_` so `clio::run` -> `clio_run`, `clio::cte` -> `clio_cte`,
  # legacy `chimaera` -> `chimaera`. Downstream code that needs the path
  # form references CLIO_RUN_PACKAGE_NAME; code that needs the C++/target
  # prefix references CLIO_RUN_NAMESPACE.
  string(REPLACE "::" "_" CLIO_RUN_PACKAGE_NAME "${CLIO_RUN_NAMESPACE}")
  set(CLIO_RUN_PACKAGE_NAME ${CLIO_RUN_PACKAGE_NAME} PARENT_SCOPE)

  # Validate extracted values
  if(NOT CLIO_RUN_MODULE_NAME)
    message(FATAL_ERROR "module_name not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
  endif()

  if(NOT CLIO_RUN_NAMESPACE)
    message(FATAL_ERROR "namespace not found in ${CONFIG_FILE}. Content preview: ${CONFIG_CONTENT}")
  endif()
endfunction()

#------------------------------------------------------------------------------
# ChiMod Client Library Function
#------------------------------------------------------------------------------

# add_clio_module_client - Create a ChiMod client library
#
# Parameters:
#   SOURCES             - Source files for the client library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., clio_cte, clio_cae), this function automatically
#   links chimaera admin and bdev client libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
#
function(add_clio_module_client)
  cmake_parse_arguments(
    ARG
    ""
    "LIB_NAME"
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )

  # Read module configuration
  clio_run_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})

  # Create target name. The optional LIB_NAME argument lets a module
  # override the auto-derived `<package>_<module>` prefix (used e.g.
  # by the bdev module to install as `clio_bdev_*` while the C++
  # namespace stays `clio::run::bdev`).  PACKAGE_NAME (not NAMESPACE)
  # is used here because the target identifier ends up as both a CMake
  # target string AND the library output file name (libfoo.so), and
  # `::` is illegal in filesystem names.
  if(ARG_LIB_NAME)
    set(TARGET_NAME "${ARG_LIB_NAME}_client")
  else()
    set(TARGET_NAME "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}_client")
  endif()

  # Create the library
  add_library(${TARGET_NAME} SHARED ${ARG_SOURCES})

  # Set C++ standard
  set(CLIO_RUN_CXX_STANDARD 20)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CLIO_RUN_CXX_STANDARD})

  # Common compile definitions
  set(CLIO_RUN_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      ${CLIO_RUN_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()

  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()

  # Link libraries - use chimaera::cxx for internal builds, ctp::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET ctp::cxx)
    set(CORE_LIB ctp::cxx)
  elseif(TARGET ClioCtp::cxx)
    set(CORE_LIB ClioCtp::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, ctp::cxx, ClioCtp::cxx nor cxx target found")
  endif()

  # Automatically add foundational ChiMod dependencies in unified builds.
  # Skip when the target being built IS one of those foundational modules
  # (admin keeps LIB_NAME=chimaera_admin, bdev uses LIB_NAME=clio_bdev) — the
  # old guard `NOT NAMESPACE STREQUAL "chimaera"` no longer works now that
  # admin's own namespace is `clio_run`.
  set(CLIO_RUN_MODULE_DEPS "")
  if(NOT "${CLIO_RUN_MODULE_NAME}" STREQUAL "admin" AND
     NOT "${CLIO_RUN_MODULE_NAME}" STREQUAL "bdev")
    if(TARGET clio_admin_client)
      list(APPEND CLIO_RUN_MODULE_DEPS clio_admin_client)
    endif()
    if(TARGET clio_bdev_client)
      list(APPEND CLIO_RUN_MODULE_DEPS clio_bdev_client)
    endif()
  endif()

  # Clients only link to ctp::cxx (no Boost)
  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${CORE_LIB}
      ${ARG_LINK_LIBRARIES}
      ${CLIO_RUN_MODULE_DEPS}
  )

  # Create alias for external use
  add_library(${CLIO_RUN_NAMESPACE}::${CLIO_RUN_MODULE_NAME}_client ALIAS ${TARGET_NAME})
  # Also expose the legacy `chimaera::` alias so consumers that pre-date
  # the module-namespace rename (e.g. coeus-adapter, in-tree CMakeLists
  # that link to chimaera::admin_client) keep working at build time.
  if(NOT TARGET chimaera::${CLIO_RUN_MODULE_NAME}_client)
    add_library(chimaera::${CLIO_RUN_MODULE_NAME}_client ALIAS ${TARGET_NAME})
  endif()

  # Set properties for installation. OUTPUT_NAME tracks LIB_NAME when given
  # so the .so file on disk matches the CMake target.  PACKAGE_NAME (not
  # NAMESPACE) is used here for the same filesystem-safety reason as
  # TARGET_NAME above.
  if(ARG_LIB_NAME)
    set(_chimod_output_name "${ARG_LIB_NAME}_client")
  else()
    set(_chimod_output_name "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}_client")
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CLIO_RUN_MODULE_NAME}_client"
    OUTPUT_NAME "${_chimod_output_name}"
    # Auto-export non-static symbols on Windows so dependent module DLLs
    # (e.g. *_runtime depends on *_client) can find an import lib without
    # each symbol needing __declspec(dllexport).
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  # Install the client library
  # MODULE_PACKAGE_NAME: install dir under lib/cmake/. For runtime modules
  # whose namespace was renamed chimaera -> clio::run (admin, bdev, MOD_NAME)
  # we pin to the legacy `chimaera_<module>` form so external find_package
  # consumers (e.g. coeus-adapter) keep working. For other namespaces
  # (clio::cte, clio::cae, …) we use the standard `<package>_<module>` form
  # (always filesystem-safe).
  if("${CLIO_RUN_NAMESPACE}" STREQUAL "clio::run")
    set(MODULE_PACKAGE_NAME "chimaera_${CLIO_RUN_MODULE_NAME}")
  else()
    set(MODULE_PACKAGE_NAME "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}")
  endif()
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")

  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  # Install headers
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include")
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()

  # Precompiled headers for faster builds
  if(CLIO_CORE_ENABLE_PCH)
    target_precompile_headers(${TARGET_NAME} PRIVATE
        <string> <vector> <memory> <unordered_map>
        <functional> <algorithm> <cstdint> <cstring> <iostream>
    )
  endif()

  # Export module info to parent scope
  set(CLIO_RUN_MODULE_CLIENT_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CLIO_RUN_MODULE_NAME ${CLIO_RUN_MODULE_NAME} PARENT_SCOPE)
  set(CLIO_RUN_NAMESPACE ${CLIO_RUN_NAMESPACE} PARENT_SCOPE)
  set(CLIO_RUN_PACKAGE_NAME ${CLIO_RUN_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

#------------------------------------------------------------------------------
# GPU Device Code Embedding Function
#------------------------------------------------------------------------------
# ChiMod Runtime Library Function
#------------------------------------------------------------------------------

# add_clio_module_runtime - Create a ChiMod runtime library
#
# Parameters:
#   SOURCES             - Source files for the runtime library
#   COMPILE_DEFINITIONS - Additional compile definitions
#   LINK_LIBRARIES      - Additional libraries to link
#   LINK_DIRECTORIES    - Additional link directories
#   INCLUDE_LIBRARIES   - Libraries whose includes should be added
#   INCLUDE_DIRECTORIES - Additional include directories
#
# Automatic Cross-Namespace Dependencies (Unified Builds):
#   For non-chimaera namespaces (e.g., clio_cte, clio_cae), this function automatically
#   links chimaera admin and bdev runtime libraries if they are available as targets.
#   This enables wrp_* ChiMods to use chimaera ChiMod headers and functionality without
#   explicit dependency declarations in their CMakeLists.txt files.
#
function(add_clio_module_runtime)
  cmake_parse_arguments(
    ARG
    ""
    "LIB_NAME"
    "SOURCES;COMPILE_DEFINITIONS;LINK_LIBRARIES;LINK_DIRECTORIES;INCLUDE_LIBRARIES;INCLUDE_DIRECTORIES"
    ${ARGN}
  )

  # Read module configuration
  clio_run_read_module_config(${CMAKE_CURRENT_SOURCE_DIR})

  # Create target name (see add_clio_module_client for LIB_NAME rationale).
  if(ARG_LIB_NAME)
    set(TARGET_NAME "${ARG_LIB_NAME}_runtime")
  else()
    set(TARGET_NAME "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}_runtime")
  endif()

  # The GPU companion library concept (separate _gpu.cc files compiled
  # into a *_runtime_gpu shared library) was removed along with the GPU
  # runtime. ChiMods now only have CPU runtime handlers; kernels submit
  # tasks to the CPU via the gpu2cpu_queue.
  add_library(${TARGET_NAME} SHARED ${ARG_SOURCES})

  # Set C++ standard
  set(CLIO_RUN_CXX_STANDARD 20)
  target_compile_features(${TARGET_NAME} PUBLIC cxx_std_${CLIO_RUN_CXX_STANDARD})

  # Common compile definitions
  set(CLIO_RUN_COMMON_COMPILE_DEFS
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
  )

  # Add compile definitions (runtime always has CHIMAERA_RUNTIME=1)
  target_compile_definitions(${TARGET_NAME}
    PUBLIC
      CHIMAERA_RUNTIME=1
      ${CLIO_RUN_COMMON_COMPILE_DEFS}
      ${ARG_COMPILE_DEFINITIONS}
  )

  # Add include directories with proper BUILD_INTERFACE and INSTALL_INTERFACE
  target_include_directories(${TARGET_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
  )

  # Add additional include directories with BUILD_INTERFACE wrapper
  foreach(INCLUDE_DIR ${ARG_INCLUDE_DIRECTORIES})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${INCLUDE_DIR}>
    )
  endforeach()

  # Add link directories
  if(ARG_LINK_DIRECTORIES)
    target_link_directories(${TARGET_NAME} PUBLIC ${ARG_LINK_DIRECTORIES})
  endif()

  # Link libraries - use ctp::cxx for internal builds, chimaera::cxx for external
  set(CORE_LIB "")
  if(TARGET chimaera::cxx)
    set(CORE_LIB chimaera::cxx)
  elseif(TARGET ctp::cxx)
    set(CORE_LIB ctp::cxx)
  elseif(TARGET ClioCtp::cxx)
    set(CORE_LIB ClioCtp::cxx)
  elseif(TARGET cxx)
    set(CORE_LIB cxx)
  else()
    message(FATAL_ERROR "Neither chimaera::cxx, ctp::cxx, ClioCtp::cxx nor cxx target found")
  endif()

  # Runtime-specific link libraries
  set(CLIO_RUN_RUNTIME_LIBS
    Threads::Threads
  )

  # Automatically link to client library if it exists
  set(RUNTIME_LINK_LIBS ${CORE_LIB} ${CLIO_RUN_RUNTIME_LIBS} ${ARG_LINK_LIBRARIES})

  # Try to find client target by name (handles cases where client was defined first)
  set(CLIENT_TARGET_NAME "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}_client")
  if(TARGET ${CLIENT_TARGET_NAME})
    list(APPEND RUNTIME_LINK_LIBS ${CLIENT_TARGET_NAME})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CLIENT_TARGET_NAME}")
  elseif(CLIO_RUN_MODULE_CLIENT_TARGET AND TARGET ${CLIO_RUN_MODULE_CLIENT_TARGET})
    # Fallback to variable-based approach for compatibility
    list(APPEND RUNTIME_LINK_LIBS ${CLIO_RUN_MODULE_CLIENT_TARGET})
    message(STATUS "Runtime ${TARGET_NAME} linking to client ${CLIO_RUN_MODULE_CLIENT_TARGET}")
  endif()

  # Automatically add foundational ChiMod dependencies in unified builds.
  # See the matching block in add_clio_module_client for why we key off
  # CLIO_RUN_MODULE_NAME instead of CLIO_RUN_NAMESPACE.
  if(NOT "${CLIO_RUN_MODULE_NAME}" STREQUAL "admin" AND
     NOT "${CLIO_RUN_MODULE_NAME}" STREQUAL "bdev")
    if(TARGET clio_admin_runtime)
      list(APPEND RUNTIME_LINK_LIBS clio_admin_runtime)
    endif()
    if(TARGET clio_bdev_runtime)
      list(APPEND RUNTIME_LINK_LIBS clio_bdev_runtime)
    endif()
  endif()

  target_link_libraries(${TARGET_NAME}
    PUBLIC
      ${RUNTIME_LINK_LIBS}
  )
  # POSIX real-time library for async I/O. Linux-only — Windows ships its
  # AIO support in kernel32/winsock, not as a separate library.
  if(UNIX AND NOT APPLE)
    target_link_libraries(${TARGET_NAME} PUBLIC rt)
  endif()

  # Create alias for external use
  add_library(${CLIO_RUN_NAMESPACE}::${CLIO_RUN_MODULE_NAME}_runtime ALIAS ${TARGET_NAME})
  # Also expose the legacy `chimaera::` alias so consumers that pre-date
  # the module-namespace rename (e.g. coeus-adapter, in-tree CMakeLists
  # that link to chimaera::admin_runtime) keep working at build time.
  if(NOT TARGET chimaera::${CLIO_RUN_MODULE_NAME}_runtime)
    add_library(chimaera::${CLIO_RUN_MODULE_NAME}_runtime ALIAS ${TARGET_NAME})
  endif()

  # Set properties for installation. OUTPUT_NAME tracks LIB_NAME when given
  # so the .so file on disk matches the CMake target.  PACKAGE_NAME (not
  # NAMESPACE) for filesystem-safety as in add_clio_module_client.
  if(ARG_LIB_NAME)
    set(_chimod_output_name "${ARG_LIB_NAME}_runtime")
  else()
    set(_chimod_output_name "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}_runtime")
  endif()
  set_target_properties(${TARGET_NAME} PROPERTIES
    EXPORT_NAME "${CLIO_RUN_MODULE_NAME}_runtime"
    OUTPUT_NAME "${_chimod_output_name}"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
  )

  # Use cmake_language(DEFER) to link to client after all targets are processed
  cmake_language(EVAL CODE "
    cmake_language(DEFER CALL _clio_run_link_runtime_to_client \"${TARGET_NAME}\" \"${CLIENT_TARGET_NAME}\")
  ")

  # Install the runtime library (add to existing export set if client exists)
  # MODULE_PACKAGE_NAME: install dir under lib/cmake/. For runtime modules
  # whose namespace was renamed chimaera -> clio::run (admin, bdev, MOD_NAME)
  # we pin to the legacy `chimaera_<module>` form so external find_package
  # consumers (e.g. coeus-adapter) keep working. For other namespaces
  # (clio::cte, clio::cae, …) we use the standard `<package>_<module>` form
  # (filesystem-safe).
  if("${CLIO_RUN_NAMESPACE}" STREQUAL "clio::run")
    set(MODULE_PACKAGE_NAME "chimaera_${CLIO_RUN_MODULE_NAME}")
  else()
    set(MODULE_PACKAGE_NAME "${CLIO_RUN_PACKAGE_NAME}_${CLIO_RUN_MODULE_NAME}")
  endif()
  set(MODULE_EXPORT_NAME "${MODULE_PACKAGE_NAME}")

  install(TARGETS ${TARGET_NAME}
    EXPORT ${MODULE_EXPORT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
  )

  # Install headers (only if not already installed by client)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/include" AND NOT CLIO_RUN_MODULE_CLIENT_TARGET)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
      DESTINATION include
      FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
    )
  endif()

  # Generate and install package config files (only do this once per module)
  set(SHOULD_GENERATE_CONFIG FALSE)
  if(CLIO_RUN_MODULE_CLIENT_TARGET AND TARGET ${CLIO_RUN_MODULE_CLIENT_TARGET})
    set(SHOULD_GENERATE_CONFIG TRUE)
  elseif(NOT CLIO_RUN_MODULE_CLIENT_TARGET)
    set(SHOULD_GENERATE_CONFIG TRUE)
  endif()

  if(SHOULD_GENERATE_CONFIG)
    # Export targets file
    # NAMESPACE: for chimaera-renamed runtime modules (admin/bdev/MOD_NAME,
    # now under clio::run::), pin to `chimaera::` so installed targets keep
    # the legacy name external consumers (coeus-adapter etc.) expect. For
    # other modules use the canonical namespace. Either way the modern
    # alias namespace is emitted in Config.cmake below.
    set(_install_namespace "${CLIO_RUN_NAMESPACE}")
    if("${CLIO_RUN_NAMESPACE}" STREQUAL "clio::run")
      set(_install_namespace "chimaera")
    endif()
    install(EXPORT ${MODULE_EXPORT_NAME}
      FILE ${MODULE_EXPORT_NAME}.cmake
      NAMESPACE ${_install_namespace}::
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Compute additional namespace aliases to emit alongside the install-time
    # targets (which carry the `${_install_namespace}::` prefix).
    #
    # ALIAS_NAMESPACES is the list of EXTRA namespaces (not equal to the
    # install namespace) that should also resolve to the same exported
    # targets. Always includes the current CLIO_RUN_NAMESPACE so new code
    # can link via the canonical name (e.g. `clio::cte::core_client`).
    # For clio::* namespaces we also emit the historical `clio_<suffix>::`
    # (pre-`::` rename waypoint) and, when the suffix is one of the
    # WRP_->CLIO_ rename targets, the further-back `wrp_<suffix>::` alias.
    set(ALIAS_NAMESPACES "${CLIO_RUN_NAMESPACE}")
    if("${CLIO_RUN_NAMESPACE}" MATCHES "^clio::(.*)$")
      set(_suffix "${CMAKE_MATCH_1}")
      # Pre-`::` waypoint, e.g. clio::cte -> clio_cte.
      list(APPEND ALIAS_NAMESPACES "clio_${_suffix}")
      # "wrp_run" was never valid; only emit wrp_ for cte/cae/etc.
      if(NOT "${_suffix}" STREQUAL "run")
        list(APPEND ALIAS_NAMESPACES "wrp_${_suffix}")
      endif()
      unset(_suffix)
    endif()
    # Don't double-emit the install namespace itself.
    list(REMOVE_ITEM ALIAS_NAMESPACES "${_install_namespace}")
    list(REMOVE_DUPLICATES ALIAS_NAMESPACES)

    # Generate Config.cmake file
    set(CONFIG_CONTENT "
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)

# Find the core CLIO Runtime package (handles all other dependencies)
find_dependency(chimaera REQUIRED)

# Include the exported targets
include(\"\${CMAKE_CURRENT_LIST_DIR}/${MODULE_EXPORT_NAME}.cmake\")
")

    # Append IMPORTED ALIAS targets for every extra namespace in
    # ALIAS_NAMESPACES. The aliases forward to the install-time target
    # (`${_install_namespace}::<module>_<x>`) so downstream consumers can
    # link via any of the historically-valid names.
    foreach(_alias_ns IN LISTS ALIAS_NAMESPACES)
      string(APPEND CONFIG_CONTENT "
# --- Alias namespace ${_alias_ns}:: -> ${_install_namespace}:: ---
# Lets downstream projects that reference ${_alias_ns}::* keep working.
foreach(_legacy_tgt IN ITEMS ${CLIO_RUN_MODULE_NAME}_client ${CLIO_RUN_MODULE_NAME}_runtime)
  if(TARGET ${_install_namespace}::\${_legacy_tgt} AND NOT TARGET ${_alias_ns}::\${_legacy_tgt})
    add_library(${_alias_ns}::\${_legacy_tgt} INTERFACE IMPORTED)
    set_target_properties(${_alias_ns}::\${_legacy_tgt} PROPERTIES
      INTERFACE_LINK_LIBRARIES ${_install_namespace}::\${_legacy_tgt})
  endif()
endforeach()
")
    endforeach()

    string(APPEND CONFIG_CONTENT "
# Provide components
check_required_components(${MODULE_PACKAGE_NAME})
")

    # Write Config.cmake template
    set(CONFIG_IN_FILE "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake.in")
    file(WRITE "${CONFIG_IN_FILE}" "${CONFIG_CONTENT}")

    # Configure and install Config.cmake
    include(CMakePackageConfigHelpers)
    configure_package_config_file(
      "${CONFIG_IN_FILE}"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      INSTALL_DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Generate ConfigVersion.cmake
    write_basic_package_version_file(
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      VERSION 1.0.0
      COMPATIBILITY SameMajorVersion
    )

    # Install Config and ConfigVersion files
    install(FILES
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}Config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_PACKAGE_NAME}ConfigVersion.cmake"
      DESTINATION lib/cmake/${MODULE_PACKAGE_NAME}
    )

    # Collect targets for status message
    set(INSTALLED_TARGETS ${TARGET_NAME})
    if(CLIO_RUN_MODULE_CLIENT_TARGET AND TARGET ${CLIO_RUN_MODULE_CLIENT_TARGET})
      list(APPEND INSTALLED_TARGETS ${CLIO_RUN_MODULE_CLIENT_TARGET})
    endif()

    message(STATUS "Created module package: ${MODULE_PACKAGE_NAME}")
    message(STATUS "  Targets: ${INSTALLED_TARGETS}")
    message(STATUS "  Aliases: ${CLIO_RUN_NAMESPACE}::${CLIO_RUN_MODULE_NAME}_client, ${CLIO_RUN_NAMESPACE}::${CLIO_RUN_MODULE_NAME}_runtime")
  endif()

  # Precompiled headers for faster builds
  if(CLIO_CORE_ENABLE_PCH)
    target_precompile_headers(${TARGET_NAME} PRIVATE
        <string> <vector> <memory> <unordered_map>
        <functional> <algorithm> <cstdint> <cstring> <iostream>
    )
  endif()

  # Export module info to parent scope
  set(CLIO_RUN_MODULE_RUNTIME_TARGET ${TARGET_NAME} PARENT_SCOPE)
  set(CLIO_RUN_MODULE_NAME ${CLIO_RUN_MODULE_NAME} PARENT_SCOPE)
  set(CLIO_RUN_NAMESPACE ${CLIO_RUN_NAMESPACE} PARENT_SCOPE)
  set(CLIO_RUN_PACKAGE_NAME ${CLIO_RUN_PACKAGE_NAME} PARENT_SCOPE)
endfunction()

message(STATUS "IowarpCoreCommon.cmake loaded successfully")

#==============================================================================
# Backward-compat aliases (CHIMAERA_* / chimaera_* / add_chimod_* names)
#
# All CMake-facing identifiers in this file were renamed CHIMAERA_* -> CLIO_RUN_*
# and the helper functions gained `clio` branding (e.g. add_chimod_client ->
# add_clio_module_client). The legacy names below are macro wrappers that
# forward to the new names so out-of-tree CMake code keeps building unchanged.
# See docs/deprecation-notes.md.
#==============================================================================

# Function wrappers
macro(add_chimod_client)
  add_clio_module_client(${ARGV})
endmacro()

macro(add_chimod_runtime)
  add_clio_module_runtime(${ARGV})
endmacro()

macro(chimaera_read_module_config _dir)
  clio_run_read_module_config(${_dir})
  # Re-export the new variables under their legacy CHIMAERA_* names so any
  # CMakeLists.txt that still reads e.g. `${CHIMAERA_MODULE_NAME}` keeps
  # working transparently.
  set(CHIMAERA_NAMESPACE             "${CLIO_RUN_NAMESPACE}"             PARENT_SCOPE)
  set(CHIMAERA_MODULE_NAME           "${CLIO_RUN_MODULE_NAME}"           PARENT_SCOPE)
endmacro()


# ---------------------------------------------------------------------------
# clio_add_force_net_test
# ---------------------------------------------------------------------------
# Register a CLIO_FORCE_NET=1 variant of a runtime test. The whole test binary
# runs with every non-Local task routed through the loopback run-to-run network
# path (SendIn/RecvIn/SendOut/RecvOut + serialize/deserialize/aggregate), so the
# network path is exercised for all task types even on a single node. Mirrors
# the original cr_bdev_force_net stress test.
#
# Such tests are always:
#   * LABELS "msan_skip"            (libzmq is uninstrumented)
#   * RESOURCE_LOCK "chimaera_runtime" (one runtime owns the fixed port at a time)
#   * given CLIO_FORCE_NET=1 appended to the supplied ENVIRONMENT
#
# Usage:
#   clio_add_force_net_test(
#     NAME <test_name>
#     COMMAND <exe> [args...]
#     ENVIRONMENT "<semicolon-separated env without CLIO_FORCE_NET>"
#     [WORKING_DIRECTORY <dir>]
#     [TIMEOUT <seconds>])      # defaults to 300
function(clio_add_force_net_test)
  # macOS: the force-net loopback path hangs for several task types
  # (GetBlob/ReadData returns 0 bytes -> Wait() wedges), timing out the Mac
  # Test suite. Pass on Windows + Linux. Gated off Apple until the macOS
  # data-path hang is fixed (issue #504). The original cr_bdev_force_net does
  # not use this helper and still runs on macOS.
  if(APPLE)
    return()
  endif()

  cmake_parse_arguments(FN "" "NAME;WORKING_DIRECTORY;TIMEOUT"
                        "COMMAND;ENVIRONMENT" ${ARGN})
  if(NOT FN_NAME)
    message(FATAL_ERROR "clio_add_force_net_test: NAME is required")
  endif()
  if(NOT FN_COMMAND)
    message(FATAL_ERROR "clio_add_force_net_test: COMMAND is required")
  endif()
  if(NOT FN_TIMEOUT)
    set(FN_TIMEOUT 300)
  endif()

  add_test(NAME ${FN_NAME} COMMAND ${FN_COMMAND})
  set_tests_properties(${FN_NAME} PROPERTIES
    TIMEOUT ${FN_TIMEOUT}
    LABELS "msan_skip"
    RESOURCE_LOCK "chimaera_runtime"
    ENVIRONMENT "${FN_ENVIRONMENT};CLIO_FORCE_NET=1")
  if(FN_WORKING_DIRECTORY)
    set_tests_properties(${FN_NAME} PROPERTIES
      WORKING_DIRECTORY "${FN_WORKING_DIRECTORY}")
  endif()
endfunction()
