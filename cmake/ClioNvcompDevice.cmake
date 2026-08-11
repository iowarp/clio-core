#------------------------------------------------------------------------------
# clio_enable_nvcomp_device(<target>)
#
# Turns on gpu_vector's IN-KERNEL nvcomp decode for one target: encoded pages in
# the kHbm tier are decompressed by the faulting warp itself, device pointer in
# and device pointer out, with no host round trip and no host staging.
#
# Without it, device_vector.h compiles a single-threaded scalar LZ4 decoder as
# the airtight fallback. That fallback is CORRECT but roughly two orders of
# magnitude slower than nvcomp's warp-cooperative decoder, so a benchmark that
# silently lands on it reports "nvcomp is slow" while never calling nvcomp.
# Anything measuring the compressed fault path MUST call this and MUST fail if
# it does not take effect -- see CLIO_NVCOMP_DEVICE_REQUIRED below.
#
# EXECUTABLES ONLY. nvcomp ships decompress_device.cu.o non-PIC, so a shared
# library needs the fatbin-rewrap dance in ggml-cuda-clio's CMakeLists; an
# executable links the archive directly.
#
# Point CLIO_NVCOMP_DEVICE_ROOT at an nvcomp archive that has
# lib/libnvcomp_device_static.a and include/nvcomp/device/user_api.hpp (the
# distro nvcomp package ships only the HOST manager API and will not do).
#------------------------------------------------------------------------------

set(CLIO_NVCOMP_DEVICE_ROOT "" CACHE PATH
    "nvcomp archive providing libnvcomp_device_static.a for in-kernel decode")
option(CLIO_NVCOMP_DEVICE_REQUIRED
       "Fail configuration if in-kernel nvcomp decode is unavailable" OFF)

# Probe the usual locations so a developer who unpacked the archive in a
# standard place needs no -D flag.
if (NOT CLIO_NVCOMP_DEVICE_ROOT)
    file(GLOB _clio_nvdev_cands
         "/opt/nvcomp-linux-*-archive"
         "$ENV{HOME}/opt/nvcomp-linux-*-archive")
    foreach(_c ${_clio_nvdev_cands})
        if (EXISTS "${_c}/lib/libnvcomp_device_static.a" AND
            EXISTS "${_c}/include/nvcomp/device/user_api.hpp")
            set(CLIO_NVCOMP_DEVICE_ROOT "${_c}" CACHE PATH "" FORCE)
            break()
        endif()
    endforeach()
endif()

function(clio_enable_nvcomp_device target)
    get_target_property(_type ${target} TYPE)
    if (NOT _type STREQUAL "EXECUTABLE")
        message(FATAL_ERROR
            "clio_enable_nvcomp_device(${target}): EXECUTABLE targets only "
            "(nvcomp's device object is non-PIC). See ggml-cuda-clio for the "
            "shared-library recipe.")
    endif()

    if (NOT CLIO_NVCOMP_DEVICE_ROOT OR
        NOT EXISTS "${CLIO_NVCOMP_DEVICE_ROOT}/lib/libnvcomp_device_static.a")
        set(_msg
            "${target}: in-kernel nvcomp decode OFF -- no nvcomp device "
            "archive (set CLIO_NVCOMP_DEVICE_ROOT). The compressed fault "
            "path will use the scalar LZ4 fallback, which is ~100x slower "
            "and is NOT a measurement of nvcomp.")
        if (CLIO_NVCOMP_DEVICE_REQUIRED)
            message(FATAL_ERROR ${_msg})
        endif()
        message(WARNING ${_msg})
        return()
    endif()

    # -rdc: the nvcomp device decompressor is an external device function, so
    # the call must survive to a separate device link step.
    set_target_properties(${target} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(${target} PRIVATE CLIO_GV_NVCOMP_DEVICE)
    target_include_directories(${target} BEFORE PRIVATE
        "${CLIO_NVCOMP_DEVICE_ROOT}/include")
    # Plain signature on purpose: add_cuda_executable() links these targets
    # without keywords, and CMake forbids mixing the two forms on one target.
    target_link_libraries(${target}
        "${CLIO_NVCOMP_DEVICE_ROOT}/lib/libnvcomp_device_static.a")

    # nvcomp ships these objects non-PIC, which a PIE executable cannot absorb
    # ("relocation R_X86_64_32 ... can not be used when making a PIE object").
    # Link this one target non-PIE; that is a property of nvcomp's shipped
    # archive, not something the fault path can work around.
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE OFF)
    target_link_options(${target} PRIVATE -no-pie)
    message(STATUS "${target}: in-kernel nvcomp decode ON "
                   "(${CLIO_NVCOMP_DEVICE_ROOT})")
endfunction()
