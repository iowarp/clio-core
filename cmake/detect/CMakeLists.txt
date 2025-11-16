# cmake/detect/CMakeLists.txt - Dependency Detection
#
# This CMakeLists.txt detects which dependencies are missing and sets boolean
# variables that can be used by the install script.
#
# Usage:
#   cmake -S cmake/detect -B build/detect -DCMAKE_PREFIX_PATH=/path/to/prefix
#   cmake --build build/detect
#
# Output variables:
#   NEED_BOOST - TRUE if Boost is missing
#   NEED_ZEROMQ - TRUE if ZeroMQ is missing
#   NEED_HDF5 - TRUE if HDF5 is missing (optional)

cmake_minimum_required(VERSION 3.20)
project(DependencyDetection)

# Find PkgConfig for various dependency checks
find_package(PkgConfig QUIET)

# Initialize all detection variables to TRUE (assume missing)
set(NEED_BOOST TRUE)
set(NEED_ZEROMQ TRUE)
set(NEED_HDF5 TRUE)

# Output file for dependency detection results
set(DETECTION_OUTPUT "${CMAKE_BINARY_DIR}/dependency_status.txt")

#------------------------------------------------------------------------------
# Boost Detection
#------------------------------------------------------------------------------
find_package(Boost QUIET COMPONENTS fiber context system filesystem atomic)
if(Boost_FOUND)
    set(NEED_BOOST FALSE)
    message(STATUS "✓ Boost found at ${Boost_INCLUDE_DIRS}")
else()
    # Also check for static libraries in prefix
    if(EXISTS "${CMAKE_PREFIX_PATH}/lib/libboost_fiber.a")
        set(NEED_BOOST FALSE)
        message(STATUS "✓ Boost libraries found in ${CMAKE_PREFIX_PATH}/lib")
    else()
        message(STATUS "✗ Boost NOT found - will build from submodule")
    endif()
endif()

#------------------------------------------------------------------------------
# ZeroMQ Detection
#------------------------------------------------------------------------------
if(PkgConfig_FOUND)
    pkg_check_modules(ZeroMQ QUIET libzmq)
    if(ZeroMQ_FOUND)
        set(NEED_ZEROMQ FALSE)
        message(STATUS "✓ ZeroMQ found at ${ZeroMQ_INCLUDE_DIRS}")
    else()
        # Also check for static library in prefix
        if(EXISTS "${CMAKE_PREFIX_PATH}/lib/libzmq.a")
            set(NEED_ZEROMQ FALSE)
            message(STATUS "✓ ZeroMQ library found in ${CMAKE_PREFIX_PATH}/lib")
        else()
            message(STATUS "✗ ZeroMQ NOT found - will build from submodule")
        endif()
    endif()
else()
    message(STATUS "⚠ PkgConfig not found, skipping ZeroMQ detection")
endif()

#------------------------------------------------------------------------------
# HDF5 Detection (Optional)
#------------------------------------------------------------------------------
find_package(HDF5 QUIET COMPONENTS C)
if(HDF5_FOUND)
    set(NEED_HDF5 FALSE)
    message(STATUS "✓ HDF5 found at ${HDF5_INCLUDE_DIRS}")
else()
    # Also check for static library in prefix
    if(EXISTS "${CMAKE_PREFIX_PATH}/lib/libhdf5.a")
        set(NEED_HDF5 FALSE)
        message(STATUS "✓ HDF5 library found in ${CMAKE_PREFIX_PATH}/lib")
    else()
        message(STATUS "✗ HDF5 NOT found - will build from submodule")
    endif()
endif()

#------------------------------------------------------------------------------
# Write Detection Results to File
#------------------------------------------------------------------------------
file(WRITE ${DETECTION_OUTPUT} "# Dependency Detection Results\n")
file(APPEND ${DETECTION_OUTPUT} "# Generated: ${CMAKE_CURRENT_LIST_FILE}\n\n")

file(APPEND ${DETECTION_OUTPUT} "NEED_BOOST=${NEED_BOOST}\n")
file(APPEND ${DETECTION_OUTPUT} "NEED_ZEROMQ=${NEED_ZEROMQ}\n")
file(APPEND ${DETECTION_OUTPUT} "NEED_HDF5=${NEED_HDF5}\n")

message(STATUS "")
message(STATUS "Dependency detection results written to: ${DETECTION_OUTPUT}")
message(STATUS "")

#------------------------------------------------------------------------------
# Summary
#------------------------------------------------------------------------------
message(STATUS "=================================================================")
message(STATUS "Dependency Detection Summary:")
message(STATUS "=================================================================")
message(STATUS "  Boost:       ${NEED_BOOST}")
message(STATUS "  ZeroMQ:      ${NEED_ZEROMQ}")
message(STATUS "  HDF5:        ${NEED_HDF5}")
message(STATUS "=================================================================")
message(STATUS "")

# Create a custom target that always runs (for scripting integration)
add_custom_target(detect ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Dependency detection complete"
    COMMENT "Running dependency detection"
)
