# Rust bindings built via Corrosion when CLIO_CORE_ENABLE_RUST=ON
# This file is only included when CLIO_CORE_ENABLE_RUST=ON (see parent wrapper/CMakeLists.txt)

cmake_minimum_required(VERSION 3.20)

# Verify Corrosion is available
if(NOT COMMAND corrosion_import_crate)
    message(FATAL_ERROR "Corrosion not available. Ensure CLIO_CORE_ENABLE_RUST=ON in root CMakeLists.txt")
endif()

# Define paths for aneris-profiler build (relative to wrapper/rust directory)
# Use CMAKE_CURRENT_LIST_DIR which always points to this directory,
# not CMAKE_CURRENT_SOURCE_DIR which can change based on who includes this file
get_filename_component(HSHM_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../context-transport-primitives" ABSOLUTE)
get_filename_component(CHIMAERA_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../context-runtime" ABSOLUTE)
# CTE_ROOT should be the parent of wrapper/rust directory (context-transfer-engine)
get_filename_component(CTE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# CTE_CORE_ROOT should be context-transfer-engine/core
get_filename_component(CTE_CORE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../core" ABSOLUTE)
# CHIMODS_ROOT should be the parent of context-runtime (context-transfer-engine)
# Note: CHIMODS_ROOT is the modules directory under context-runtime, so:
# context-runtime/modules = context-runtime/.. + modules = context-transfer-engine + modules
get_filename_component(CHIMODS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../context-runtime/modules" ABSOLUTE)

# Get cereal include directory
get_target_property(CEREAL_INCLUDE_DIR cereal::cereal INTERFACE_INCLUDE_DIRECTORIES)
if(NOT CEREAL_INCLUDE_DIR)
    get_filename_component(CEREAL_ROOT "${cereal_DIR}/../../.." ABSOLUTE)
    set(CEREAL_INCLUDE_DIR "${CEREAL_ROOT}/include")
endif()

#------------------------------------------------------------------------------
# Aneris Profiler Build Configuration
#------------------------------------------------------------------------------
# Define RUSTFLAGS for aneris compile-time paths
# Use semicolon-separated list for proper argument splitting
set(ANERIS_RUSTFLAGS "--cfg;aneris_build_dir=\\\"${CMAKE_BINARY_DIR}\\\";--cfg;aneris_source_dir=\\\"${CMAKE_SOURCE_DIR}\\\"" CACHE STRING "RUSTFLAGS for aneris compile-time paths")

# Make it available to parent scope
set(ANERIS_RUSTFLAGS ${ANERIS_RUSTFLAGS} PARENT_SCOPE)

#------------------------------------------------------------------------------
# Corrosion: Import the Rust crate
#------------------------------------------------------------------------------
# Import the clio-cte-rs crate and build it with Corrosion
# This creates CMake targets for the Rust binaries and library
corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_LIST_DIR}/Cargo.toml"
)

# corrosion_import_crate creates the following targets based on Cargo.toml:
# - clio_cte (main library target, resolves to cdylib or staticlib based on BUILD_SHARED_LIBS)
# - clio_cte-shared (explicit cdylib target)
# - clio_cte-static (explicit staticlib target)
# - clio-cte-rs-aneris-profiler (binary)
# - clio-cte-rs-aneris-rescorer (binary)
# - clio-cte-rs-aneris-tune (binary)

# Apply RUSTFLAGS for aneris compile-time paths to all Rust targets
# Use corrosion_add_target_local_rustflags to avoid rebuilding dependencies
# Corrosion creates targets named after the binary name (not crate-binary format)
# Public targets: clio_cte (library), clio-cte-rs-aneris-* (binaries)
# Internal targets: _cargo-build_clio_cte, _cargo-build_aneris-*
set(RUST_BIN_TARGETS
    _cargo-build_aneris-profiler
    _cargo-build_aneris-rescorer
    _cargo-build_aneris-tune
)

# Apply RUSTFLAGS to library target using internal build target
if(TARGET _cargo-build_clio_cte)
    corrosion_add_target_local_rustflags(_cargo-build_clio_cte "${ANERIS_RUSTFLAGS}")
    message(STATUS "Applied RUSTFLAGS to _cargo-build_clio_cte")
endif()

# Apply RUSTFLAGS to binary targets using internal build targets
foreach(rust_target IN LISTS RUST_BIN_TARGETS)
    if(TARGET ${rust_target})
        corrosion_add_target_local_rustflags(${rust_target} "${ANERIS_RUSTFLAGS}")
    endif()
endforeach()

# Set environment variables for the Rust build so build.rs can find C++ headers
# These are read by build.rs and take precedence over fallback paths
# CRITICAL: Environment variables must be set on ALL targets (library + binaries)
set(RUST_ENV_VARS
    "IOWARP_INCLUDE_DIR=${HSHM_ROOT}/include"
    "IOWARP_EXTRA_INCLUDES=${CHIMAERA_ROOT}/include:${CTE_CORE_ROOT}/include:${CEREAL_INCLUDE_DIR}:${CHIMODS_ROOT}/admin/include:${CHIMODS_ROOT}/bdev/include"
    "IOWARP_LIB_DIR=${CMAKE_BINARY_DIR}/bin"
    "IOWARP_ZMQ_LIBS=${ZMQ_LIBS}"
    "IOWARP_ZMQ_LIB_DIRS=${ZMQ_LIB_DIRS}"
)

# Apply environment variables to the Rust library target
# Corrosion creates the _cargo-build_clio_cte target for the library (internal build target)
if(TARGET _cargo-build_clio_cte)
    # Use corrosion_set_env_vars to properly set environment variables
    corrosion_set_env_vars(_cargo-build_clio_cte ${RUST_ENV_VARS})
    message(STATUS "Set environment variables for _cargo-build_clio_cte")
    # Debug: print the env vars
    foreach(env_var IN LISTS RUST_ENV_VARS)
        message(STATUS "  - ${env_var}")
    endforeach()
    
    # Also add RUSTFLAGS for library search path
    corrosion_add_target_local_rustflags(_cargo-build_clio_cte "-Clink-arg=-L${CMAKE_BINARY_DIR}/bin")
    message(STATUS "Added RUSTFLAGS for library search path: -L${CMAKE_BINARY_DIR}/bin")
    
    # The Rust library also needs the C++ libraries to be built first
    # so the shared libraries exist for linking
    add_dependencies(_cargo-build_clio_cte
        clio_cte_core_client
        clio_run_cxx
        clio_ctp_host
    )
    message(STATUS "Added dependencies for _cargo-build_clio_cte")
endif()

# Apply environment variables and link dependencies to binary targets
# This is critical - each binary has its own build context
foreach(rust_target IN LISTS RUST_BIN_TARGETS)
    if(TARGET ${rust_target})
        corrosion_set_env_vars(${rust_target} ${RUST_ENV_VARS})
        message(STATUS "Set environment variables for ${rust_target}")
        
        # Add RUSTFLAGS for library search path
        corrosion_add_target_local_rustflags(${rust_target} "-Clink-arg=-L${CMAKE_BINARY_DIR}/bin")
        
        # The Rust binaries need the C++ libraries to be built first
        # so the shared libraries exist for linking
        # Use actual target names (not legacy aliases):
        # - clio_cte_core_client: CTE client library
        # - clio_run_cxx: Core runtime library (alias: chimaera_cxx)
        # - clio_ctp_host: CTP host library (old name: hermes_shm_host)
        # - _cargo-build_clio_cte: Rust library must be built first
        add_dependencies(${rust_target}
            clio_cte_core_client
            clio_run_cxx
            clio_ctp_host
            _cargo-build_clio_cte
        )
        message(STATUS "Added dependencies for ${rust_target}")
    else()
        message(STATUS "Rust binary target not found (may not be built): ${rust_target}")
    endif()
endforeach()

# Create aneris subdirectory for wrapper scripts to avoid conflicts with tests
# that run with WORKING_DIRECTORY=${CMAKE_BINARY_DIR}/bin
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin/aneris)

# Generate complete aneris_telemetry wrapper script from template
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/Aneris_telemetry.in
    ${CMAKE_BINARY_DIR}/bin/aneris/aneris_telemetry
    @ONLY
)

file(CHMOD ${CMAKE_BINARY_DIR}/bin/aneris/aneris_telemetry
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "Generated aneris_telemetry wrapper: ${CMAKE_BINARY_DIR}/bin/aneris/aneris_telemetry")

# ADIOS2 telemetry script
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/Aneris_telemetry_adios.in
    ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_telemetry_adios
    @ONLY
)

file(CHMOD ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_telemetry_adios
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "Generated Aneris_telemetry_adios wrapper: ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_telemetry_adios")

# ADIOS2 tuning script
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/Aneris_tune_adios.in
    ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_tune_adios
    @ONLY
)

file(CHMOD ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_tune_adios
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "Generated Aneris_tune_adios wrapper: ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_tune_adios")

# ADIOS2 intercept script (I/O interception without runtime/telemetry)
configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/Aneris_intercept_adios.in
    ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_intercept_adios
    @ONLY
)

file(CHMOD ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_intercept_adios
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "Generated Aneris_intercept_adios wrapper: ${CMAKE_BINARY_DIR}/bin/aneris/Aneris_intercept_adios")
