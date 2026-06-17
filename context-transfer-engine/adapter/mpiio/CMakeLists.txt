include_directories(
        ${CLIO_CTE_ROOT}
        ${HERMES_SRC_DIR}
        ${HERMES_ADAPTER_DIR}
        ${HERMES_IO_CLIENT_DIR}
        .)

# Detect MPI implementation. The MPIIO adapter's UpdateIoStatus reads
# implementation-specific fields off MPI_Status, so it has to know
# whether we link against OpenMPI or MPICH. Probe by compiling a tiny
# snippet against the resolved MPI headers.
# Detect MPI implementation via `mpicc -showme:version` (OpenMPI) or
# `mpicc -v` (MPICH). The adapter's UpdateIoStatus reads
# implementation-specific MPI_Status fields, so it has to know which.
find_package(MPI REQUIRED)
if(CLIO_CTE_MPICH)
        message(STATUS "MPIIO adapter: forced MPICH via -DWRP_CTE_MPICH")
        add_definitions(-DWRP_CTE_MPICH)
elseif(CLIO_CTE_OPENMPI)
        message(STATUS "MPIIO adapter: forced OpenMPI via -DWRP_CTE_OPENMPI")
        add_definitions(-DWRP_CTE_OPENMPI)
else()
        execute_process(
                COMMAND ${MPI_C_COMPILER} -showme:version
                OUTPUT_VARIABLE _mpiio_showme
                ERROR_VARIABLE _mpiio_showme_err
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_STRIP_TRAILING_WHITESPACE)
        if(_mpiio_showme MATCHES "[Oo]pen ?MPI"
           OR _mpiio_showme_err MATCHES "[Oo]pen ?MPI")
                message(STATUS "MPIIO adapter: detected OpenMPI")
                add_definitions(-DWRP_CTE_OPENMPI)
        else()
                execute_process(
                        COMMAND ${MPI_C_COMPILER} -compile_info
                        OUTPUT_VARIABLE _mpiio_compile_info
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET)
                if(_mpiio_compile_info MATCHES "mpich"
                   OR _mpiio_compile_info MATCHES "MPICH")
                        message(STATUS "MPIIO adapter: detected MPICH")
                        add_definitions(-DWRP_CTE_MPICH)
                else()
                        message(FATAL_ERROR
                                "MPIIO adapter: could not detect MPI implementation. "
                                "Pass -DWRP_CTE_OPENMPI=ON or -DWRP_CTE_MPICH=ON.")
                endif()
        endif()
endif()

add_library(clio_cte_mpiio SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/mpiio_api.cc)
target_link_libraries(clio_cte_mpiio
        MPI::MPI_CXX
        clio_cte_cfs_adapter      # cfs-backed I/O core (context-filesystem chimod)
        clio_cte_core_client
        ctp::cxx
        ctp::interceptor
        dl)
set_target_properties(clio_cte_mpiio PROPERTIES POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Add Target(s) to CMake Install
# -----------------------------------------------------------------------------
install(
        TARGETS
        clio_cte_mpiio
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

# -----------------------------------------------------------------------------
# Install headers
# -----------------------------------------------------------------------------
file(GLOB_RECURSE CLIO_CTE_HEADERS "*.h")
install(
        FILES
        ${CLIO_CTE_HEADERS}
        DESTINATION
        include/adapter/mpiio
        COMPONENT
        headers
)

# -----------------------------------------------------------------------------
