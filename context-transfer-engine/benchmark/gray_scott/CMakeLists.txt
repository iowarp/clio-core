cmake_minimum_required(VERSION 3.20)

project(iowarp-gray-scott VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project options
option(ENABLE_RPATH "Enable RPATH for installed binaries (recommended)" ON)
option(USE_TIMERS "Use profiling timers" OFF)
option(ENABLE_VTK "Build VTK visualization apps" OFF)

# Find required packages
if(NOT MPI_FOUND)
  find_package(MPI REQUIRED)
endif()
if(NOT ADIOS2_FOUND)
  find_package(ADIOS2 REQUIRED)
endif()

# Configuration messages
message(STATUS "========================================")
message(STATUS "IOWarp Gray-Scott Simulation")
message(STATUS "========================================")
message(STATUS "Build configuration:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  ENABLE_RPATH: ${ENABLE_RPATH}")
message(STATUS "  USE_TIMERS: ${USE_TIMERS}")
message(STATUS "  ENABLE_VTK: ${ENABLE_VTK}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  MPI_C_COMPILER: ${MPI_C_COMPILER}")
message(STATUS "  MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
message(STATUS "  ADIOS2_DIR: ${ADIOS2_DIR}")
message(STATUS "========================================")

# RPATH Configuration
# This ensures binaries can find shared libraries without LD_LIBRARY_PATH
if(ENABLE_RPATH)
    # Use RPATH for build tree
    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # When building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

    # CRITICAL: Use old-style RPATH instead of RUNPATH
    # RPATH is searched BEFORE LD_LIBRARY_PATH, while RUNPATH is searched AFTER
    # This is essential when LD_LIBRARY_PATH contains system paths that conflict
    # with conda libraries (e.g., OpenSSL version mismatch)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")

    # Set the RPATH to be used when installing
    # Include conda lib path for ADIOS2 dependencies (libcurl, OpenSSL, etc.)
    set(CMAKE_INSTALL_RPATH
        "${CMAKE_INSTALL_PREFIX}/lib"
    )

    message(STATUS "RPATH enabled (old-style, checked before LD_LIBRARY_PATH):")
    message(STATUS "  Install RPATH: ${CMAKE_INSTALL_RPATH}")
endif()

# Compile options
if(USE_TIMERS)
    message(STATUS "Enabling profiling timers")
    add_compile_definitions(ENABLE_TIMERS)
endif()

# We are not using the C++ API of MPI
add_compile_definitions(OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)

# ============================================================================
# Simulation executable
# ============================================================================
add_executable(gray-scott
    simulation/main.cpp
    simulation/gray-scott.cpp
    simulation/settings.cpp
    simulation/writer.cpp
)

target_include_directories(gray-scott PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(gray-scott
    PRIVATE
        adios2::adios2
        MPI::MPI_C
)

# Set output directories
set_target_properties(gray-scott PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# Analysis executable: pdf_calc
# ============================================================================
add_executable(pdf_calc
    analysis/pdf_calc.cpp
)

target_include_directories(pdf_calc PRIVATE ${MPI_C_INCLUDE_DIRS})
target_link_libraries(pdf_calc
    PRIVATE
        adios2::adios2
        MPI::MPI_C
)

set_target_properties(pdf_calc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ============================================================================
# VTK-based analysis and visualization tools (optional)
# ============================================================================
if(ENABLE_VTK OR VTK_ROOT)
    message(STATUS "Configuring VTK apps")

    # Isosurface extraction
    find_package(VTK QUIET COMPONENTS
        vtkFiltersCore
        vtkIOImage
        vtkIOXML
    )

    if(VTK_FOUND)
        message(STATUS "  Building isosurface (VTK ${VTK_VERSION})")
        add_executable(isosurface analysis/isosurface.cpp)
        target_include_directories(isosurface PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(isosurface
            PRIVATE
                adios2::adios2
                ${VTK_LIBRARIES}
                MPI::MPI_C
        )
        set_target_properties(isosurface PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    else()
        message(STATUS "  Skipping isosurface (VTK not found)")
    endif()

    # Blob detection
    find_package(VTK QUIET COMPONENTS
        vtkFiltersCore
        vtkFiltersGeometry
    )

    if(VTK_FOUND)
        message(STATUS "  Building find_blobs (VTK ${VTK_VERSION})")
        add_executable(find_blobs analysis/find_blobs.cpp)
        target_include_directories(find_blobs PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(find_blobs
            PRIVATE
                adios2::adios2
                ${VTK_LIBRARIES}
                MPI::MPI_C
        )
        set_target_properties(find_blobs PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    else()
        message(STATUS "  Skipping find_blobs (VTK not found)")
    endif()

    # Curvature computation
    find_package(VTK QUIET COMPONENTS
        vtkFiltersGeneral
    )

    if(VTK_FOUND)
        message(STATUS "  Building compute_curvature (VTK ${VTK_VERSION})")
        add_executable(compute_curvature analysis/curvature.cpp)
        target_include_directories(compute_curvature PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(compute_curvature
            PRIVATE
                adios2::adios2
                ${VTK_LIBRARIES}
                MPI::MPI_C
        )
        set_target_properties(compute_curvature PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    else()
        message(STATUS "  Skipping compute_curvature (VTK not found)")
    endif()

    # Rendering
    find_package(VTK QUIET COMPONENTS
        vtkRenderingOpenGL2
        vtkViewsInfovis
    )

    if(VTK_FOUND)
        message(STATUS "  Building render_isosurface (VTK ${VTK_VERSION})")
        add_executable(render_isosurface plot/render_isosurface.cpp)
        target_include_directories(render_isosurface PRIVATE ${MPI_C_INCLUDE_DIRS})
        target_link_libraries(render_isosurface
            PRIVATE
                adios2::adios2
                ${VTK_LIBRARIES}
                MPI::MPI_C
        )
        set_target_properties(render_isosurface PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        )
    else()
        message(STATUS "  Skipping render_isosurface (VTK not found)")
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================
# Install binaries to CMAKE_INSTALL_PREFIX/bin
install(TARGETS gray-scott pdf_calc
    RUNTIME DESTINATION bin
)

# Install VTK apps if built
if(TARGET isosurface)
    install(TARGETS isosurface RUNTIME DESTINATION bin)
endif()

if(TARGET find_blobs)
    install(TARGETS find_blobs RUNTIME DESTINATION bin)
endif()

if(TARGET compute_curvature)
    install(TARGETS compute_curvature RUNTIME DESTINATION bin)
endif()

if(TARGET render_isosurface)
    install(TARGETS render_isosurface RUNTIME DESTINATION bin)
endif()

# Install configuration files
install(FILES adios2.xml
    DESTINATION share/iowarp-gray-scott
)

# Install scripts
install(PROGRAMS
    scripts/cleanup.sh
    scripts/run-with-conda.sh
    DESTINATION share/iowarp-gray-scott/scripts
)

message(STATUS "========================================")
message(STATUS "Install configuration:")
message(STATUS "  Binaries: ${CMAKE_INSTALL_PREFIX}/bin")
message(STATUS "  Config: ${CMAKE_INSTALL_PREFIX}/share/iowarp-gray-scott")
message(STATUS "  Scripts: ${CMAKE_INSTALL_PREFIX}/share/iowarp-gray-scott/scripts")
message(STATUS "========================================")
