#------------------------------------------------------------------------------
# CPack Configuration
# Handles DEB, RPM, and TGZ package generation with flexible control
#------------------------------------------------------------------------------

# Common CPack metadata
set(CPACK_PACKAGE_NAME "iowarp-core")
set(CPACK_PACKAGE_VENDOR "IOWarp Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "IOWarp Core: High-performance distributed I/O and task execution runtime")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_CONTACT "grc@illinoistech.edu")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Only package the default (Unspecified) component — excludes pip_package files
# such as iowarp_core.pth which are wheel-only and must not land in system packages.
# CPACK_*_COMPONENT_INSTALL enables component-aware mode so CPACK_COMPONENTS_ALL
# is actually respected; without it CPack includes all components unconditionally.
set(CPACK_COMPONENTS_ALL Unspecified)
set(CPACK_DEB_COMPONENT_INSTALL ON)
set(CPACK_RPM_COMPONENT_INSTALL ON)
# Keep a single combined package (don't split per-component)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

# Determine which generators to enable based on options
set(CPACK_GENERATORS_ENABLED OFF)

# DEB Package Configuration
if(CLIO_CORE_ENABLE_DEB_PACKAGE OR CLIO_CORE_ENABLE_CPACK)
    list(APPEND CPACK_GENERATOR "DEB")
    set(CPACK_GENERATORS_ENABLED ON)

    # DEB-specific settings
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "IOWarp Team <grc@illinoistech.edu>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    # Runtime dependencies: libzmq5 or libzmq3-dev, libyaml-cpp0.8 or libyaml-cpp-dev
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libzmq3-dev, libyaml-cpp-dev")

    message(STATUS "CPack: DEB generator enabled")
endif()

# RPM Package Configuration
if(CLIO_CORE_ENABLE_RPM_PACKAGE OR CLIO_CORE_ENABLE_CPACK)
    list(APPEND CPACK_GENERATOR "RPM")
    set(CPACK_GENERATORS_ENABLED ON)

    # RPM-specific settings
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "System/Libraries")
    set(CPACK_RPM_PACKAGE_REQUIRES "zeromq, yaml-cpp")

    message(STATUS "CPack: RPM generator enabled")
endif()

# Legacy TGZ support (from CLIO_CORE_ENABLE_CPACK)
if(CLIO_CORE_ENABLE_CPACK)
    if(NOT "TGZ" IN_LIST CPACK_GENERATOR)
        list(APPEND CPACK_GENERATOR "TGZ")
    endif()
    set(CPACK_GENERATORS_ENABLED ON)

    message(STATUS "CPack: TGZ generator enabled (legacy CLIO_CORE_ENABLE_CPACK)")
endif()

# Only include CPack if at least one generator is enabled
if(CPACK_GENERATORS_ENABLED)
    include(CPack)
    message(STATUS "CPack configuration: ${CPACK_GENERATOR}")
else()
    message(STATUS "CPack disabled (no package generators enabled)")
endif()
