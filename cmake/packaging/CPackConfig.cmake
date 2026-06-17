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
    # Architecture: derive from the build host via `dpkg --print-architecture`
    # instead of hardcoding amd64. The previous hardcode shipped an "amd64"-
    # labeled .deb from the arm64 builder, and apt on arm64 then tried to
    # install it as a foreign arch and failed pulling :amd64 dependencies.
    # CPack falls back to dpkg auto-detection when CPACK_DEBIAN_PACKAGE_ARCHITECTURE
    # is unset.
    find_program(DPKG_CMD dpkg)
    if(DPKG_CMD)
        execute_process(COMMAND ${DPKG_CMD} --print-architecture
            OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    # Runtime dependencies: external shared libraries our binaries NEED.
    # Use the -dev package names (instead of the runtime-only soname
    # packages like libzmq5 / libmsgpack-c2t64) because:
    #   - the runtime package names are unstable across Ubuntu versions
    #     (libmsgpack-c2 vs libmsgpack-c2t64 after the 64-bit time_t
    #     transition; libyaml-cpp0.7 vs libyaml-cpp0.8 between LTS
    #     releases)
    #   - apt resolves -dev → runtime automatically (-dev `Depends:`
    #     the matching versioned soname)
    #   - they exactly match what the build job apt-installs to
    #     COMPILE clio-core, so the .deb's declared deps and the build
    #     deps stay in lockstep
    # Slight overhead: -dev pulls headers the runtime user doesn't
    # need, but Debian convention permits this and the disk-space
    # cost is small. The -dev packages are also the only set we've
    # actually validated in CI.
    set(CPACK_DEBIAN_PACKAGE_DEPENDS
        "libzmq3-dev, libyaml-cpp-dev, libmsgpack-dev, libsodium-dev")
    # The pure-Python context visualizer ships in the package (see
    # CLIO_CORE_ENABLE_VISUALIZER). Declare its Python runtime deps so
    # `context-visualizer` works out of the box after `apt install`.
    if(CLIO_CORE_ENABLE_VISUALIZER)
        set(CPACK_DEBIAN_PACKAGE_DEPENDS
            "${CPACK_DEBIAN_PACKAGE_DEPENDS}, python3, python3-flask, python3-yaml, python3-msgpack")
    endif()

    # Use Debian-standard filename: <pkg>_<ver>-<rel>_<arch>.deb. Without this,
    # CPack defaults to "iowarp-core-<ver>-Linux.deb" with no arch suffix, so
    # the amd64 and arm64 builds both produce the same filename and the second
    # release upload overwrites the first.
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

    message(STATUS "CPack: DEB generator enabled (arch=${CPACK_DEBIAN_PACKAGE_ARCHITECTURE})")
endif()

# RPM Package Configuration
if(CLIO_CORE_ENABLE_RPM_PACKAGE OR CLIO_CORE_ENABLE_CPACK)
    list(APPEND CPACK_GENERATOR "RPM")
    set(CPACK_GENERATORS_ENABLED ON)

    # RPM-specific settings
    set(CPACK_RPM_PACKAGE_LICENSE "MIT")
    set(CPACK_RPM_PACKAGE_GROUP "System/Libraries")
    # External shared-library deps. AUTOREQ is off below (otherwise rpm
    # generates spurious Requires on our OWN internal libs and fails to
    # self-resolve), so list every external lib we link against. Use
    # the -devel package names — those are what the build job dnf-
    # installs to compile clio-core, so they're guaranteed to exist
    # and pull in the matching runtime libs. Runtime-only Fedora
    # package names (e.g. `msgpack-c`) are not consistently published
    # across Fedora versions and were observed to fail with
    # "nothing provides msgpack-c" on Fedora 40.
    set(CPACK_RPM_PACKAGE_REQUIRES
        "zeromq-devel, yaml-cpp-devel, msgpack-devel, libsodium-devel")
    # Python runtime deps for the bundled context visualizer (AUTOREQ is off,
    # so these must be declared explicitly). Fedora names: python3-pyyaml
    # (vs Debian's python3-yaml) and python3-msgpack.
    if(CLIO_CORE_ENABLE_VISUALIZER)
        set(CPACK_RPM_PACKAGE_REQUIRES
            "${CPACK_RPM_PACKAGE_REQUIRES}, python3, python3-flask, python3-pyyaml, python3-msgpack")
    endif()
    # Disable auto-generated Requires on internal libraries. With AUTOREQ
    # default-on, rpmbuild scans every installed .so and adds a
    # Requires: lib<x>.so()(64bit) for each one — including our OWN
    # libclio_admin_client.so / libchimaera_MOD_NAME_*.so / etc. which
    # ARE in the same RPM. The matching Provides: side isn't generated
    # at the same path (sym-version mismatch under the cpack flow),
    # so dnf refuses to install with "nothing provides libclio_*". Turn
    # both directions off and rely on CPACK_RPM_PACKAGE_REQUIRES above
    # for the external deps that actually need declaring (zeromq,
    # yaml-cpp). Internal .so resolution happens at runtime via
    # rpath (`$ORIGIN/../lib`).
    set(CPACK_RPM_PACKAGE_AUTOREQ OFF)
    set(CPACK_RPM_PACKAGE_AUTOPROV OFF)
    # Disable Fedora's brp-check-rpaths spec-file post-policy. CMake bakes
    # an $ORIGIN/../lib rpath into clio_run / clio_cae / clio_cte_bench so
    # they can find their sibling libclio_*.so files at /usr/lib/. Fedora's
    # rpmbuild macros run check-rpaths-worker as part of the install-time
    # %install scriptlet, which sees the $ORIGIN-relative rpath and strips
    # it on the grounds that "binaries in /usr/bin shouldn't need rpath."
    # Net effect: `clio_run --help` exits 127 with
    #   error while loading shared libraries: libclio_admin_client.so:
    #   cannot open shared object file
    # because ld.so falls back to LD_LIBRARY_PATH + ld.so.cache and finds
    # no entry for our libs. Disabling the brp check preserves the same
    # install layout the .deb already uses (which Debian's policy doesn't
    # touch), so both packages now resolve their internal libs identically.
    set(CPACK_RPM_SPEC_MORE_DEFINE "%define __brp_check_rpaths %{nil}")

    # Use RPM-standard filename: <pkg>-<ver>-<rel>.<arch>.rpm. Same rationale
    # as CPACK_DEBIAN_FILE_NAME above — the default "iowarp-core-<ver>-Linux.rpm"
    # has no arch suffix and collides when uploading both amd64 and arm64 RPMs.
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

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
