# IOWarp Core vcpkg portfile
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO iowarp/core
    REF "${VERSION}"
    SHA512 0
    HEAD_REF main
)

# Configure CMake
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DWRP_CORE_ENABLE_TESTS=OFF
        -DWRP_CORE_ENABLE_BENCHMARKS=OFF
        -DWRP_CORE_ENABLE_PYTHON=OFF
        -DWRP_CORE_ENABLE_DOXYGEN=OFF
        -DWRP_CORE_ENABLE_COVERAGE=OFF
        -DWRP_CORE_ENABLE_ASAN=OFF
        -DHSHM_ENABLE_TESTS=OFF
        -DHSHM_ENABLE_BENCHMARKS=OFF
        -DCHIMAERA_ENABLE_TESTS=OFF
        -DCHIMAERA_ENABLE_BENCHMARKS=OFF
)

# Build
vcpkg_cmake_install()

# Fix CMake config paths
vcpkg_cmake_config_fixup(PACKAGE_NAME iowarp-core CONFIG_PATH lib/cmake/iowarp-core)
vcpkg_cmake_config_fixup(PACKAGE_NAME HermesShm CONFIG_PATH lib/cmake/HermesShm)
vcpkg_cmake_config_fixup(PACKAGE_NAME chimaera CONFIG_PATH lib/cmake/chimaera)
vcpkg_cmake_config_fixup(PACKAGE_NAME chimaera_admin CONFIG_PATH lib/cmake/chimaera_admin)
vcpkg_cmake_config_fixup(PACKAGE_NAME chimaera_bdev CONFIG_PATH lib/cmake/chimaera_bdev)

# Handle optional ChiMod packages
if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/cmake/wrp_cte_core")
    vcpkg_cmake_config_fixup(PACKAGE_NAME wrp_cte_core CONFIG_PATH lib/cmake/wrp_cte_core)
endif()
if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/cmake/wrp_cae_core")
    vcpkg_cmake_config_fixup(PACKAGE_NAME wrp_cae_core CONFIG_PATH lib/cmake/wrp_cae_core)
endif()
if(EXISTS "${CURRENT_PACKAGES_DIR}/lib/cmake/wrp_cee_core")
    vcpkg_cmake_config_fixup(PACKAGE_NAME wrp_cee_core CONFIG_PATH lib/cmake/wrp_cee_core)
endif()

# Remove duplicate files from debug
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Install license
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

# Copy usage file
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
