# IOWarp Core (CLIO Core) vcpkg portfile.
#
# Two ways to get the source:
#   * a local checkout: set IOWARP_CORE_SOURCE_DIR to it and pass it through
#     with VCPKG_KEEP_ENV_VARS=IOWARP_CORE_SOURCE_DIR. This is what CI does
#     (.github/workflows/vcpkg.yml), so the port is tested against the commit
#     being changed rather than against a release.
#   * GitHub: `vcpkg install iowarp-core --head` builds the dev branch.
#
# Both overlay directories are needed:
#   vcpkg install iowarp-core --head \
#     --overlay-ports=<clio-core>/installers/vcpkg \
#     --overlay-ports=<clio-core>/installers/vcpkg/overlay-ports
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

if(DEFINED ENV{IOWARP_CORE_SOURCE_DIR} AND NOT "$ENV{IOWARP_CORE_SOURCE_DIR}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{IOWARP_CORE_SOURCE_DIR}" SOURCE_PATH)
    if(NOT EXISTS "${SOURCE_PATH}/CMakeLists.txt")
        message(FATAL_ERROR "IOWARP_CORE_SOURCE_DIR=${SOURCE_PATH} has no CMakeLists.txt")
    endif()
    message(STATUS "Building iowarp-core from the local checkout ${SOURCE_PATH}")
else()
    vcpkg_from_github(
        OUT_SOURCE_PATH SOURCE_PATH
        REPO iowarp/clio-core
        REF "v${VERSION}"
        SHA512 0
        HEAD_REF dev
    )
endif()

# viz:  the web dashboard and the S3 / GCS storage tiers (all need Poco).
# hdf5: HDF5 import in the CAE, plus the HDF5 VOL connector and VFD.
vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        viz              CLIO_ENABLE_AMAZON_DRIVE
        viz              CLIO_ENABLE_GOOGLE_CLOUD
        hdf5             CLIO_CORE_ENABLE_HDF5
        hdf5             CLIO_CTE_ENABLE_HDF5_VOL
        hdf5             CLIO_CTE_ENABLE_VFD
        boost-coroutines CLIO_CORE_ENABLE_BOOST_COROUTINES
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        -DCLIO_CORE_ENABLE_TESTS=OFF
        -DCLIO_CORE_ENABLE_BENCHMARKS=OFF
        -DCLIO_CORE_ENABLE_PYTHON=OFF
        -DCLIO_CORE_ENABLE_DOXYGEN=OFF
        -DCLIO_CORE_ENABLE_COVERAGE=OFF
        -DCLIO_CORE_ENABLE_ASAN=OFF
        -DCLIO_CORE_ENABLE_PCH=OFF
        -DCLIO_CORE_ENABLE_CCACHE=OFF
        -DCLIO_CORE_SEED_USER_CONFIG=OFF   # never write ~/.clio from a port build
)

vcpkg_cmake_install()

# The install rules also serve the pip wheel: they drop iowarp_core.pth one
# level above the prefix (the wheel's site-packages root) and, on Windows,
# copy every vcpkg dependency DLL into bin/ so the wheel is self-contained.
# In a vcpkg install those DLLs belong to their own ports, so remove the
# copies (vcpkg refuses two packages owning one file) and the .pth.
file(REMOVE "${CURRENT_PACKAGES_DIR}/iowarp_core.pth"
            "${CURRENT_PACKAGES_DIR}/debug/iowarp_core.pth")
foreach(_cfg_dir IN ITEMS "" "debug/")
    file(GLOB _iowarp_dlls "${CURRENT_PACKAGES_DIR}/${_cfg_dir}bin/*.dll")
    foreach(_dll IN LISTS _iowarp_dlls)
        get_filename_component(_dll_name "${_dll}" NAME)
        if(EXISTS "${CURRENT_INSTALLED_DIR}/${_cfg_dir}bin/${_dll_name}")
            file(REMOVE "${_dll}")
        endif()
    endforeach()
endforeach()

# Every CMake package the build installs lives in lib/cmake/<name>; move each
# to share/<name> the vcpkg way. Listing them here instead would silently
# miss packages the build adds later.
file(GLOB _iowarp_cmake_dirs LIST_DIRECTORIES true "${CURRENT_PACKAGES_DIR}/lib/cmake/*")
foreach(_dir IN LISTS _iowarp_cmake_dirs)
    if(IS_DIRECTORY "${_dir}")
        get_filename_component(_pkg "${_dir}" NAME)
        vcpkg_cmake_config_fixup(PACKAGE_NAME "${_pkg}" CONFIG_PATH "lib/cmake/${_pkg}"
                                 DO_NOT_DELETE_PARENT_CONFIG_PATH)
    endif()
endforeach()
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/cmake" "${CURRENT_PACKAGES_DIR}/debug/lib/cmake")

# Executables (clio_run and friends) go to tools/<port>.
if(VCPKG_TARGET_IS_WINDOWS)
    file(GLOB _iowarp_exes "${CURRENT_PACKAGES_DIR}/bin/*.exe")
else()
    file(GLOB _iowarp_exes LIST_DIRECTORIES false "${CURRENT_PACKAGES_DIR}/bin/*")
endif()
set(_iowarp_tool_names "")
foreach(_exe IN LISTS _iowarp_exes)
    get_filename_component(_name "${_exe}" NAME)
    string(REGEX REPLACE "\\.exe$" "" _name "${_name}")
    list(APPEND _iowarp_tool_names "${_name}")
endforeach()
if(_iowarp_tool_names)
    vcpkg_copy_tools(TOOL_NAMES ${_iowarp_tool_names} AUTO_CLEAN)
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
