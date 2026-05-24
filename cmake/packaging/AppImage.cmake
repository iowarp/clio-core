#------------------------------------------------------------------------------
# AppImage Configuration
# Builds a portable AppImage for IOWarp Core
#------------------------------------------------------------------------------

# Define AppImage staging directory
set(APPIMAGE_STAGING_DIR "${CMAKE_BINARY_DIR}/AppDir")

# Configure the appimagetool driver script so shell variables ($APPIMAGETOOL,
# $@) never end up unescaped in Ninja rule files.
configure_file(
    "${CMAKE_CURRENT_LIST_DIR}/build_appimage.sh.in"
    "${CMAKE_BINARY_DIR}/build_appimage.sh"
    @ONLY
)

# Pre-generate the AppRun entry-point outside AppDir so the build step can
# copy it without needing bash to write shell-variable literals.
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/AppRun.generated"
    CONTENT "#!/bin/bash
exec \"$APPDIR/usr/bin/clio_run\" \"$@\"
")

#------------------------------------------------------------------------------
# Target: appimage-stage
# Stages the installation to AppDir structure for AppImage
#------------------------------------------------------------------------------
add_custom_target(appimage-stage
    COMMENT "Staging AppImage directory: ${APPIMAGE_STAGING_DIR}"
    # Install project into AppDir/usr
    COMMAND ${CMAKE_COMMAND} -E make_directory ${APPIMAGE_STAGING_DIR}/usr
    COMMAND ${CMAKE_COMMAND} --install ${CMAKE_BINARY_DIR} --prefix ${APPIMAGE_STAGING_DIR}/usr --component Unspecified
    # .desktop file
    COMMAND ${CMAKE_COMMAND} -E make_directory ${APPIMAGE_STAGING_DIR}/usr/share/applications
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_LIST_DIR}/clio_run.desktop
        ${APPIMAGE_STAGING_DIR}/usr/share/applications/clio_run.desktop
    # appimagetool also requires the .desktop at the AppDir root
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_CURRENT_LIST_DIR}/clio_run.desktop
        ${APPIMAGE_STAGING_DIR}/clio_run.desktop
    # Placeholder icon (required by appimagetool)
    COMMAND ${CMAKE_COMMAND} -E make_directory
        ${APPIMAGE_STAGING_DIR}/usr/share/icons/hicolor/256x256/apps
    COMMAND ${CMAKE_COMMAND} -E touch
        ${APPIMAGE_STAGING_DIR}/usr/share/icons/hicolor/256x256/apps/clio_run.png
    COMMAND ${CMAKE_COMMAND} -E touch
        ${APPIMAGE_STAGING_DIR}/clio_run.png
    # AppRun entry point
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/AppRun.generated
        ${APPIMAGE_STAGING_DIR}/AppRun
    COMMAND chmod +x ${APPIMAGE_STAGING_DIR}/AppRun
)

#------------------------------------------------------------------------------
# Target: appimage
# Builds the final AppImage using appimagetool
#------------------------------------------------------------------------------
add_custom_target(appimage
    COMMENT "Building AppImage: ${CMAKE_BINARY_DIR}/iowarp-core-${PROJECT_VERSION}-x86_64.AppImage"
    DEPENDS appimage-stage
    COMMAND bash ${CMAKE_BINARY_DIR}/build_appimage.sh
)

message(STATUS "AppImage target registered — run: cmake --build . --target appimage")
