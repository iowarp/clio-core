# Context Visualizer -- pure Python web dashboard, no compilation.
#
# Installs the context_visualizer package into the target Python's
# site-packages plus a `context-visualizer` launcher on PATH. This runs for
# conda / .deb / .rpm / system installs.
#
# Under scikit-build-core (pip wheels) the package instead ships at the wheel
# root via pyproject.toml's `wheel.packages`, and the `context-visualizer`
# console script is generated from `[project.scripts]`. Installing here too
# would duplicate both, so we bail out early when SKBUILD is defined.
if(DEFINED SKBUILD)
    message(STATUS "Context Visualizer: shipped in the pip wheel via wheel.packages; skipping CMake install")
    return()
endif()

# A Python interpreter is needed only to locate site-packages. When
# CLIO_CORE_ENABLE_PYTHON is ON the root already found one (Python_EXECUTABLE);
# otherwise find just the Interpreter component, which is lightweight and does
# NOT pull in nanobind / Development.Module.
if(NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

# Resolve the interpreter's site-packages directory *relative to its prefix*
# and re-root it under CMAKE_INSTALL_PREFIX. A prefix-relative DESTINATION
# keeps every file inside the install tree so cpack DEB/RPM package them, and
# it adapts automatically to each layout. We pick the distro packaging scheme:
# Debian/Ubuntu expose 'deb_system' -> lib/python3/dist-packages (the only
# dir on the default sys.path there; the standard scheme's site-packages is
# NOT), while Fedora and conda use 'posix_prefix' -> lib/pythonX.Y/site-packages.
execute_process(
    COMMAND ${Python_EXECUTABLE} -c
        "import sysconfig, sys, os; s = 'deb_system' if 'deb_system' in sysconfig.get_scheme_names() else 'posix_prefix'; print(os.path.relpath(sysconfig.get_path('purelib', s), sys.prefix))"
    OUTPUT_VARIABLE _VIS_SITE_REL
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _VIS_SITE_RC)
if(NOT _VIS_SITE_RC EQUAL 0 OR _VIS_SITE_REL STREQUAL "")
    set(_VIS_SITE_REL "lib/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
endif()

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/context_visualizer/
    DESTINATION "${_VIS_SITE_REL}/context_visualizer"
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.html"
        PATTERN "*.css"
        PATTERN "*.js"
)

# `context-visualizer` launcher on PATH. pip generates an equivalent console
# script from pyproject.toml's [project.scripts]; conda/.deb/.rpm get this
# small wrapper instead.
install(
    PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/context-visualizer
    DESTINATION bin
)

message(STATUS "Context Visualizer will be installed to: ${CMAKE_INSTALL_PREFIX}/${_VIS_SITE_REL}/context_visualizer")

# Check for runtime Python dependencies (informational only -- not a build failure)
foreach(_mod flask yaml msgpack)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import ${_mod}"
        RESULT_VARIABLE _rc
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT _rc EQUAL 0)
        list(APPEND _VISUALIZER_MISSING ${_mod})
    endif()
endforeach()
if(_VISUALIZER_MISSING)
    string(REPLACE ";" ", " _VISUALIZER_MISSING_STR "${_VISUALIZER_MISSING}")
    message(STATUS "  NOTE: context_visualizer runtime deps not yet installed: ${_VISUALIZER_MISSING_STR}")
    message(STATUS "  Install with: ${Python_EXECUTABLE} -m pip install flask pyyaml msgpack")
endif()
