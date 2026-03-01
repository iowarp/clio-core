# Context Visualizer -- pure Python, no compilation
# Installs the context_visualizer package to site-packages when WRP_CORE_ENABLE_PYTHON=ON
# PYTHON_SITE_PACKAGES is set by the root CMakeLists.txt (absolute path).

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/context_visualizer/
    DESTINATION "${PYTHON_SITE_PACKAGES}/context_visualizer"
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.html"
        PATTERN "*.css"
        PATTERN "*.js"
)

message(STATUS "Context Visualizer will be installed to: ${PYTHON_SITE_PACKAGES}/context_visualizer")

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
